import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from collections import defaultdict

DATASET_ROOT = r"C:\Users\hamsa\OneDrive\Desktop\ML and DL\Genik project\Wild Animal detection using YOLO\elephant-dataset-yolov"
NUM_CLASSES = 1
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Dataset root: {DATASET_ROOT}")

class YOLOv13(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(YOLOv13, self).__init__()
        self.num_classes = num_classes
        
        # Simplified backbone
        self.backbone = torch.nn.Sequential(
            # Stem
            torch.nn.Conv2d(3, 32, 6, 2, 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.SiLU(),
            
            # Stage 1
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
            self._make_c3k2_block(64, 64, 2),
            
            # Stage 2
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),
            self._make_c3k2_block(128, 128, 3),
            
            # Stage 3
            torch.nn.Conv2d(128, 256, 3, 2, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
            self._make_c3k2_block(256, 256, 3),
            
            # Stage 4
            torch.nn.Conv2d(256, 512, 3, 2, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.SiLU(),
            self._make_c3k2_block(512, 512, 2),
            
            # SPPF
            self._make_sppf_block(512, 512),
        )
        
        # Detection heads - one for each scale
        self.head_large = torch.nn.Conv2d(512, (5 + num_classes) * 3, 1, 1, 0)  # Large objects
        self.head_medium = torch.nn.Conv2d(256, (5 + num_classes) * 3, 1, 1, 0)  # Medium objects  
        self.head_small = torch.nn.Conv2d(128, (5 + num_classes) * 3, 1, 1, 0)  # Small objects
        
        # Neck layers for feature fusion
        self.conv_512_256 = torch.nn.Conv2d(512, 256, 1, 1, 0)
        self.conv_256_128 = torch.nn.Conv2d(256, 128, 1, 1, 0)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        
        self._initialize_weights()
    
    def _make_c3k2_block(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                torch.nn.Conv2d(in_channels, out_channels//2, 1, 1, 0),
                torch.nn.BatchNorm2d(out_channels//2),
                torch.nn.SiLU(),
                torch.nn.Conv2d(out_channels//2, out_channels//2, 3, 1, 1, groups=out_channels//2),
                torch.nn.BatchNorm2d(out_channels//2),
                torch.nn.SiLU(),
                torch.nn.Conv2d(out_channels//2, out_channels, 1, 1, 0),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.SiLU()
            ])
            in_channels = out_channels
        return torch.nn.Sequential(*layers)
    
    def _make_sppf_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels//2, 1, 1, 0),
            torch.nn.BatchNorm2d(out_channels//2),
            torch.nn.SiLU(),
            torch.nn.MaxPool2d(5, 1, 2),
            torch.nn.Conv2d(out_channels//2, out_channels, 1, 1, 0),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.SiLU()
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract multi-scale features
        features = []
        
        # Stage 1: 64 channels
        x = self.backbone[:7](x)
        feat_128 = x  # Save for small objects
        features.append(feat_128)
        
        # Stage 2: 128 channels  
        x = self.backbone[7:11](x)
        feat_256 = x  # Save for medium objects
        features.append(feat_256)
        
        # Stage 3: 256 channels
        x = self.backbone[11:15](x)
        feat_512_pre = x
        features.append(feat_512_pre)
        
        # Stage 4: 512 channels + SPPF
        x = self.backbone[15:](x)
        feat_512 = x  # For large objects
        features.append(feat_512)
        
        outputs = []
        
        # Large objects detection (lowest resolution, highest channels)
        out_large = self.head_large(feat_512)
        B, C, H, W = out_large.shape
        out_large = out_large.view(B, 3, 5 + self.num_classes, H, W).permute(0, 1, 3, 4, 2)
        outputs.append(out_large)
        
        # Medium objects detection
        feat_up = self.conv_512_256(feat_512)
        feat_up = self.upsample(feat_up)
        feat_medium = feat_up + feat_512_pre  # Skip connection
        out_medium = self.head_medium(feat_medium)
        B, C, H, W = out_medium.shape
        out_medium = out_medium.view(B, 3, 5 + self.num_classes, H, W).permute(0, 1, 3, 4, 2)
        outputs.append(out_medium)
        
        # Small objects detection  
        feat_up = self.conv_256_128(feat_medium)
        feat_up = self.upsample(feat_up)
        feat_small = feat_up + feat_256  # Skip connection
        out_small = self.head_small(feat_small)
        B, C, H, W = out_small.shape
        out_small = out_small.view(B, 3, 5 + self.num_classes, H, W).permute(0, 1, 3, 4, 2)
        outputs.append(out_small)
        
        return outputs

class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, box_weight=1.0, obj_weight=1.0, cls_weight=1.0):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets):
        total_loss = 0
        num_samples = 0
        
        for pred in predictions:
            if len(targets) == 0:
                continue
                
            B, A, H, W, C = pred.shape
            pred = pred.view(-1, C)
            
            for target in targets:
                if len(target) == 0:
                    continue
                    
                target_tensor = torch.zeros((B * A * H * W, C), device=pred.device)
                
                for gt in target:
                    if len(gt) >= 5:
                        cls, x, y, w, h = gt[:5]
                        
                        grid_x = int(x * W)
                        grid_y = int(y * H)
                        
                        if 0 <= grid_x < W and 0 <= grid_y < H:
                            idx = grid_y * W + grid_x
                            
                            target_tensor[idx, 0] = 1.0
                            target_tensor[idx, 1:5] = torch.tensor([x, y, w, h])
                            if C > 5:
                                target_tensor[idx, 5 + int(cls)] = 1.0
                
                obj_loss = self.bce_loss(pred[:, 0], target_tensor[:, 0]).mean()
                
                box_mask = target_tensor[:, 0] > 0
                if box_mask.sum() > 0:
                    box_loss = self.mse_loss(pred[box_mask, 1:5], target_tensor[box_mask, 1:5]).mean()
                else:
                    box_loss = torch.tensor(0.0, device=pred.device)
                
                if C > 5:
                    cls_loss = self.bce_loss(pred[:, 5:], target_tensor[:, 5:]).mean()
                else:
                    cls_loss = torch.tensor(0.0, device=pred.device)
                
                total_loss += (self.obj_weight * obj_loss + 
                              self.box_weight * box_loss + 
                              self.cls_weight * cls_loss)
                num_samples += 1
        
        return total_loss / max(num_samples, 1)

train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
    A.RandomScale(scale_limit=0.1, p=0.4),  # Reduced scale limit to prevent size issues
    A.HueSaturationValue(
        hue_shift_limit=15,
        sat_shift_limit=30,
        val_shift_limit=30,
        p=0.7
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.7
    ),
    A.GaussNoise(noise_scale_factor=0.1, p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.2),
    A.CoarseDropout(
        num_holes_range=(3, 10),
        hole_height_range=(10, 40),
        hole_width_range=(10, 40),
        p=0.4
    ),
    A.OneOf([
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
    ], p=0.4),
    A.GridDropout(ratio=0.2, random_offset=True, p=0.3),
    A.Resize(IMG_SIZE, IMG_SIZE),  # Final resize to ensure consistent dimensions
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.2
))

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
))

def calculate_iou_vectorized(boxes1, boxes2):
    x1_inter = np.maximum(boxes1[:, 0], boxes2[:, 0])
    y1_inter = np.maximum(boxes1[:, 1], boxes2[:, 1])
    x2_inter = np.minimum(boxes1[:, 2], boxes2[:, 2])
    y2_inter = np.minimum(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / (union_area + 1e-8)

def calculate_map_precision_recall_f1(predictions_list, ground_truths_list, iou_thresholds=[0.5], conf_threshold=0.25):
    metrics = {}
    
    for iou_thresh in iou_thresholds:
        all_predictions = []
        all_ground_truths = []
        
        for preds, gts in zip(predictions_list, ground_truths_list):
            filtered_preds = [pred for pred in preds if len(pred) >= 5 and pred[4] >= conf_threshold]
            all_predictions.extend(filtered_preds)
            all_ground_truths.extend(gts)
        
        if len(all_predictions) == 0:
            if len(all_ground_truths) == 0:
                metrics[f'mAP_{iou_thresh}'] = 1.0
                metrics[f'precision_{iou_thresh}'] = 1.0
                metrics[f'recall_{iou_thresh}'] = 1.0
                metrics[f'f1_{iou_thresh}'] = 1.0
            else:
                metrics[f'mAP_{iou_thresh}'] = 0.0
                metrics[f'precision_{iou_thresh}'] = 0.0
                metrics[f'recall_{iou_thresh}'] = 0.0
                metrics[f'f1_{iou_thresh}'] = 0.0
            continue
        
        predictions_sorted = sorted(all_predictions, key=lambda x: x[4], reverse=True)
        
        tp = np.zeros(len(predictions_sorted))
        fp = np.zeros(len(predictions_sorted))
        gt_matched = np.zeros(len(all_ground_truths), dtype=bool)
        
        for pred_idx, pred in enumerate(predictions_sorted):
            if len(all_ground_truths) == 0:
                fp[pred_idx] = 1
                continue
                
            pred_box = np.array([pred[:4]])
            gt_boxes = np.array([gt[:4] for gt in all_ground_truths])
            
            ious = calculate_iou_vectorized(pred_box, gt_boxes).flatten()
            
            best_gt_idx = np.argmax(ious)
            best_iou = ious[best_gt_idx]
            
            if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recalls = tp_cumsum / (len(all_ground_truths) + 1e-8)
        
        precisions = np.concatenate(([0], precisions, [0]))
        recalls = np.concatenate(([0], recalls, [1]))
        
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        map_score = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        final_precision = precisions[-2] if len(precisions) > 1 else 0.0
        final_recall = recalls[-2] if len(recalls) > 1 else 0.0
        final_f1 = (2 * final_precision * final_recall / (final_precision + final_recall + 1e-8))
        
        metrics[f'mAP_{iou_thresh}'] = map_score
        metrics[f'precision_{iou_thresh}'] = final_precision
        metrics[f'recall_{iou_thresh}'] = final_recall
        metrics[f'f1_{iou_thresh}'] = final_f1
    
    return metrics

def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append([x, y, w, h, cls])
    return boxes

def letterbox_resize(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def non_max_suppression(boxes, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_others = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        union = area_i + area_others - inter
        
        iou = inter / (union + 1e-8)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def collate_fn(batch):
    imgs, targets, names = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, list(targets), list(names)

class ElephantDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.transforms = transforms
        
        self.img_paths = sorted([
            p for p in self.images_dir.iterdir() 
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        print(f"Found {len(self.img_paths)} images in {images_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]
        
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = load_yolo_labels(label_path, w0, h0)
        
        class_labels = [int(box[4]) if len(box) > 4 else 0 for box in boxes]
        yolo_boxes = [box[:4] for box in boxes]
        
        if self.transforms:
            try:
                transformed = self.transforms(
                    image=img_rgb,
                    bboxes=yolo_boxes,
                    class_labels=class_labels
                )
                
                img_tensor = transformed['image']
                transformed_boxes = transformed['bboxes']
                transformed_labels = transformed['class_labels']
                
                targets = []
                for box, label in zip(transformed_boxes, transformed_labels):
                    if len(box) == 4:
                        targets.append([float(label)] + [float(b) for b in box])
                
                target_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))
                
            except Exception as e:
                print(f"Transform failed for {img_path}: {e}")
                img_resized, _, _ = letterbox_resize(img, (self.img_size, self.img_size))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1)))
                
                targets = []
                for box in boxes:
                    if len(box) >= 4:
                        targets.append([0.0] + [float(b) for b in box[:4]])
                
                target_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))
        else:
            img_resized, _, _ = letterbox_resize(img, (self.img_size, self.img_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1)))
            
            targets = []
            for box in boxes:
                if len(box) >= 4:
                    targets.append([0.0] + [float(b) for b in box[:4]])
            
            target_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))
        
        return img_tensor, target_tensor, img_path.name

def evaluate_model(model, dataloader, conf_threshold=0.25, iou_threshold=0.5):
    model.eval()
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for imgs, targets, names in dataloader:
            imgs = imgs.to(DEVICE)
            predictions = model(imgs)
            
            batch_preds = []
            batch_gts = []
            
            for i, (preds, target) in enumerate(zip(predictions, targets)):
                img_predictions = []
                img_ground_truths = []
                
                if isinstance(preds, list):
                    for pred_scale in preds:
                        pred_np = pred_scale[i].detach().cpu().numpy()
                        
                        if len(pred_np.shape) == 4:
                            B, H, W, C = pred_np.shape
                            for h in range(H):
                                for w in range(W):
                                    for a in range(B):
                                        if pred_np[a, h, w, 0] > conf_threshold:
                                            conf = pred_np[a, h, w, 0]
                                            x = pred_np[a, h, w, 1]
                                            y = pred_np[a, h, w, 2]
                                            width = pred_np[a, h, w, 3]
                                            height = pred_np[a, h, w, 4]
                                            
                                            x1 = (x - width/2) * IMG_SIZE
                                            y1 = (y - height/2) * IMG_SIZE
                                            x2 = (x + width/2) * IMG_SIZE
                                            y2 = (y + height/2) * IMG_SIZE
                                            
                                            img_predictions.append([x1, y1, x2, y2, conf])
                
                target_np = target.numpy()
                for gt in target_np:
                    if len(gt) >= 5 and gt[0] >= 0:
                        _, x_center, y_center, width, height = gt[:5]
                        x1 = (x_center - width/2) * IMG_SIZE
                        y1 = (y_center - height/2) * IMG_SIZE
                        x2 = (x_center + width/2) * IMG_SIZE
                        y2 = (y_center + height/2) * IMG_SIZE
                        img_ground_truths.append([x1, y1, x2, y2])
                
                if img_predictions:
                    keep_indices = non_max_suppression(img_predictions, iou_threshold)
                    img_predictions = [img_predictions[i] for i in keep_indices]
                
                batch_preds.append(img_predictions)
                batch_gts.append(img_ground_truths)
            
            all_predictions.extend(batch_preds)
            all_ground_truths.extend(batch_gts)
    
    metrics = calculate_map_precision_recall_f1(all_predictions, all_ground_truths, [0.5, 0.75], conf_threshold)
    
    return {
        'mAP@0.5': metrics.get('mAP_0.5', 0.0),
        'mAP@0.75': metrics.get('mAP_0.75', 0.0),
        'precision': metrics.get('precision_0.5', 0.0),
        'recall': metrics.get('recall_0.5', 0.0),
        'f1_score': metrics.get('f1_0.5', 0.0)
    }

def train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    print("Starting Elephant Detection Training with YOLOv13")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Device: {DEVICE}")
    
    train_images = Path(DATASET_ROOT) / 'train' / 'images'
    train_labels = Path(DATASET_ROOT) / 'train' / 'labels'
    val_images = Path(DATASET_ROOT) / 'valid' / 'images'
    val_labels = Path(DATASET_ROOT) / 'valid' / 'labels'
    
    train_dataset = ElephantDataset(
        str(train_images), str(train_labels), 
        img_size=IMG_SIZE, transforms=train_transforms
    )
    val_dataset = ElephantDataset(
        str(val_images), str(val_labels), 
        img_size=IMG_SIZE, transforms=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    model = YOLOv13(num_classes=NUM_CLASSES).to(DEVICE)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = HybridLoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    best_map = 0.0
    training_history = []
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets, names) in enumerate(train_loader):
            images = images.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 20 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f}")
        
        avg_train_loss = epoch_loss / max(num_batches, 1)
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets, names in val_loader:
                images = images.to(DEVICE)
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
        print(f"Computing evaluation metrics...")
        metrics = evaluate_model(model, val_loader)
        
        current_map = metrics['mAP@0.5']
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  mAP@0.5: {current_map:.4f}")
        print(f"  mAP@0.75: {metrics['mAP@0.75']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'mAP_0.5': current_map,
            'mAP_0.75': metrics['mAP@0.75'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'metrics': metrics,
            'training_history': training_history
        }
        
        torch.save(checkpoint, CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth')
        
        if current_map > best_map:
            best_map = current_map
            torch.save(checkpoint, CHECKPOINT_DIR / 'best_model.pth')
            print(f"  New best model saved! mAP@0.5: {best_map:.4f}")
        
        scheduler.step()
        
        with open(CHECKPOINT_DIR / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best mAP@0.5 achieved: {best_map:.4f}")
    print(f"Models saved in: {CHECKPOINT_DIR}")
    
    return model, training_history

def inference(image_path, checkpoint_path=None, conf_threshold=0.25, save_result=True):
    print(f"Running inference on: {image_path}")
    print(f"Confidence threshold: {conf_threshold}")
    
    model = YOLOv13(num_classes=NUM_CLASSES).to(DEVICE)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"Model mAP@0.5: {checkpoint['metrics'].get('mAP@0.5', 'N/A'):.4f}")
    else:
        print("Using randomly initialized weights")
    
    model.eval()
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_image = image.copy()
    processed_image, ratio, padding = letterbox_resize(image, (IMG_SIZE, IMG_SIZE))
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    processed_image = processed_image.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(np.transpose(processed_image, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(input_tensor)
    
    detections = []
    for pred_scale in predictions:
        pred_np = pred_scale[0].detach().cpu().numpy()
        
        if len(pred_np.shape) == 4:
            B, H, W, C = pred_np.shape
            for h in range(H):
                for w in range(W):
                    for a in range(B):
                        if pred_np[a, h, w, 0] > conf_threshold:
                            conf = pred_np[a, h, w, 0]
                            x = pred_np[a, h, w, 1]
                            y = pred_np[a, h, w, 2]
                            width = pred_np[a, h, w, 3]
                            height = pred_np[a, h, w, 4]
                            
                            x1 = (x - width/2) * IMG_SIZE
                            y1 = (y - height/2) * IMG_SIZE
                            x2 = (x + width/2) * IMG_SIZE
                            y2 = (y + height/2) * IMG_SIZE
                            
                            detections.append([x1, y1, x2, y2, conf])
    
    if len(detections) == 0:
        print("No elephants detected")
        return []
    
    keep_indices = non_max_suppression(detections, iou_threshold=0.45)
    final_detections = [detections[i] for i in keep_indices]
    
    print(f"Found {len(final_detections)} elephant(s)")
    
    for i, detection in enumerate(final_detections):
        x1, y1, x2, y2, conf = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(original_image, f'Elephant {conf:.2f}', (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"  Detection {i+1}: Confidence={conf:.3f}, BBox=({x1},{y1},{x2},{y2})")
    
    if save_result:
        output_dir = Path('inference_results')
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"{Path(image_path).stem}_result.jpg"
        cv2.imwrite(str(output_path), original_image)
        print(f"Result saved to: {output_path}")
    
    return final_detections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv13 Elephant Detection System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--inference', action='store_true', help='Run inference')
    parser.add_argument('--image', type=str, help='Path to image for inference')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    elif args.inference:
        if not args.image:
            print("Please provide --image path for inference")
        else:
            inference(
                args.image,
                checkpoint_path=args.checkpoint,
                conf_threshold=args.conf_threshold
            )
    else:
        print("YOLOv13 Elephant Detection System")
        print("\nUsage:")
        print("  Training: python train_yolo.py --train")
        print("  Inference: python train_yolo.py --inference --image path/to/image.jpg --checkpoint path/to/model.pth")
        print("\nKey Features:")
        print("  - YOLOv13 architecture with C3k2 blocks")
        print("  - Comprehensive data augmentation")
        print("  - mAP, Precision, Recall, F1-Score metrics")
        print("  - SGD optimizer with Cosine Annealing")
        print("  - Advanced NMS and evaluation")
