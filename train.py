import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import json
from collections import defaultdict
import glob
import yaml

from model import YOLOv13
from loss_functions import HybridLoss

DATASET_YAML = r"C:\Users\hamsa\OneDrive\Desktop\ML and DL\Genik project\Wild Animal detection using YOLO\elephant-dataset-yolov\dataset.yaml"

def load_dataset_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

dataset_config = load_dataset_config(DATASET_YAML)

DATASET_ROOT = Path(dataset_config['train']).parent.parent
TRAIN_IMAGES = dataset_config['train']
TRAIN_LABELS = TRAIN_IMAGES.replace('images', 'labels')
VAL_IMAGES = dataset_config['val']
VAL_LABELS = VAL_IMAGES.replace('images', 'labels')
TEST_IMAGES = dataset_config['test']
TEST_LABELS = TEST_IMAGES.replace('images', 'labels')

NUM_CLASSES = dataset_config['nc']
CLASS_NAMES = dataset_config['names']

IMG_SIZE = 640
BATCH_SIZE = 4
EPOCHS = 80
LEARNING_RATE = 5e-5  # Ultra-low for HybridLoss stability
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

calculated_mean = [0.40454071177538753, 0.41537247881627276, 0.3674748984405444]
calculated_std = [0.2783741618388076, 0.2825628982753976, 0.2717806466121383]

print(f"Using device: {DEVICE}")
print(f"Dataset root: {DATASET_ROOT}")
print(f"Classes: {CLASS_NAMES}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Using STABILIZED HybridLoss with NaN protection")
print(f"Learning rate: {LEARNING_RATE} (ultra-low for stability)")

def validate_image(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False
        return True
    except:
        return False

# Minimal transforms for stability
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
    A.Normalize(mean=calculated_mean, std=calculated_std),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.4
))

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=calculated_mean, std=calculated_std),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
))

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_ap(predictions, ground_truths, iou_threshold=0.5):
    if len(predictions) == 0:
        return 0.0 if len(ground_truths) > 0 else 1.0
    
    if len(ground_truths) == 0:
        return 0.0
    
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    gt_matched = np.zeros(len(ground_truths))
    
    for pred_idx, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
                
            iou = calculate_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = 1
        else:
            fp[pred_idx] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def calculate_metrics(predictions_list, ground_truths_list, iou_thresholds=[0.5, 0.75]):
    metrics = {}
    
    for iou_thresh in iou_thresholds:
        all_predictions = []
        all_ground_truths = []
        
        for preds, gts in zip(predictions_list, ground_truths_list):
            all_predictions.extend(preds)
            all_ground_truths.extend(gts)
        
        if len(all_predictions) == 0 and len(all_ground_truths) == 0:
            metrics[f'mAP_{iou_thresh}'] = 1.0
            metrics[f'precision_{iou_thresh}'] = 1.0
            metrics[f'recall_{iou_thresh}'] = 1.0
            metrics[f'f1_{iou_thresh}'] = 1.0
            continue
        
        if len(all_predictions) == 0:
            metrics[f'mAP_{iou_thresh}'] = 0.0
            metrics[f'precision_{iou_thresh}'] = 0.0
            metrics[f'recall_{iou_thresh}'] = 0.0
            metrics[f'f1_{iou_thresh}'] = 0.0
            continue
        
        ap = calculate_ap(all_predictions, all_ground_truths, iou_thresh)
        
        predictions_sorted = sorted(all_predictions, key=lambda x: x[4], reverse=True)
        
        tp = 0
        fp = 0
        gt_matched = [False] * len(all_ground_truths)
        
        for pred in predictions_sorted:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(all_ground_truths):
                if gt_matched[gt_idx]:
                    continue
                iou = calculate_iou(pred[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_thresh and best_gt_idx != -1:
                tp += 1
                gt_matched[best_gt_idx] = 1
            else:
                fp += 1
        
        fn = len(all_ground_truths) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'mAP_{iou_thresh}'] = ap
        metrics[f'precision_{iou_thresh}'] = precision
        metrics[f'recall_{iou_thresh}'] = recall
        metrics[f'f1_{iou_thresh}'] = f1_score
    
    return metrics

def load_yolo_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        # Stricter validation for coordinates
                        x = max(0.05, min(0.95, x))
                        y = max(0.05, min(0.95, y))
                        w = max(0.02, min(0.9, w))
                        h = max(0.02, min(0.9, h))
                        boxes.append([x, y, w, h])
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return []
    
    return boxes

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
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
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    scores = boxes[:, 4]
    
    areas = (x2 - x1) * (y2 - y1)
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
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def collate_fn(batch):
    imgs, targets, names = zip(*batch)
    imgs = torch.stack(imgs, 0)
    
    device_targets = []
    for target in targets:
        if len(target) > 0:
            device_targets.append(target.to(imgs.device))
        else:
            device_targets.append(torch.zeros((0, 5), device=imgs.device))
    
    return imgs, device_targets, list(names)

class ElephantDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.transforms = transforms
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_img_paths = []
        
        for ext in image_extensions:
            all_img_paths.extend(self.images_dir.glob(f"*{ext}"))
            all_img_paths.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        self.img_paths = []
        print(f"Validating {len(all_img_paths)} images...")
        
        for img_path in all_img_paths:
            if validate_image(img_path):
                self.img_paths.append(img_path)
        
        self.img_paths = sorted(self.img_paths)
        print(f"Found {len(self.img_paths)} valid images")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h0, w0 = img_rgb.shape[:2]
            
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            boxes = load_yolo_labels(label_path)
            class_labels = [0] * len(boxes)
            
            if self.transforms:
                try:
                    transformed = self.transforms(
                        image=img_rgb,
                        bboxes=boxes,
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
                    img_resized, _, _ = letterbox(img, (self.img_size, self.img_size))
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_normalized = img_rgb.astype(np.float32) / 255.0
                    img_normalized = (img_normalized - np.array(calculated_mean)) / np.array(calculated_std)
                    img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1)))
                    
                    targets = [[0.0] + list(box) for box in boxes]
                    target_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))
            else:
                img_resized, _, _ = letterbox(img, (self.img_size, self.img_size))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                img_normalized = (img_normalized - np.array(calculated_mean)) / np.array(calculated_std)
                img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1)))
                
                targets = [[0.0] + list(box) for box in boxes]
                target_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))
            
            return img_tensor, target_tensor, img_path.name
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            dummy_img = torch.zeros((3, self.img_size, self.img_size))
            dummy_target = torch.zeros((0, 5))
            return dummy_img, dummy_target, img_path.name

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
            
            for i, target in enumerate(targets):
                img_predictions = []
                img_ground_truths = []
                
                for scale_idx, pred_scale in enumerate(predictions):
                    pred = pred_scale[i].detach().cpu().numpy()
                    
                    if len(pred.shape) == 3:
                        C, H, W = pred.shape
                        for h in range(H):
                            for w in range(W):
                                conf = torch.sigmoid(torch.tensor(pred[0, h, w])).item()
                                if conf > conf_threshold:
                                    x = torch.sigmoid(torch.tensor(pred[1, h, w])).item()
                                    y = torch.sigmoid(torch.tensor(pred[2, h, w])).item()
                                    width = torch.exp(torch.tensor(pred[3, h, w])).item()
                                    height = torch.exp(torch.tensor(pred[4, h, w])).item()
                                    
                                    x = (x + w) / W
                                    y = (y + h) / H
                                    width = width / IMG_SIZE * (2 ** scale_idx)
                                    height = height / IMG_SIZE * (2 ** scale_idx)
                                    
                                    x1 = (x - width/2) * IMG_SIZE
                                    y1 = (y - height/2) * IMG_SIZE
                                    x2 = (x + width/2) * IMG_SIZE
                                    y2 = (y + height/2) * IMG_SIZE
                                    
                                    img_predictions.append([x1, y1, x2, y2, conf])
                
                target_np = target.cpu().numpy()
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
    
    metrics = calculate_metrics(all_predictions, all_ground_truths, [0.5, 0.75])
    
    return {
        'mAP@0.5': metrics.get('mAP_0.5', 0.0),
        'mAP@0.75': metrics.get('mAP_0.75', 0.0),
        'precision': metrics.get('precision_0.5', 0.0),
        'recall': metrics.get('recall_0.5', 0.0),
        'f1_score': metrics.get('f1_0.5', 0.0)
    }

def train_model():
    print("Starting YOLOv13 Elephant Detection Training")
    print("=" * 70)
    print(f"Architecture: CSPDarkNet53 + BiFPN + Multi-scale Head")
    print(f"Optimizer: Adam with lr={LEARNING_RATE}")
    print(f"Loss: STABILIZED Hybrid Loss (Focal + CIoU/DIoU + Tweak)")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    train_dataset = ElephantDataset(TRAIN_IMAGES, TRAIN_LABELS, IMG_SIZE, train_transforms)
    val_dataset = ElephantDataset(VAL_IMAGES, VAL_LABELS, IMG_SIZE, val_transforms)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=1, pin_memory=True
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    model = YOLOv13(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Conservative initialization for HybridLoss stability
    def conservative_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0, 0.01)  # Small std
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(conservative_init)
    
    # Use Adam optimizer for better stability
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )
    
    # Adaptive learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
   
    criterion = HybridLoss(
        focal_weight=0.2,      
        ciou_weight=0.1,       
        diou_weight=0.05,      
        tweak_weight=0.02,     
        use_ciou=True,
        use_diou=True,
        focal_gamma=1.0,      
        focal_alpha=0.25,
        img_size=IMG_SIZE
    ).to(DEVICE)
    
    model_info = model.get_model_info()
    print(f"\nModel Configuration:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    loss_info = criterion.get_loss_info()
    print(f"\nULTRA-STABLE Hybrid Loss Configuration:")
    for key, value in loss_info.items():
        print(f"  {key}: {value}")
    
    best_map = 0.0
    training_history = []
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        valid_batches = 0
        start_time = time.time()
        nan_count = 0
        
        for batch_idx, (images, targets, names) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            
            # Skip empty target batches
            has_targets = any(len(t) > 0 for t in targets)
            if not has_targets:
                continue
            
            # Warmup phase with even lower learning rate
            if epoch == 1 and batch_idx < 100:
                warmup_lr = LEARNING_RATE * (batch_idx + 1) / 100
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            optimizer.zero_grad()
            
            try:
                predictions = model(images)
                loss = criterion(predictions, targets)
                
                # Check for NaN with HybridLoss
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                    nan_count += 1
                    if nan_count <= 5:  # Only log first 5
                        print(f"  Invalid loss at batch {batch_idx + 1}: {loss.item()}")
                    continue
                
                # Cap extremely high losses
                if loss.item() > 100:
                    loss = torch.clamp(loss, max=10.0)
                
                loss.backward()
                
                # Check gradients for NaN/Inf
                grad_valid = True
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            grad_valid = False
                            break
                
                if not grad_valid:
                    nan_count += 1
                    if nan_count <= 5:
                        print(f"  Invalid gradients at batch {batch_idx + 1}")
                    optimizer.zero_grad()
                    continue
                
                # Very conservative gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                
            except Exception as e:
                nan_count += 1
                if nan_count <= 3:
                    print(f"  Exception in batch {batch_idx + 1}: {e}")
                continue
            
            num_batches += 1
            
            # Progress reporting
            if (batch_idx + 1) % 200 == 0 and valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                elapsed = time.time() - start_time
                success_rate = valid_batches / num_batches * 100
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f} - Success: {success_rate:.1f}% - Time: {elapsed:.1f}s")
        
        # Check if epoch had any valid batches
        if valid_batches == 0:
            print(f"  Epoch {epoch}: No valid batches! ({nan_count} failed)")
            patience_counter += 1
            if patience_counter >= 5:
                print("  Training failed: Too many consecutive epochs without valid batches")
                return None, 0.0
            continue
        else:
            patience_counter = 0
            
        avg_train_loss = epoch_loss / valid_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_nan_count = 0
        
        with torch.no_grad():
            for images, targets, names in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                
                has_targets = any(len(t) > 0 for t in targets)
                if not has_targets:
                    continue
                    
                try:
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0):
                        val_loss += loss.item()
                        val_batches += 1
                    else:
                        val_nan_count += 1
                except Exception as e:
                    val_nan_count += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
        # Metrics evaluation
        print(f"  Computing metrics...")
        metrics = evaluate_model(model, val_loader)
        
        current_map = metrics['mAP@0.5']
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Valid: {valid_batches}/{num_batches}, NaN: {nan_count})")
        print(f"  Val Loss: {avg_val_loss:.4f} (Valid: {val_batches}, NaN: {val_nan_count})")
        print(f"  mAP@0.5: {current_map:.4f}")
        print(f"  mAP@0.75: {metrics['mAP@0.75']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'mAP_0.5': current_map,
            'mAP_0.75': metrics['mAP@0.75'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'valid_batches': valid_batches,
            'nan_batches': nan_count
        })
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'metrics': metrics,
            'training_history': training_history,
            'dataset_mean': calculated_mean,
            'dataset_std': calculated_std,
            'dataset_config': dataset_config
        }
        
        torch.save(checkpoint, CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth')
        
        if current_map > best_map:
            best_map = current_map
            torch.save(checkpoint, CHECKPOINT_DIR / 'best_model.pth')
            print(f"  NEW BEST MODEL! mAP@0.5: {best_map:.4f}")
        
        with open(CHECKPOINT_DIR / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        torch.cuda.empty_cache()
    
    print(f"\nTraining Complete!")
    print(f"Best mAP@0.5: {best_map:.4f}")
    print(f"Models saved in: {CHECKPOINT_DIR}")
    
    return str(CHECKPOINT_DIR / 'best_model.pth'), best_map

def find_test_images():
    test_images = []
    test_dir = Path(TEST_IMAGES)
    
    if test_dir.exists():
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in image_extensions:
            test_images.extend(glob.glob(str(test_dir / ext)))
            test_images.extend(glob.glob(str(test_dir / ext.upper())))
        
        test_images = [img for img in test_images if validate_image(img)]
    
    if not test_images:
        val_dir = Path(VAL_IMAGES)
        if val_dir.exists():
            val_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                val_images.extend(list(val_dir.glob(ext)))
            valid_val = [str(img) for img in val_images if validate_image(img)]
            test_images = valid_val[:3]
    
    return test_images

def inference(image_path, checkpoint_path, conf_threshold=0.25):
    print(f"\nRunning inference on: {Path(image_path).name}")
    
    if not validate_image(image_path):
        print(f"Invalid/corrupted image: {image_path}")
        return []
    
    model = YOLOv13(num_classes=NUM_CLASSES).to(DEVICE)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"Model mAP@0.5: {checkpoint['metrics'].get('mAP@0.5', 0):.4f}")
    else:
        print("Using random weights (no checkpoint provided)")
    
    model.eval()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return []
    
    original_image = image.copy()
    processed_image, _, _ = letterbox(image, (IMG_SIZE, IMG_SIZE))
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    processed_image = processed_image.astype(np.float32) / 255.0
    processed_image = (processed_image - np.array(calculated_mean)) / np.array(calculated_std)
    
    input_tensor = torch.from_numpy(np.transpose(processed_image, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(input_tensor)
    
    detections = []
    for scale_idx, pred_scale in enumerate(predictions):
        pred = pred_scale[0].detach().cpu().numpy()
        
        if len(pred.shape) == 3:
            C, H, W = pred.shape
            for h in range(H):
                for w in range(W):
                    conf = torch.sigmoid(torch.tensor(pred[0, h, w])).item()
                    if conf > conf_threshold:
                        x = torch.sigmoid(torch.tensor(pred[1, h, w])).item()
                        y = torch.sigmoid(torch.tensor(pred[2, h, w])).item()
                        width = torch.exp(torch.tensor(pred[3, h, w])).item()
                        height = torch.exp(torch.tensor(pred[4, h, w])).item()
                        
                        x = (x + w) / W
                        y = (y + h) / H
                        
                        x1 = (x - width/2) * IMG_SIZE
                        y1 = (y - height/2) * IMG_SIZE
                        x2 = (x + width/2) * IMG_SIZE
                        y2 = (y + height/2) * IMG_SIZE
                        
                        detections.append([x1, y1, x2, y2, conf])
    
    if len(detections) == 0:
        print("No elephants detected")
        return []
    
    keep_indices = non_max_suppression(detections, 0.45)
    final_detections = [detections[i] for i in keep_indices]
    
    print(f"Found {len(final_detections)} elephant(s)")
    
    for i, detection in enumerate(final_detections):
        x1, y1, x2, y2, conf = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(original_image, f'{CLASS_NAMES[0]} {conf:.2f}', (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"  Detection {i+1}: {CLASS_NAMES[0]} - Confidence={conf:.3f}, BBox=({x1},{y1},{x2},{y2})")
    
    output_dir = Path('inference_results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{Path(image_path).stem}_result.jpg"
    cv2.imwrite(str(output_path), original_image)
    print(f"Result saved to: {output_path}")
    
    return final_detections

def run_complete_pipeline():
    print("STARTING ULTRA-STABLE HYBRID LOSS YOLO TRAINING")
    print("=" * 80)
    
    result = train_model()
    if result is None:
        print("Training failed due to numerical instability.")
        return
        
    best_checkpoint, best_map = result
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED - STARTING INFERENCE")
    print("=" * 80)
    
    test_images = find_test_images()
    
    if not test_images:
        print("No valid test images found.")
        return
    
    print(f"Found {len(test_images)} valid test images for inference")
    
    total_detections = 0
    for i, image_path in enumerate(test_images, 1):
        print(f"\n--- Inference {i}/{len(test_images)} ---")
        detections = inference(image_path, best_checkpoint)
        total_detections += len(detections)
    
    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE FINISHED")
    print("=" * 80)
    print(f"Training Results:")
    print(f"  Best mAP@0.5: {best_map:.4f}")
    print(f"  Model saved: {best_checkpoint}")
    print(f"\nInference Results:")
    print(f"  Processed {len(test_images)} images")
    print(f"  Total detections: {total_detections}")
    print(f"  Results saved in: inference_results/")

if __name__ == '__main__':
    run_complete_pipeline()
