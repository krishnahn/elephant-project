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
import time

# Set environment variable for memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from model import YOLOv13
from loss_functions import HybridLoss, FocalLoss

DATASET_ROOT = r"C:\Users\hamsa\OneDrive\Desktop\ML and DL\Genik project\Wild Animal detection using YOLO\elephant-dataset-yolov"
DATASET_YAML = r"C:\Users\hamsa\OneDrive\Desktop\ML and DL\Genik project\Wild Animal detection using YOLO\elephant-dataset-yolov\dataset.yaml"
NUM_CLASSES = 1
IMG_SIZE = 416  # Reduced from 640 for 4GB GPU
BATCH_SIZE = 2  # Reduced from 16 for 4GB GPU
EPOCHS = 100
LEARNING_RATE = 5e-4  # Slightly higher since we're using simpler loss

# GPU Configuration and Diagnostics
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')  # Explicitly use first GPU
    # Set memory allocation strategy for better performance
    torch.cuda.empty_cache()
    # Set memory fraction to avoid OOM
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    # Enable memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    DEVICE = torch.device('cpu')
    print("CUDA not available. Please check:")
    print("1. GPU drivers are installed")
    print("2. CUDA toolkit is installed") 
    print("3. PyTorch was installed with CUDA support")
    print("Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, p=1.0),  # Ensure this always runs
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3), 
    A.RandomScale(scale_limit=0.5, p=0.4),
    A.HueSaturationValue(
        hue_shift_limit=15,
        sat_shift_limit=25,
        val_shift_limit=20,
        p=0.6
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.6
    ),
    A.GaussNoise(p=0.2),  # Using defaults
    A.CoarseDropout(
        num_holes_range=(1, 8),  # Updated parameter names
        hole_height_range=(8, 32), 
        hole_width_range=(8, 32),
        p=0.3
    ),
    A.OneOf([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1.0),
    ], p=0.3),
    A.OneOf([
        A.GridDropout(ratio=0.2, p=1.0),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(10, 20), hole_width_range=(10, 20), p=1.0),
    ], p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.3
))

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, p=1.0),  # Ensure this always runs
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

def calculate_metrics(predictions_list, ground_truths_list, iou_threshold=0.5):
    all_predictions = []
    all_ground_truths = []
    
    for preds, gts in zip(predictions_list, ground_truths_list):
        all_predictions.extend(preds)
        all_ground_truths.extend(gts)
    
    if len(all_predictions) == 0 and len(all_ground_truths) == 0:
        return 1.0, 1.0, 1.0, 1.0
    
    if len(all_predictions) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    if len(all_ground_truths) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    ap = calculate_ap(all_predictions, all_ground_truths, iou_threshold)
    
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
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = 1
        else:
            fp += 1
    
    fn = len(all_ground_truths) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return ap, precision, recall, f1_score

def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return x1, y1, x2, y2

def load_yolo_label(label_path: Path, img_w: int, img_h: int):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            boxes.append([x, y, w, h])
    return boxes

def letterbox(im, new_shape=640, color=(114, 114, 114), auto=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def collate_fn(batch):
    imgs, targets, names = zip(*batch)
    
    # Ensure all images have the same size
    target_size = (3, IMG_SIZE, IMG_SIZE)
    processed_imgs = []
    
    for i, img in enumerate(imgs):
        if img.shape != target_size:
            # Convert tensor to numpy, resize, and convert back
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            
            # Resize using OpenCV
            img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
            
            # Convert back to tensor
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            processed_imgs.append(img_tensor)
        else:
            processed_imgs.append(img)
    
    imgs = torch.stack(processed_imgs)
    return imgs, targets, names

def simple_nms(boxes: np.ndarray, iou_thresh=0.45):
    if boxes.shape[0] == 0:
        return []
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
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

class YoloDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 640, 
                 augment: bool = False, advanced_aug: bool = False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_paths = sorted([p for p in self.images_dir.iterdir() 
                               if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        self.img_size = img_size
        self.augment = augment
        self.advanced_aug = advanced_aug
        
        if augment:
            self.transform = train_transforms
            print("Using training augmentations")
        else:
            self.transform = val_transforms
            print("Using validation transforms")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]
        
        label_path = self.labels_dir / (img_path.stem + '.txt')
        boxes = []
        class_labels = []
        
        if label_path.exists():
            yolo_boxes = load_yolo_label(label_path, w0, h0)
            boxes = yolo_boxes
            class_labels = [0] * len(boxes)
        
        try:
            if self.augment and boxes:
                transformed = self.transform(
                    image=img_rgb,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                
                img_tensor = transformed['image']
                aug_boxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
                
                target = []
                for box, label in zip(aug_boxes, aug_labels):
                    if len(box) == 4:
                        target.append([label] + list(box))
                
                target = np.array(target, dtype=np.float32) if target else np.zeros((0, 5), dtype=np.float32)
                
            else:
                if boxes:
                    transformed = self.transform(
                        image=img_rgb,
                        bboxes=boxes,
                        class_labels=class_labels
                    )
                    img_tensor = transformed['image']
                    target = []
                    for box in transformed['bboxes']:
                        if len(box) == 4:
                            target.append([0] + list(box))
                else:
                    transformed = self.transform(image=img_rgb, bboxes=[], class_labels=[])
                    img_tensor = transformed['image']
                    target = []
                
                target = np.array(target, dtype=np.float32) if target else np.zeros((0, 5), dtype=np.float32)
            
        except Exception as e:
            print(f"Augmentation failed for {img_path}: {e}")
            img, ratio, pad = letterbox(img, self.img_size, auto=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
            
            if label_path.exists():
                basic_boxes = load_yolo_label(label_path, w0, h0)
                target = []
                for box in basic_boxes:
                    if len(box) == 4:
                        target.append([0] + list(box))
                target = np.array(target, dtype=np.float32) if target else np.zeros((0, 5), dtype=np.float32)
            else:
                target = np.zeros((0, 5), dtype=np.float32)
        
        return img_tensor, torch.from_numpy(target), img_path.name

def evaluate_model(model, dataloader, conf_thresh=0.25, iou_thresh=0.5):
    model.eval()
    predictions_list = []
    ground_truths_list = []
    
    with torch.no_grad():
        for imgs, targets, names in dataloader:
            imgs = imgs.to(DEVICE)
            # Move targets to device - targets is a list of tensors, so move each one
            targets = [target.to(DEVICE, non_blocking=True) if isinstance(target, torch.Tensor) else target for target in targets]
            preds = model(imgs)
            
            batch_predictions = []
            batch_ground_truths = []
            
            for i, (target,) in enumerate(zip(targets)):
                img_predictions = []
                img_ground_truths = []
                
                for pred_scale in preds:
                    pred = pred_scale[i:i+1].detach().cpu().numpy()
                    B, C, H, W = pred.shape
                    
                    pred = pred.reshape(1, H*W, C)
                    
                    for anchor_idx in range(H*W):
                        conf = torch.sigmoid(torch.tensor(pred[0, anchor_idx, 0])).item()
                        if conf > conf_thresh:
                            x = torch.sigmoid(torch.tensor(pred[0, anchor_idx, 1])).item()
                            y = torch.sigmoid(torch.tensor(pred[0, anchor_idx, 2])).item()
                            w = torch.exp(torch.tensor(pred[0, anchor_idx, 3])).item()
                            h = torch.exp(torch.tensor(pred[0, anchor_idx, 4])).item()
                            
                            grid_x = anchor_idx % W
                            grid_y = anchor_idx // W
                            
                            x = (x + grid_x) / W
                            y = (y + grid_y) / H
                            w = w / W
                            h = h / H
                            
                            x1 = (x - w/2) * IMG_SIZE
                            y1 = (y - h/2) * IMG_SIZE
                            x2 = (x + w/2) * IMG_SIZE
                            y2 = (y + h/2) * IMG_SIZE
                            
                            img_predictions.append([x1, y1, x2, y2, conf])
                
                target = target.numpy()
                for gt in target:
                    if len(gt) >= 5 and gt[0] >= 0:
                        _, x_center, y_center, width, height = gt[:5]
                        x1 = (x_center - width/2) * IMG_SIZE
                        y1 = (y_center - height/2) * IMG_SIZE
                        x2 = (x_center + width/2) * IMG_SIZE
                        y2 = (y_center + height/2) * IMG_SIZE
                        img_ground_truths.append([x1, y1, x2, y2])
                
                if img_predictions:
                    keep_indices = simple_nms(np.array(img_predictions), iou_thresh)
                    img_predictions = [img_predictions[i] for i in keep_indices]
                
                batch_predictions.append(img_predictions)
                batch_ground_truths.append(img_ground_truths)
            
            predictions_list.extend(batch_predictions)
            ground_truths_list.extend(batch_ground_truths)
    
    map_score, precision, recall, f1_score = calculate_metrics(
        predictions_list, ground_truths_list, iou_thresh
    )
    
    return map_score, precision, recall, f1_score

def train(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, 
          use_hybrid_loss=True, advanced_augmentation=False):
    print(f"Starting YOLOv13 Elephant Detection Training")
    print("=" * 60)
    print(f"Architecture: CSPDarkNet53 + BiFPN + Multi-scale Head")
    print(f"Loss: {'Focal + CIoU/DIoU + Tweak' if use_hybrid_loss else 'Focal only'}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Optimizer: SGD with momentum")
    
    train_images = Path(DATASET_ROOT) / 'train' / 'images'
    train_labels = Path(DATASET_ROOT) / 'train' / 'labels'
    val_images = Path(DATASET_ROOT) / 'valid' / 'images'
    val_labels = Path(DATASET_ROOT) / 'valid' / 'labels'
    
    train_ds = YoloDataset(
        str(train_images), str(train_labels), 
        img_size=IMG_SIZE, augment=True, advanced_aug=advanced_augmentation
    )
    val_ds = YoloDataset(
        str(val_images), str(val_labels), 
        img_size=IMG_SIZE, augment=False
    )
    
    # Configure DataLoaders with GPU optimizations
    num_workers = 0  # Use 0 to avoid multiprocessing issues on Windows
    pin_memory = DEVICE.type == 'cuda'
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            collate_fn=collate_fn, num_workers=num_workers, 
                            pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                          collate_fn=collate_fn, num_workers=0, 
                          pin_memory=pin_memory)
    
    print(f"\nDataset Statistics:")
    print(f"  Training images: {len(train_ds)}")
    print(f"  Validation images: {len(val_ds)}")
    
    model = YOLOv13(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Initialize model weights properly to prevent NaN
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Additional initialization for YOLO head to prevent large initial predictions
    for module in model.modules():
        if hasattr(module, 'conv') and hasattr(module.conv, 'weight'):
            # Initialize final detection layers with smaller weights
            if module.conv.out_channels == NUM_CLASSES + 5:  # Detection head
                torch.nn.init.normal_(module.conv.weight, 0, 0.01)
                if module.conv.bias is not None:
                    torch.nn.init.constant_(module.conv.bias, 0)
    
    print("✓ Model weights initialized with YOLO-specific initialization")
    
    # GPU Optimizations
    if DEVICE.type == 'cuda':
        print(f"\n=== GPU Optimizations ===")
        # Enable mixed precision training for faster performance
        scaler = torch.amp.GradScaler('cuda')
        print("✓ Mixed precision training enabled")
        
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmark enabled")
        
        # Memory optimization
        torch.cuda.empty_cache()
        print("✓ GPU memory cache cleared")
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        
        # Adjust batch size based on GPU memory if needed
        if gpu_memory < 8 and batch_size > 8:
            print(f"Large batch size ({batch_size}) for GPU memory ({gpu_memory:.1f} GB)")
            print("   Consider reducing batch size if you encounter OOM errors")
    else:
        print("\n Running on CPU - training will be slower")
        scaler = None
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Start with simple FocalLoss for stability, then can upgrade to HybridLoss once working
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
    print("Using FocalLoss for training stability")

    
    model_info = model.get_model_info()
    print(f"\nModel Configuration:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    best_map = 0.0
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        start_time = time.time()
        
        for batch_idx, (imgs, targets, _) in enumerate(train_loader):
            try:
                imgs = imgs.to(DEVICE, non_blocking=True)
                # Move targets to device - targets is a list of tensors, so move each one
                targets = [target.to(DEVICE, non_blocking=True) if isinstance(target, torch.Tensor) else target for target in targets]
                
                optimizer.zero_grad()
                
                # Clear GPU cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
                # Use mixed precision if GPU is available
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        preds = model(imgs)
                        loss = criterion(preds, targets)
                    
                    # Check for NaN/Inf values and scale loss
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                        print(f"Warning: Invalid loss detected: {loss.item()}")
                        print(f"Predictions range: [{preds[0].min().item():.6f}, {preds[0].max().item():.6f}]")
                        torch.cuda.empty_cache()  # Clear memory on error
                        continue  # Skip this batch
                    
                    # Scale down large losses
                    if loss.item() > 100:
                        loss = loss / 10.0
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Very conservative clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = model(imgs)
                    loss = criterion(preds, targets)
                    
                    # Check for NaN/Inf values
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                        print(f"Warning: Invalid loss detected: {loss.item()}")
                        continue  # Skip this batch
                    
                    # Scale down large losses
                    if loss.item() > 100:
                        loss = loss / 10.0
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM at batch {batch_idx}. Clearing cache and skipping batch...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                # Show GPU memory usage if available
                if DEVICE.type == 'cuda':
                    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s, GPU Mem: {gpu_mem:.1f}GB")
                else:
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")
        
        avg_train_loss = running_loss / max(1, len(train_loader))
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                # Move targets to device - targets is a list of tensors, so move each one
                targets = [target.to(DEVICE, non_blocking=True) if isinstance(target, torch.Tensor) else target for target in targets]
                
                # Use mixed precision for validation too
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        preds = model(imgs)
                        loss = criterion(preds, targets)
                else:
                    preds = model(imgs)
                    loss = criterion(preds, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        
        print(f"Computing evaluation metrics...")
        map_score, precision, recall, f1_score = evaluate_model(model, val_loader)
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  mAP@0.5: {map_score:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        
        scheduler.step()
        
        ckpt_path = CHECKPOINT_DIR / f'yolov13_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'map_score': map_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }, ckpt_path)
        
        if map_score > best_map:
            best_map = map_score
            best_ckpt_path = CHECKPOINT_DIR / 'yolov13_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'map_score': map_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }, best_ckpt_path)
            print(f"  Saved best model! mAP: {map_score:.4f}")
    
    print(f"\nTraining Complete!")
    print(f"Best mAP@0.5: {best_map:.4f}")
    print(f"Models saved in: {CHECKPOINT_DIR}")

def infer(image_path: str, checkpoint: str = None, conf_thresh: float = 0.25, save: bool = True):
    print(f"Running inference on: {image_path}")
    print(f"Confidence threshold: {conf_thresh}")
    
    model = YOLOv13(num_classes=NUM_CLASSES).to(DEVICE)
    
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "unknown")}')
        if 'map_score' in ckpt:
            print(f"Model mAP@0.5: {ckpt['map_score']:.4f}")
    else:
        print('Using randomly initialized model')
    
    model.eval()
    img = cv2.imread(image_path)
    assert img is not None, f'Image not found: {image_path}'
    orig = img.copy()
    img, ratio, pad = letterbox(img, IMG_SIZE, auto=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    img_t = torch.from_numpy(img).to(DEVICE)
    
    with torch.no_grad():
        preds = model(img_t)
    
    boxes = []
    for pred_scale in preds:
        pred = pred_scale[0].detach().cpu().numpy()
        C, H, W = pred.shape
        
        for h in range(H):
            for w in range(W):
                conf = torch.sigmoid(torch.tensor(pred[0, h, w])).item()
                if conf > conf_thresh:
                    x = torch.sigmoid(torch.tensor(pred[1, h, w])).item()
                    y = torch.sigmoid(torch.tensor(pred[2, h, w])).item()
                    width = torch.exp(torch.tensor(pred[3, h, w])).item()
                    height = torch.exp(torch.tensor(pred[4, h, w])).item()
                    
                    x = (x + w) / W
                    y = (y + h) / H
                    width = width / W
                    height = height / H
                    
                    x1 = (x - width/2) * IMG_SIZE
                    y1 = (y - height/2) * IMG_SIZE
                    x2 = (x + width/2) * IMG_SIZE
                    y2 = (y + height/2) * IMG_SIZE
                    
                    boxes.append([x1, y1, x2, y2, conf, 0])
    
    boxes = np.array(boxes)
    if boxes.size == 0:
        print('No detections found')
        return []
    
    keep = simple_nms(boxes, iou_thresh=0.45)
    boxes = boxes[keep]
    print(f'Found {len(boxes)} detections after NMS')
    
    for b in boxes:
        x1, y1, x2, y2, conf, cls_id = map(float, b)
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(IMG_SIZE - 1, x2))
        y2 = int(min(IMG_SIZE - 1, y2))
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig, f'elephant {conf:.2f}', (x1, max(0, y1 - 6)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f'  Detection: confidence={conf:.3f}, bbox=({x1},{y1},{x2},{y2})')
    
    if save:
        outp = Path('runs')
        outp.mkdir(exist_ok=True)
        out_file = outp / (Path(image_path).stem + '_pred.jpg')
        cv2.imwrite(str(out_file), orig)
        print(f'Saved result: {out_file}')
    
    return boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv13 Elephant Detection with Hybrid Loss')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    parser.add_argument('--simple_loss', action='store_true')
    parser.add_argument('--advanced_aug', action='store_true')
    
    args = parser.parse_args()
    
    # Quick GPU check and recommendations
    if not torch.cuda.is_available() and (args.train or args.infer):
        print("\n" + "="*60)
        print("⚠️  CUDA/GPU NOT AVAILABLE")
        print("="*60)
        print("For faster training, install PyTorch with CUDA support:")
        print("1. Check your NVIDIA GPU: nvidia-smi")
        print("2. Install CUDA toolkit from NVIDIA")
        print("3. Install PyTorch with CUDA:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("4. Verify installation: python -c \"import torch; print(torch.cuda.is_available())\"")
        print("="*60)
        
        if args.train:
            response = input("Continue training on CPU? (y/N): ")
            if response.lower() != 'y':
                print("Training cancelled. Please set up GPU support for faster training.")
                exit(0)
    
    if args.train:
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_hybrid_loss=not args.simple_loss,
            advanced_augmentation=args.advanced_aug
        )
    elif args.infer:
        if args.image is None:
            print('Please provide --image path for inference')
        else:
            infer(args.image, checkpoint=args.checkpoint, conf_thresh=args.conf_thresh)
    else:
        # No arguments provided - start training with default settings
        print('YOLOv13 Elephant Detection System')
        print('\nArchitecture:')
        print('  Backbone: CSPDarkNet53')
        print('  Neck: BiFPN (Bidirectional Feature Pyramid Network)')
        print('  Head: Multi-scale prediction heads')
        print('  Loss: Focal + CIoU/DIoU + Tweak Loss')
        print('\nNo arguments provided - starting training with default settings...')
        print('='*60)
        
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_hybrid_loss=not args.simple_loss,
            advanced_augmentation=args.advanced_aug
        )
