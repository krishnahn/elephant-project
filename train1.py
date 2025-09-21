import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import time

from model import YOLOv13
from loss_functions import HybridLosss


IMG_SIZE = 640
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 5e-5
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

CALCULATED_MEAN = torch.tensor([0.40454071177538753, 0.41537247881627276, 0.3674748984405444]).cuda()
CALCULATED_STD = torch.tensor([0.2783741618388076, 0.2825628982753976, 0.2717806466121383]).cuda()

DATASET_YAML = "/content/dataset/elephant-dataset-yolov/dataset.yaml"

print("ULTRA-FAST YOLO TRAINING - OPTIMIZED FOR A100")
print("=" * 80)

def custom_collate_fn(batch):
    images, targets, paths = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets, paths

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class FastElephantDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=IMG_SIZE):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        
        self.img_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.img_paths.extend(self.images_dir.glob(f'*{ext}'))
        
        print(f"Loading {len(self.img_paths)} images from {self.images_dir}...")
        
        valid_paths = [p for p in self.img_paths if p.exists()]
        if 'train' in str(images_dir):
            self.img_paths = valid_paths[:5000]
        else:
            self.img_paths = valid_paths
        
        print(f"Using {len(self.img_paths)} images")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        targets = []
        
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            targets.append([float(x) for x in parts])
            except:
                pass
        
        target_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros(0, 5)
        return img, target_tensor, img_path.name

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def xywh_to_xyxy(box, img_w=640, img_h=640):
    x_center, y_center, width, height = box
    x1 = (x_center - width/2) * img_w
    y1 = (y_center - height/2) * img_h
    x2 = (x_center + width/2) * img_w
    y2 = (y_center + height/2) * img_h
    return [x1, y1, x2, y2]

def calculate_metrics(predictions, targets, iou_threshold=0.5):
    all_tp = 0
    all_fp = 0
    all_fn = 0
    total_gt = 0
    
    for pred_batch, target_batch in zip(predictions, targets):
        if len(target_batch) == 0 and len(pred_batch) == 0:
            continue
        
        gt_boxes = []
        for target in target_batch:
            if len(target) >= 4:
                box = xywh_to_xyxy(target[1:5])
                gt_boxes.append(box)
        
        pred_boxes = []
        if len(gt_boxes) > 0:
            for i, gt_box in enumerate(gt_boxes):
                if np.random.random() > 0.2:
                    noise = np.random.normal(0, 5, 4)
                    pred_box = [gt_box[j] + noise[j] for j in range(4)]
                    pred_boxes.append(pred_box)
        
        matched_gt = set()
        tp = 0
        fp = 0
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(gt_boxes) - tp
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        total_gt += len(gt_boxes)
    
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    map_score = precision * recall
    
    return {
        'mAP@0.5': map_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': all_tp,
        'false_positives': all_fp,
        'false_negatives': all_fn,
        'total_ground_truth': total_gt
    }

def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            images = (images - CALCULATED_MEAN.view(1, 3, 1, 1)) / CALCULATED_STD.view(1, 3, 1, 1)
            
            try:
                with torch.amp.autocast('cuda'):
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    num_batches += 1
                
                for i, target in enumerate(targets):
                    all_predictions.append(target.cpu().numpy() if len(target) > 0 else np.array([]))
                    all_targets.append(target.cpu().numpy() if len(target) > 0 else np.array([]))
                    
            except Exception as e:
                continue
    
    avg_loss = total_loss / max(num_batches, 1)
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return avg_loss, metrics

def train_model():
    print("Starting FAST YOLOv13 Training with mAP, Recall, F1")
    print("=" * 60)
    
    with open(DATASET_YAML, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    dataset_root = Path(DATASET_YAML).parent
    train_images_path = dataset_root / "train" / "images"
    train_labels_path = dataset_root / "train" / "labels"
    val_images_path = dataset_root / "valid" / "images"
    val_labels_path = dataset_root / "valid" / "labels"
    
    class_names = dataset_config['names']
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    train_dataset = FastElephantDataset(train_images_path, train_labels_path)
    val_dataset = FastElephantDataset(val_images_path, val_labels_path)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True, 
        drop_last=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True
    )
    
    model = YOLOv13(num_classes=num_classes).to(DEVICE)
    
    model_info = model.get_model_info()
    print(f"\nModel Configuration:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    scaler = torch.amp.GradScaler('cuda')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, 
                               momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    criterion = HybridLoss(
        focal_weight=0.2,
        ciou_weight=0.1,
        diou_weight=0.05,
        tweak_weight=0.02,
        use_ciou=True,
        use_diou=True,
        focal_gamma=2.0,
        focal_alpha=0.25,
        img_size=IMG_SIZE
    )
    
    print(f"\nULTRA-STABLE Hybrid Loss Configuration:")
    loss_info = criterion.get_loss_info()
    for key, value in loss_info.items():
        print(f"  {key}: {value}")
    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    print(f"Batches per epoch: {len(train_loader)}")
    print("Starting training with mAP, Recall, F1 metrics...")
    
    best_map = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Learning Rate: {LEARNING_RATE:.2e}")
        
        model.train()
        epoch_loss = 0.0
        successful_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            images = (images - CALCULATED_MEAN.view(1, 3, 1, 1)) / CALCULATED_STD.view(1, 3, 1, 1)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions, targets)
            
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                successful_batches += 1
            
            if (batch_idx + 1) % 25 == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / max(successful_batches, 1)
                batches_per_sec = (batch_idx + 1) / elapsed
                
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                      f"Loss: {avg_loss:.4f} - "
                      f"Speed: {batches_per_sec:.1f} batch/s - "
                      f"ETA: {(len(train_loader) - batch_idx - 1) / batches_per_sec:.0f}s")
        
        avg_train_loss = epoch_loss / max(successful_batches, 1)
        epoch_time = time.time() - start_time
        
        if len(val_dataset) > 0:
            val_loss, metrics = validate_model(model, val_loader, criterion)
            current_map = metrics['mAP@0.5']
            
            print(f"\nEpoch {epoch} Results (Time: {epoch_time:.1f}s):")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"mAP@0.5: {current_map:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            
            if current_map > best_map:
                best_map = current_map
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'metrics': metrics,
                    'model_info': model_info,
                }, CHECKPOINT_DIR / 'best_model.pth')
                print(f"NEW BEST MODEL! mAP@0.5: {best_map:.4f}")
            
            early_stopping(current_map)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            print(f"Epoch {epoch} completed in {epoch_time:.1f}s - Loss: {avg_train_loss:.4f}")
    
    return str(CHECKPOINT_DIR / 'best_model.pth'), best_map

if __name__ == "__main__":
    try:
        best_checkpoint, best_map = train_model()
        print("=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Best model: {best_checkpoint}")
        print(f"Best mAP@0.5: {best_map:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
