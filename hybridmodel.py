import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from pathlib import Path
import shutil
from datetime import datetime
import os
from google.colab import files

IMG_SIZE = 640
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_YAML = "/content/dataset/elephant-dataset-yolov/dataset.yaml"
CHECKPOINT_DIR = Path('/content/elephant_yolo11_focal_ciou')
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / 'yolo11_elephant_best.pt'
LAST_MODEL_PATH = CHECKPOINT_DIR / 'yolo11_elephant_last.pt'

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * ((1 - pt) ** self.gamma)
        loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        return loss.mean()

class CIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = b1_area + b2_area - inter_area + self.eps
        iou = inter_area / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + self.eps

        b1_cx = (b1_x1 + b1_x2) / 2
        b1_cy = (b1_y1 + b1_y2) / 2
        b2_cx = (b2_x1 + b2_x2) / 2
        b2_cy = (b2_y1 + b2_y2) / 2
        rho2 = ((b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + self.eps)) - torch.atan(w1 / (h1 + self.eps)), 2)

        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        ciou = iou - (rho2 / c2 + alpha * v)
        return 1 - ciou

class HybridLoss(v8DetectionLoss):
    def __init__(self, model, use_ciou=True):
        super().__init__(model)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.bbox_loss = CIoULoss()
        print(f"HybridLoss initialized: Focal + CIoU")

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        loss_cls = torch.zeros(1, device=self.device)
        loss_box = torch.zeros(1, device=self.device)

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            target_scores_sum = max(target_scores.sum(), 1)

            loss_cls = self.focal_loss(pred_scores, target_scores.to(dtype))
            loss_box = self.bbox_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask]).sum() / target_scores_sum

        loss_cls *= self.hyp.cls
        loss_box *= self.hyp.box

        total_loss = loss_cls + loss_box
        loss_items = torch.cat((loss_box.unsqueeze(0), loss_cls.unsqueeze(0))).detach()

        return total_loss, loss_items

class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return HybridLoss(self, use_ciou=True)

class HybridTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        rank = getattr(self, 'rank', -1)
        model = CustomDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and rank == -1)
        if weights:
            model.load(weights)
        return model

print("="*80)
print("YOLO11 ELEPHANT DETECTION - FOCAL + CIoU LOSS")
print("="*80)
print(f"\nConfiguration:")
print(f"  Model: YOLO11n")
print(f"  Dataset: {DATASET_YAML}")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Device: {DEVICE}")
print(f"\nLoss Function (as per PDF):")
print(f"  Focal Loss: α=0.25, γ=2.0")
print(f"  CIoU Loss: Complete IoU")
print(f"  Formula: L_total = λ_cls*L_focal + λ_box*L_CIoU")

model = YOLO('yolo11n.pt')

print(f"\nStarting training...")
print("="*80)

results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    lr0=LEARNING_RATE,
    device=DEVICE,
    project=str(CHECKPOINT_DIR),
    name='training',
    exist_ok=True,
    verbose=True,
    save=True,
    val=True,
    plots=True,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    box=5.0,
    cls=1.0,
    patience=50,
    save_period=10,
    trainer=HybridTrainer,
)

training_weights = CHECKPOINT_DIR / 'training' / 'weights'
if (training_weights / 'best.pt').exists():
    shutil.copy(training_weights / 'best.pt', BEST_MODEL_PATH)
    print(f"\n✓ Best model: {BEST_MODEL_PATH}")
if (training_weights / 'last.pt').exists():
    shutil.copy(training_weights / 'last.pt', LAST_MODEL_PATH)
    print(f"✓ Last model: {LAST_MODEL_PATH}")

print("\n" + "="*80)
print("TRAINING COMPLETED")
print("="*80)

if BEST_MODEL_PATH.exists():
    best_model = YOLO(str(BEST_MODEL_PATH))
    metrics = best_model.val(data=DATASET_YAML)
    print(f"\nValidation Metrics:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = CHECKPOINT_DIR / f'yolo11_elephant_{timestamp}.pt'
if BEST_MODEL_PATH.exists():
    shutil.copy(BEST_MODEL_PATH, backup)
    print(f"\n✓ Backup: {backup.name}")

print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}")

os.system(f"zip -q -r elephant_yolo11_focal_ciou.zip {CHECKPOINT_DIR}")
files.download('elephant_yolo11_focal_ciou.zip')
print("\n✓ Download complete!")
