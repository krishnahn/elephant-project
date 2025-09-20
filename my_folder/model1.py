import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou

DATASET_ROOT = r"C:\Users\hamsa\OneDrive\Desktop\ML and DL\Genik project\Wild Animal detection using YOLO\elephant-dataset-yolov"
DATASET_YAML = r"C:\Users\hamsa\OneDrive\Desktop\ML and DL\Genik project\Wild Animal detection using YOLO\elephant-dataset-yolov\dataset.yaml"
NUM_CLASSES = 1
IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ----------------------- UTILITIES -----------------------

def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    # normalized center x,y,w,h -> absolute x1,y1,x2,y2
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return x1, y1, x2, y2


def load_yolo_label(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = xywhn_to_xyxy(x, y, w, h, img_w, img_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


# ----------------------- DATASET -----------------------
class YoloDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 640, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h0, w0 = img.shape[:2]

        # read labels
        label_path = self.labels_dir / (img_path.stem + '.txt')
        boxes = []
        if label_path.exists():
            boxes = load_yolo_label(label_path, w0, h0)

        # resize with letterbox (preserve aspect ratio)
        img, ratio, pad = letterbox(img, self.img_size, auto=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW

        # transform boxes to resized coordinates
        transformed_boxes = []
        for cls, x1, y1, x2, y2 in boxes:
            x1n = x1 * ratio + pad[0]
            y1n = y1 * ratio + pad[1]
            x2n = x2 * ratio + pad[0]
            y2n = y2 * ratio + pad[1]
            transformed_boxes.append([cls, x1n, y1n, x2n, y2n])

        target = np.array(transformed_boxes, dtype=np.float32) if transformed_boxes else np.zeros((0,5), dtype=np.float32)

        return torch.from_numpy(img), target, img_path.name


def letterbox(im, new_shape=640, color=(114,114,114), auto=True):
    # Resize and pad image to new_shape square
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # compute padding
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
    imgs = torch.stack(imgs)
    # targets: list of (num_boxes,5) arrays
    return imgs, targets, names

# ----------------------- MODEL BUILDING BLOCKS -----------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=2):
        super().__init__()
        hidden = out_ch // 2
        self.conv1 = ConvBNAct(in_ch, hidden, k=1, s=1, p=0)
        self.conv2 = ConvBNAct(in_ch, hidden, k=1, s=1, p=0)
        self.blocks = nn.Sequential(*[ConvBNAct(hidden, hidden) for _ in range(num_blocks)])
        self.conv3 = ConvBNAct(2*hidden, out_ch, k=1, s=1, p=0)
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.blocks(y2)
        out = torch.cat([y1, y2], dim=1)
        out = self.conv3(out)
        return out

class CSPDarknet53Lite(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.stem = ConvBNAct(in_ch, 32, k=3, s=1, p=1)
        self.layer1 = nn.Sequential(ConvBNAct(32,64,k=3,s=2,p=1), CSPBlock(64,64, num_blocks=1))
        self.layer2 = nn.Sequential(ConvBNAct(64,128,k=3,s=2,p=1), CSPBlock(128,128, num_blocks=2))
        self.layer3 = nn.Sequential(ConvBNAct(128,256,k=3,s=2,p=1), CSPBlock(256,256, num_blocks=3))
        self.layer4 = nn.Sequential(ConvBNAct(256,512,k=3,s=2,p=1), CSPBlock(512,512, num_blocks=3))
        self.layer5 = nn.Sequential(ConvBNAct(512,1024,k=3,s=2,p=1), CSPBlock(1024,1024, num_blocks=1))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c2 = self.layer2(x)  
        c3 = self.layer3(c2) 
        c4 = self.layer4(c3)
        c5 = self.layer5(c4) 
        return c3, c4, c5

class BiFPN(nn.Module):
    def __init__(self, channels=256, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        # proj to same channels
        self.p3 = ConvBNAct(256, channels, k=1, s=1, p=0)
        self.p4 = ConvBNAct(512, channels, k=1, s=1, p=0)
        self.p5 = ConvBNAct(1024, channels, k=1, s=1, p=0)
        # weighted fusion parameters
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))
        self.w3 = nn.Parameter(torch.ones(2))
        # smoothing convs
        self.conv_td3 = ConvBNAct(channels, channels)
        self.conv_td4 = ConvBNAct(channels, channels)
        self.conv_bu4 = ConvBNAct(channels, channels)
        self.conv_bu5 = ConvBNAct(channels, channels)

    def fusion(self, inputs: List[torch.Tensor], weights_param: nn.Parameter):
        weights = F.relu(weights_param)
        weights = weights / (torch.sum(weights) + self.epsilon)
        out = weights[0] * inputs[0] + weights[1] * inputs[1]
        return out

    def forward(self, c3, c4, c5):
        p3 = self.p3(c3)
        p4 = self.p4(c4)
        p5 = self.p5(c5)

        # top-down
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        td4 = self.fusion([p4, p5_up], self.w1)
        td4 = self.conv_td4(td4)

        td4_up = F.interpolate(td4, size=p3.shape[-2:], mode='nearest')
        td3 = self.fusion([p3, td4_up], self.w2)
        td3 = self.conv_td3(td3)

        # bottom-up
        td3_down = F.interpolate(td3, size=td4.shape[-2:], mode='nearest')
        bu4 = self.fusion([td4, td3_down], self.w3)
        bu4 = self.conv_bu4(bu4)

        # final upsample for p5 (simple residual combine)
        p5_out = self.conv_bu5(p5)

        # outputs: three fused feature maps
        return td3, bu4, p5_out

class YOLOHead(nn.Module):

    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.pred3 = nn.Conv2d(in_channels, (5 + num_classes), kernel_size=1)
        self.pred4 = nn.Conv2d(in_channels, (5 + num_classes), kernel_size=1)
        self.pred5 = nn.Conv2d(in_channels, (5 + num_classes), kernel_size=1)

    def forward(self, f3, f4, f5):
        o3 = self.pred3(f3)
        o4 = self.pred4(f4)
        o5 = self.pred5(f5)
        return [o3, o4, o5]

class YOLOv13(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = CSPDarknet53Lite()
        self.neck = BiFPN(channels=256)
        self.head = YOLOHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        f3, f4, f5 = self.neck(c3, c4, c5)
        preds = self.head(f3, f4, f5)
        return preds

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        prob = torch.sigmoid(pred)
        pt = prob * target + (1 - prob) * (1 - target)
        w = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -w * (1 - pt)**self.gamma * torch.log(pt + 1e-8)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DIoULoss(nn.Module):
    """Distance IoU Loss for better bounding box regression"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes: (N, 4) [x1, y1, x2, y2]
        target_boxes: (N, 4) [x1, y1, x2, y2]
        """
        # Calculate IoU
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + 1e-8
        iou = inter_area / union_area
        
        # Calculate center distance
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        center_dist = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2
        
        # Calculate enclosing box diagonal
        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        c2 = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-8
        
        # DIoU
        diou = iou - center_dist / c2
        diou_loss = 1 - diou
        
        if self.reduction == 'mean':
            return diou_loss.mean()
        elif self.reduction == 'sum':
            return diou_loss.sum()
        else:
            return diou_loss


class CIoULoss(nn.Module):
    """Complete IoU Loss for improved bounding box regression"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes: (N, 4) [x1, y1, x2, y2]
        target_boxes: (N, 4) [x1, y1, x2, y2]
        """
        # Calculate IoU
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + 1e-8
        iou = inter_area / union_area
        
        # Calculate center distance
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        center_dist = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2
        
        # Calculate enclosing box diagonal
        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        c2 = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-8
        
        # Aspect ratio consistency (v)
        w_pred = pred_boxes[:, 2] - pred_boxes[:, 0]
        h_pred = pred_boxes[:, 3] - pred_boxes[:, 1]
        w_target = target_boxes[:, 2] - target_boxes[:, 0]
        h_target = target_boxes[:, 3] - target_boxes[:, 1]
        
        v = (4 / (3.14159265**2)) * (torch.atan(w_target / (h_target + 1e-8)) - 
                                     torch.atan(w_pred / (h_pred + 1e-8)))**2
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-8)
        
        # CIoU
        ciou = iou - (center_dist / c2 + alpha * v)
        ciou_loss = 1 - ciou
        
        if self.reduction == 'mean':
            return ciou_loss.mean()
        elif self.reduction == 'sum':
            return ciou_loss.sum()
        else:
            return ciou_loss


class TweakLoss(nn.Module):
    """Custom tweak loss for fine-tuning specific aspects of detection"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_conf, target_conf, pred_boxes, target_boxes):
        """
        Tweak loss combining confidence penalty and size consistency
        pred_conf: (N,) confidence scores
        target_conf: (N,) target confidence (0 or 1)
        pred_boxes: (N, 4) predicted boxes
        target_boxes: (N, 4) target boxes
        """
        # Confidence smoothing penalty
        conf_diff = torch.abs(pred_conf - target_conf)
        conf_penalty = conf_diff * torch.exp(conf_diff)
        
        # Size consistency penalty (penalize extreme size differences)
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        w_ratio = torch.max(pred_w / (target_w + 1e-8), target_w / (pred_w + 1e-8))
        h_ratio = torch.max(pred_h / (target_h + 1e-8), target_h / (pred_h + 1e-8))
        size_penalty = torch.log(w_ratio + 1e-8) + torch.log(h_ratio + 1e-8)
        
        # Combine penalties
        tweak_loss = conf_penalty + 0.5 * size_penalty
        
        if self.reduction == 'mean':
            return tweak_loss.mean()
        elif self.reduction == 'sum':
            return tweak_loss.sum()
        else:
            return tweak_loss


class HybridLoss(nn.Module):
    """
    Hybrid Loss combining Focal Loss + CIoU/DIoU + Tweak Loss
    for improved object detection performance
    """
    def __init__(self, 
                 focal_weight=1.0, 
                 ciou_weight=1.0, 
                 diou_weight=0.5, 
                 tweak_weight=0.3,
                 use_ciou=True,
                 use_diou=True,
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 img_size=640):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.ciou_weight = ciou_weight
        self.diou_weight = diou_weight
        self.tweak_weight = tweak_weight
        self.use_ciou = use_ciou
        self.use_diou = use_diou
        self.img_size = img_size
        
        # Initialize loss components
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction='none')
        if use_ciou:
            self.ciou_loss = CIoULoss(reduction='none')
        if use_diou:
            self.diou_loss = DIoULoss(reduction='none')
        self.tweak_loss = TweakLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        predictions: list of tensors from YOLO head [pred1, pred2, pred3]
        targets: list of target arrays per image
        """
        total_loss = 0.0
        focal_loss_sum = 0.0
        ciou_loss_sum = 0.0
        diou_loss_sum = 0.0
        tweak_loss_sum = 0.0
        num_matches = 0
        
        batch_size = len(targets)
        
        for pred_level in predictions:
            # pred_level shape: (B, 5+num_classes, H, W)
            B, C, H, W = pred_level.shape
            
            # Reshape predictions
            pred_level = pred_level.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            pred_level = pred_level.view(B, H * W, C)  # (B, HW, C)
            
            # Extract components
            pred_conf = torch.sigmoid(pred_level[..., 0])  # (B, HW)
            pred_boxes = pred_level[..., 1:5]  # (B, HW, 4) - raw box predictions
            pred_cls = pred_level[..., 5:] if C > 5 else None  # (B, HW, num_classes)
            
            # Create targets for this level
            for b in range(batch_size):
                if len(targets[b]) == 0:
                    # No targets for this image, apply background loss
                    target_conf = torch.zeros_like(pred_conf[b])
                    focal_loss_batch = self.focal_loss(pred_conf[b], target_conf)
                    focal_loss_sum += focal_loss_batch.mean()
                    continue
                
                target_boxes = targets[b][:, 1:5]  # Extract boxes (skip class)
                target_classes = targets[b][:, 0].long()  # Extract classes
                
                # Convert target boxes to grid coordinates
                target_boxes_scaled = target_boxes.clone()
                target_boxes_scaled[:, [0, 2]] *= W  # x coordinates
                target_boxes_scaled[:, [1, 3]] *= H  # y coordinates
                
                # Simple assignment: find closest grid cell for each target
                target_cx = (target_boxes_scaled[:, 0] + target_boxes_scaled[:, 2]) / 2
                target_cy = (target_boxes_scaled[:, 1] + target_boxes_scaled[:, 3]) / 2
                
                grid_x = torch.clamp(target_cx.long(), 0, W - 1)
                grid_y = torch.clamp(target_cy.long(), 0, H - 1)
                grid_idx = grid_y * W + grid_x
                
                # Create target tensors
                target_conf = torch.zeros_like(pred_conf[b])
                target_conf[grid_idx] = 1.0
                
                # Extract matched predictions
                matched_pred_conf = pred_conf[b][grid_idx]
                matched_pred_boxes = pred_boxes[b][grid_idx]
                
                # Convert predicted boxes to absolute coordinates for loss calculation
                matched_pred_boxes_abs = matched_pred_boxes.clone()
                # Assuming the raw predictions are offsets that need to be converted
                # This is a simplified conversion - you may need to adjust based on your exact format
                matched_pred_boxes_abs[:, 0] = torch.sigmoid(matched_pred_boxes_abs[:, 0]) * W
                matched_pred_boxes_abs[:, 1] = torch.sigmoid(matched_pred_boxes_abs[:, 1]) * H
                matched_pred_boxes_abs[:, 2] = torch.exp(matched_pred_boxes_abs[:, 2]) * W / 4
                matched_pred_boxes_abs[:, 3] = torch.exp(matched_pred_boxes_abs[:, 3]) * H / 4
                
                # Convert to x1,y1,x2,y2 format
                pred_x1 = matched_pred_boxes_abs[:, 0] - matched_pred_boxes_abs[:, 2] / 2
                pred_y1 = matched_pred_boxes_abs[:, 1] - matched_pred_boxes_abs[:, 3] / 2
                pred_x2 = matched_pred_boxes_abs[:, 0] + matched_pred_boxes_abs[:, 2] / 2
                pred_y2 = matched_pred_boxes_abs[:, 1] + matched_pred_boxes_abs[:, 3] / 2
                matched_pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
                
                # Target boxes in x1,y1,x2,y2 format
                target_boxes_xyxy = target_boxes_scaled.clone()
                
                # Calculate individual loss components
                # 1. Focal Loss (for all predictions)
                focal_loss_batch = self.focal_loss(pred_conf[b], target_conf)
                focal_loss_sum += self.focal_weight * focal_loss_batch.mean()
                
                # 2. CIoU Loss (for matched boxes only)
                if self.use_ciou and len(matched_pred_boxes_xyxy) > 0:
                    ciou_loss_batch = self.ciou_loss(matched_pred_boxes_xyxy, target_boxes_xyxy)
                    ciou_loss_sum += self.ciou_weight * ciou_loss_batch.mean()
                
                # 3. DIoU Loss (for matched boxes only)
                if self.use_diou and len(matched_pred_boxes_xyxy) > 0:
                    diou_loss_batch = self.diou_loss(matched_pred_boxes_xyxy, target_boxes_xyxy)
                    diou_loss_sum += self.diou_weight * diou_loss_batch.mean()
                
                # 4. Tweak Loss (for matched predictions only)
                if len(matched_pred_conf) > 0:
                    target_conf_matched = torch.ones_like(matched_pred_conf)
                    tweak_loss_batch = self.tweak_loss(matched_pred_conf, target_conf_matched, 
                                                     matched_pred_boxes_xyxy, target_boxes_xyxy)
                    tweak_loss_sum += self.tweak_weight * tweak_loss_batch.mean()
                
                num_matches += len(matched_pred_conf)
        
        # Combine all loss components
        total_loss = focal_loss_sum + ciou_loss_sum + diou_loss_sum + tweak_loss_sum
        
        # Normalize by number of images
        if batch_size > 0:
            total_loss = total_loss / batch_size
        
        return total_loss


def bbox_iou_ciou(pred_boxes, target_boxes):
    """
    Legacy CIoU function - now replaced by HybridLoss class
    Keep for backward compatibility
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
    target_x1, target_y1, target_x2, target_y2 = target_boxes[:,0], target_boxes[:,1], target_boxes[:,2], target_boxes[:,3]

    # IoU
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + 1e-8
    iou = inter_area / union_area

    # center distance
    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2
    target_cx = (target_x1 + target_x2) / 2
    target_cy = (target_y1 + target_y2) / 2
    center_dist = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2

    # enclosing box diagonal
    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)
    c2 = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-8

    # aspect ratio consistency (v)
    w_pred = pred_x2 - pred_x1
    h_pred = pred_y2 - pred_y1
    w_target = target_x2 - target_x1
    h_target = target_y2 - target_y1
    v = (4 / (3.14159265**2)) * (torch.atan(w_target / (h_target + 1e-8)) - torch.atan(w_pred / (h_pred + 1e-8)))**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-8)

    ciou = iou - (center_dist / c2 + alpha * v)
    return 1 - ciou

def train():
    # dataset paths
    train_images = Path(DATASET_ROOT) / 'train' / 'images'
    train_labels = Path(DATASET_ROOT) / 'train' / 'labels'
    val_images = Path(DATASET_ROOT) / 'valid' / 'images'
    val_labels = Path(DATASET_ROOT) / 'valid' / 'labels'

    train_ds = YoloDataset(str(train_images), str(train_labels), img_size=IMG_SIZE, augment=True)
    val_ds = YoloDataset(str(val_images), str(val_labels), img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = YOLOv13(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Use the new Hybrid Loss function
    criterion = HybridLoss(
        focal_weight=1.0,      # Weight for focal loss component
        ciou_weight=1.0,       # Weight for CIoU loss component  
        diou_weight=0.5,       # Weight for DIoU loss component
        tweak_weight=0.3,      # Weight for tweak loss component
        use_ciou=True,         # Enable CIoU loss
        use_diou=True,         # Enable DIoU loss
        focal_gamma=2.0,       # Focal loss gamma parameter
        focal_alpha=0.25,      # Focal loss alpha parameter
        img_size=IMG_SIZE
    )

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for imgs, targets, _ in train_loader:
            imgs = imgs.to(DEVICE)
            # targets is list of numpy arrays (per image)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{EPOCHS} - Train loss: {avg_train_loss:.4f}")

        # validation (basic)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(DEVICE)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"           Val loss:   {avg_val_loss:.4f}")

        # save checkpoint
        ckpt_path = CHECKPOINT_DIR / f'yolov13_epoch{epoch}.pth'
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)

    print('Training complete.')


def infer(image_path: str, checkpoint: str = None, conf_thresh: float = 0.25, save: bool = True):
    model = YOLOv13(num_classes=NUM_CLASSES).to(DEVICE)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        print(f'Loaded checkpoint: {checkpoint}')
    model.eval()

    img = cv2.imread(image_path)
    assert img is not None, 'Image not found'
    orig = img.copy()
    img, ratio, pad = letterbox(img, IMG_SIZE, auto=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))[None,...]
    img_t = torch.from_numpy(img).to(DEVICE)

    with torch.no_grad():
        preds = model(img_t)
    # preds list [o3,o4,o5], each (B,5+nc,H,W)
    # convert preds to boxes by simple thresholding on objectness
    boxes = []
    for p in preds:
        p = p[0].detach().cpu().numpy()  # (5+nc,H,W)
        conf_map = p[0]
        box_maps = p[1:5]
        cls_map = p[5:] if p.shape[0]>5 else np.zeros((NUM_CLASSES,)+conf_map.shape)
        H,W = conf_map.shape
        ys, xs = np.where(conf_map > conf_thresh)
        for y,x in zip(ys,xs):
            conf = conf_map[y,x]
            ctrx = x + 0.5
            ctry = y + 0.5
            # decode box (this decode is heuristic because training uses raw values)
            bx = box_maps[0][y,x]
            by = box_maps[1][y,x]
            bw = box_maps[2][y,x]
            bh = box_maps[3][y,x]
            # approximate to pixel coords
            # map cell coord to image coord
            cell_w = IMG_SIZE / W
            cell_h = IMG_SIZE / H
            cx = (ctrx) * cell_w
            cy = (ctry) * cell_h
            x1 = cx - abs(bw)/2
            y1 = cy - abs(bh)/2
            x2 = cx + abs(bw)/2
            y2 = cy + abs(bh)/2
            cls_scores = cls_map[:,y,x] if cls_map.size>0 else np.array([1.0])
            cls_id = int(np.argmax(cls_scores))
            boxes.append([x1, y1, x2, y2, conf, cls_id])

    # NMS (simple)
    boxes = np.array(boxes)
    if boxes.size == 0:
        print('No detections')
        return []

    keep = simple_nms(boxes, iou_thresh=0.45)
    boxes = boxes[keep]

    # draw
    for b in boxes:
        x1,y1,x2,y2,conf,cls_id = map(float, b)
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(IMG_SIZE-1, x2))
        y2 = int(min(IMG_SIZE-1, y2))
        cv2.rectangle(orig, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(orig, f'elephant {conf:.2f}', (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    if save:
        outp = Path('runs')
        outp.mkdir(exist_ok=True)
        out_file = outp / (Path(image_path).stem + '_pred.jpg')
        cv2.imwrite(str(out_file), orig)
        print('Saved:', out_file)
    return boxes


def simple_nms(boxes: np.ndarray, iou_thresh=0.45):
    # boxes: (N,6) x1,y1,x2,y2,conf,cls
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]
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
        area_i = (x2[i]-x1[i])*(y2[i]-y1[i])
        area_others = (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]])
        union = area_i + area_others - inter
        iou = inter / (union + 1e-8)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds+1]
    return keep

# ----------------------- CLI -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    if args.train:
        train()
    elif args.infer:
        if args.image is None:
            print('Provide --image path for inference')
        else:
            infer(args.image, checkpoint=args.checkpoint)
    else:
        print('Please specify --train or --infer')
