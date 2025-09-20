import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + 1e-8
        iou = inter_area / union_area
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        center_dist = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2
        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        c2 = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-8
        diou = iou - center_dist / c2
        diou_loss = 1 - diou
        if self.reduction == 'mean':
            return diou_loss.mean()
        elif self.reduction == 'sum':
            return diou_loss.sum()
        else:
            return diou_loss

class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + 1e-8
        iou = inter_area / union_area
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        center_dist = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2
        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        c2 = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-8
        w_pred = pred_boxes[:, 2] - pred_boxes[:, 0]
        h_pred = pred_boxes[:, 3] - pred_boxes[:, 1]
        w_target = target_boxes[:, 2] - target_boxes[:, 0]
        h_target = target_boxes[:, 3] - target_boxes[:, 1]
        v = (4 / (3.14159265**2)) * (torch.atan(w_target / (h_target + 1e-8)) - 
                                     torch.atan(w_pred / (h_pred + 1e-8)))**2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-8)
        ciou = iou - (center_dist / c2 + alpha * v)
        ciou_loss = 1 - ciou
        if self.reduction == 'mean':
            return ciou_loss.mean()
        elif self.reduction == 'sum':
            return ciou_loss.sum()
        else:
            return ciou_loss

class TweakLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_conf, target_conf, pred_boxes, target_boxes):
        conf_diff = torch.abs(pred_conf - target_conf)
        conf_penalty = conf_diff * torch.exp(conf_diff)
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        w_ratio = torch.max(pred_w / (target_w + 1e-8), target_w / (pred_w + 1e-8))
        h_ratio = torch.max(pred_h / (target_h + 1e-8), target_h / (pred_h + 1e-8))
        size_penalty = torch.log(w_ratio + 1e-8) + torch.log(h_ratio + 1e-8)
        tweak_loss = conf_penalty + 0.5 * size_penalty
        if self.reduction == 'mean':
            return tweak_loss.mean()
        elif self.reduction == 'sum':
            return tweak_loss.sum()
        else:
            return tweak_loss

class HybridLoss(nn.Module):
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
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction='none')
        if use_ciou:
            self.ciou_loss = CIoULoss(reduction='none')
        if use_diou:
            self.diou_loss = DIoULoss(reduction='none')
        self.tweak_loss = TweakLoss(reduction='none')
    
    def forward(self, predictions, targets):
        device = predictions[0].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        batch_size = len(targets)
        if batch_size == 0:
            return total_loss
        
        valid_losses = []
        
        for pred_level in predictions:
            B, C, H, W = pred_level.shape
            pred_level = pred_level.permute(0, 2, 3, 1).contiguous()
            pred_level = pred_level.view(B, H * W, C)
            
            for b in range(batch_size):
                if len(targets[b]) == 0:
                    # No targets for this batch item
                    continue
                
                try:
                    # Get target data - FIXED FORMAT HANDLING
                    target_data = targets[b].to(device)
                    target_classes = target_data[:, 0].long()
                    target_boxes_yolo = target_data[:, 1:5]  # [x_center, y_center, width, height] in normalized coords
                    
                    # Convert YOLO format to absolute pixel coordinates correctly
                    target_boxes_abs = target_boxes_yolo.clone()
                    target_boxes_abs[:, 0] *= W  # x_center
                    target_boxes_abs[:, 1] *= H  # y_center  
                    target_boxes_abs[:, 2] *= W  # width
                    target_boxes_abs[:, 3] *= H  # height
                    
                    # Convert to x1, y1, x2, y2 format
                    x1 = target_boxes_abs[:, 0] - target_boxes_abs[:, 2] / 2
                    y1 = target_boxes_abs[:, 1] - target_boxes_abs[:, 3] / 2
                    x2 = target_boxes_abs[:, 0] + target_boxes_abs[:, 2] / 2
                    y2 = target_boxes_abs[:, 1] + target_boxes_abs[:, 3] / 2
                    
                    target_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    # Create confidence targets
                    target_conf = torch.zeros(H * W, device=device)
                    
                    # Assign positive samples to grid cells
                    grid_x = torch.clamp((target_boxes_abs[:, 0] / W * W).long(), 0, W - 1)
                    grid_y = torch.clamp((target_boxes_abs[:, 1] / H * H).long(), 0, H - 1)
                    grid_idx = grid_y * W + grid_x
                    
                    # Ensure valid indices
                    valid_idx = (grid_idx >= 0) & (grid_idx < H * W)
                    if valid_idx.sum() == 0:
                        continue
                    
                    grid_idx = grid_idx[valid_idx]
                    target_boxes_xyxy = target_boxes_xyxy[valid_idx]
                    
                    target_conf[grid_idx] = 1.0
                    
                    # Get predictions for this batch item
                    pred_conf = torch.sigmoid(pred_level[b, :, 0])
                    pred_boxes_raw = pred_level[b, :, 1:5]
                    
                    # Focal loss
                    focal_loss_batch = self.focal_loss(pred_conf, target_conf)
                    focal_loss_val = self.focal_weight * focal_loss_batch.mean()
                    
                    if torch.isfinite(focal_loss_val):
                        valid_losses.append(focal_loss_val)
                    
                    # Only compute box losses if we have valid targets
                    if len(grid_idx) > 0:
                        # Get matched predictions
                        matched_pred_boxes_raw = pred_boxes_raw[grid_idx]
                        matched_pred_conf = pred_conf[grid_idx]
                        
                        # Convert predictions to absolute coordinates
                        grid_x_matched = grid_idx % W
                        grid_y_matched = grid_idx // W
                        
                        pred_x = (torch.sigmoid(matched_pred_boxes_raw[:, 0]) + grid_x_matched.float()) * (self.img_size / W)
                        pred_y = (torch.sigmoid(matched_pred_boxes_raw[:, 1]) + grid_y_matched.float()) * (self.img_size / H)
                        pred_w = torch.exp(matched_pred_boxes_raw[:, 2]) * (self.img_size / W)
                        pred_h = torch.exp(matched_pred_boxes_raw[:, 3]) * (self.img_size / H)
                        
                        # Convert to x1, y1, x2, y2
                        pred_x1 = pred_x - pred_w / 2
                        pred_y1 = pred_y - pred_h / 2
                        pred_x2 = pred_x + pred_w / 2
                        pred_y2 = pred_y + pred_h / 2
                        
                        matched_pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
                        
                        # Scale target boxes to same coordinate system
                        target_boxes_scaled = target_boxes_xyxy * (self.img_size / W)
                        
                        # CIoU loss
                        if self.use_ciou:
                            try:
                                ciou_loss_batch = self.ciou_loss(matched_pred_boxes_xyxy, target_boxes_scaled)
                                ciou_loss_val = self.ciou_weight * ciou_loss_batch.mean()
                                if torch.isfinite(ciou_loss_val):
                                    valid_losses.append(ciou_loss_val)
                            except:
                                pass
                        
                        # DIoU loss
                        if self.use_diou:
                            try:
                                diou_loss_batch = self.diou_loss(matched_pred_boxes_xyxy, target_boxes_scaled)
                                diou_loss_val = self.diou_weight * diou_loss_batch.mean()
                                if torch.isfinite(diou_loss_val):
                                    valid_losses.append(diou_loss_val)
                            except:
                                pass
                        
                        # Tweak loss
                        try:
                            target_conf_matched = torch.ones_like(matched_pred_conf)
                            tweak_loss_batch = self.tweak_loss(matched_pred_conf, target_conf_matched, 
                                                             matched_pred_boxes_xyxy, target_boxes_scaled)
                            tweak_loss_val = self.tweak_weight * tweak_loss_batch.mean()
                            if torch.isfinite(tweak_loss_val):
                                valid_losses.append(tweak_loss_val)
                        except:
                            pass
                
                except Exception as e:
                    # Skip this batch item if there's an error
                    continue
        
        # Combine all valid losses
        if len(valid_losses) > 0:
            total_loss = torch.stack(valid_losses).sum()
        else:
            # Return a small positive loss if no valid losses
            total_loss = torch.tensor(0.01, device=device, requires_grad=True)
        
        return total_loss
    
    def get_loss_info(self):
        return {
            'focal_weight': self.focal_weight,
            'ciou_weight': self.ciou_weight if self.use_ciou else 0,
            'diou_weight': self.diou_weight if self.use_diou else 0,
            'tweak_weight': self.tweak_weight,
            'components': ['Focal'] + 
                         (['CIoU'] if self.use_ciou else []) +
                         (['DIoU'] if self.use_diou else []) +
                         ['Tweak']
        }

if __name__ == "__main__":
    print("Testing Fixed Loss Functions...")
    batch_size = 2
    num_boxes = 3
    pred_boxes = torch.rand(num_boxes, 4) * 100
    target_boxes = torch.rand(num_boxes, 4) * 100
    focal_loss = FocalLoss()
    ciou_loss = CIoULoss()
    diou_loss = DIoULoss()
    pred_conf = torch.rand(num_boxes)
    target_conf = torch.randint(0, 2, (num_boxes,)).float()
    print(f"Focal Loss: {focal_loss(pred_conf, target_conf).item():.4f}")
    print(f"CIoU Loss: {ciou_loss(pred_boxes, target_boxes).item():.4f}")
    print(f"DIoU Loss: {diou_loss(pred_boxes, target_boxes).item():.4f}")
    hybrid_loss = HybridLoss()
    print(f"\nFixed Hybrid Loss Configuration:")
    for key, value in hybrid_loss.get_loss_info().items():
        print(f"  {key}: {value}")
