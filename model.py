import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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
        self.layer1 = nn.Sequential(
            ConvBNAct(32, 64, k=3, s=2, p=1), 
            CSPBlock(64, 64, num_blocks=1)
        )
        self.layer2 = nn.Sequential(
            ConvBNAct(64, 128, k=3, s=2, p=1), 
            CSPBlock(128, 128, num_blocks=2)
        )
        self.layer3 = nn.Sequential(
            ConvBNAct(128, 256, k=3, s=2, p=1), 
            CSPBlock(256, 256, num_blocks=3)
        )
        self.layer4 = nn.Sequential(
            ConvBNAct(256, 512, k=3, s=2, p=1), 
            CSPBlock(512, 512, num_blocks=3)
        )
        self.layer5 = nn.Sequential(
            ConvBNAct(512, 1024, k=3, s=2, p=1), 
            CSPBlock(1024, 1024, num_blocks=1)
        )
    
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
        self.p3 = ConvBNAct(256, channels, k=1, s=1, p=0)
        self.p4 = ConvBNAct(512, channels, k=1, s=1, p=0)
        self.p5 = ConvBNAct(1024, channels, k=1, s=1, p=0)
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))
        self.w3 = nn.Parameter(torch.ones(2))
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
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        td4 = self.fusion([p4, p5_up], self.w1)
        td4 = self.conv_td4(td4)
        td4_up = F.interpolate(td4, size=p3.shape[-2:], mode='nearest')
        td3 = self.fusion([p3, td4_up], self.w2)
        td3 = self.conv_td3(td3)
        td3_down = F.interpolate(td3, size=td4.shape[-2:], mode='nearest')
        bu4 = self.fusion([td4, td3_down], self.w3)
        bu4 = self.conv_bu4(bu4)
        p5_out = self.conv_bu5(p5)
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
        self.num_classes = num_classes
        self.backbone = CSPDarknet53Lite()
        self.neck = BiFPN(channels=256)
        self.head = YOLOHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        f3, f4, f5 = self.neck(c3, c4, c5)
        predictions = self.head(f3, f4, f5)
        return predictions

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self):
        total_params = self.get_num_params()
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'backbone': 'CSPDarknet53Lite',
            'neck': 'BiFPN',
            'head': 'YOLOHead'
        }

if __name__ == "__main__":
    model = YOLOv13(num_classes=1)
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        predictions = model(x)
    print("YOLO v13 Model Test:")
    print(f"Input shape: {x.shape}")
    print("Output shapes:")
    for i, pred in enumerate(predictions):
        print(f"  Scale {i+1}: {pred.shape}")
    info = model.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
