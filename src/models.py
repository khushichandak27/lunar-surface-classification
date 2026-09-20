"""
Model architectures for Lunar Surface Classification.
Implements:
- ResNet-18 & ResNet-34
- EfficientNet-B0 & EfficientNet-B2
- ConvNeXt-Tiny
- DualBranchResNet (fusing optical appearance and shape relief)
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)

class DualBranchResNet(nn.Module):
    """
    Dual-branch deep architecture:
    - Branch 1: Encodes optical surface appearance [I, I, I]
    - Branch 2: Encodes topographic relief gradients [E, E, E]
    - Fusion Head: Merges both representations via late feature concatenation
    """
    def __init__(self, backbone_type: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        m1 = resnet18(weights=weights)
        m2 = resnet18(weights=weights)
        
        self.branch_optical = nn.Sequential(*list(m1.children())[:-1])
        self.branch_relief = nn.Sequential(*list(m2.children())[:-1])
        
        self.fusion_head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x_optical: torch.Tensor, x_relief: torch.Tensor):
        f_opt = torch.flatten(self.branch_optical(x_optical), 1)
        f_rel = torch.flatten(self.branch_relief(x_relief), 1)
        f_fused = torch.cat([f_opt, f_rel], dim=1)
        return self.fusion_head(f_fused)

def build_resnet18(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    return model

def build_resnet34(pretrained: bool = True) -> nn.Module:
    weights = ResNet34_Weights.DEFAULT if pretrained else None
    model = resnet34(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    return model

def build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    return model

def build_efficientnet_b2(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
    model = efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    return model

def build_convnext_tiny(pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    return model
