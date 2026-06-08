"""
Visual Encoder for RGB frames
===============================
 visual modality encoder (originally for 2D keypoints).
Now processes full RGB frames via ResNet/CNN backbone → feature pyramid.

Supports multiple backbones: ResNet-18/34/50, EfficientNet-B0, or custom MobileNet for UAV.
Outputs multi-scale feature maps for the hierarchy module.


Paper: https://arxiv.org/pdf/2411.13607
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, efficientnet_b0
from typing import List, Optional, Dict


class ConvBlock(nn.Module):
    """Basic conv-bn-relu block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            ConvBlock(out_channels, out_channels) for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build FPN top-down pathway."""
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        # Output convs
        return [conv(l) for conv, l in zip(self.output_convs, laterals)]


class VisualEncoder(nn.Module):
    """
    RGB/Visual encoder for UAV imagery.
    Extracts multi-scale features for object detection and tracking.

    Architecture:
        Backbone (ResNet/EfficientNet) → FPN → Feature maps at multiple scales
        Output is a set of feature maps + a global descriptor for VSA/HDC encoding.

    Args:
        backbone: "resnet18", "resnet34", "resnet50", "efficientnet-b0"
        pretrained: Use ImageNet pretrained weights
        out_dim: Final global feature dimension (for HD projection)
        fpn_channels: FPN output channels per level
        img_channels: Input image channels (3 for RGB)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        out_dim: int = 1024,
        fpn_channels: int = 256,
        img_channels: int = 3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.out_dim = out_dim
        self.fpn_channels = fpn_channels

        # Build backbone
        if backbone == "resnet18":
            base = resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.feat_channels = [64, 128, 256, 512]
        elif backbone == "resnet34":
            base = resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.feat_channels = [64, 128, 256, 512]
        elif backbone == "resnet50":
            base = resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.feat_channels = [256, 512, 1024, 2048]
        elif backbone == "efficientnet-b0":
            base = efficientnet_b0(pretrained=pretrained)
            self.backbone = base.features
            self.feat_channels = [16, 24, 40, 112, 320]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Adjust first conv for non-RGB input
        if img_channels != 3 and backbone.startswith("resnet"):
            old_conv = self.backbone[0]
            self.backbone[0] = nn.Conv2d(
                img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # FPN for multi-scale features
        self.fpn = FPN(
            in_channels=self.feat_channels[-4:] if len(self.feat_channels) >= 4 else self.feat_channels,
            out_channels=fpn_channels,
        )

        # Global pooling + projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(self.feat_channels[-1], out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

        # Multi-scale output projections (for hierarchy module)
        self.scale_projections = nn.ModuleList([
            nn.Conv2d(fpn_channels, out_dim // 4, 1) for _ in range(len(self.feat_channels[-4:]))
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, 'pretrained'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and not hasattr(m, 'pretrained'):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract intermediate features from backbone.

        Returns list of feature maps at different scales.
        """
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Collect features at key reduction points
            if hasattr(self, '_feature_indices'):
                if i in self._feature_indices:
                    features.append(x)
            else:
                # Heuristic: collect after each major spatial reduction
                if i > 0 and x.shape[-1] != features[-1].shape[-1] if features else True:
                    features.append(x)
        features.append(x)  # Final layer
        return features[-4:] if len(features) >= 4 else features

    def forward(
        self, x: torch.Tensor, return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: RGB frames [B, C, H, W]
            return_all: If True, return all intermediate features

        Returns:
            dict with:
                - "global": Global descriptor [B, out_dim]
                - "multiscale": List of [B, out_dim//4, H_i, W_i] feature maps
                - "fpn_features": List of FPN feature maps
        """
        B = x.shape[0]

        # Extract backbone features
        backbone_features = self.extract_features(x)

        # FPN
        fpn_features = self.fpn(backbone_features)

        # Global descriptor
        global_feat = self.global_pool(backbone_features[-1])
        global_feat = global_feat.view(B, -1)
        global_feat = self.projection(global_feat)

        # Multi-scale projections
        multiscale = []
        for feat, proj in zip(fpn_features, self.scale_projections):
            multiscale.append(proj(feat))

        result = {
            "global": global_feat,
            "multiscale": multiscale,
            "fpn_features": fpn_features,
        }

        if return_all:
            result["backbone_features"] = backbone_features

        return result


class LightweightVisualEncoder(VisualEncoder):
    """
    Lightweight visual encoder for UAV onboard deployment.
    Uses reduced channels and MobileNet-style depthwise separable convolutions.
    Suitable for FPGA quantization and SNN conversion.
    """

    def __init__(
        self,
        out_dim: int = 512,
        width_mult: float = 0.5,
    ):
        # Minimal backbone
        super().__init__(
            backbone="resnet18",
            pretrained=False,
            out_dim=out_dim,
            fpn_channels=128,
        )
        self.width_mult = width_mult
        self._apply_width_mult()

    def _apply_width_mult(self):
        """Reduce channel widths."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels > 3:
                    m.out_channels = max(1, int(m.out_channels * self.width_mult))
                    m.weight.data = m.weight.data[:m.out_channels]
                    if m.bias is not None:
                        m.bias.data = m.bias.data[:m.out_channels]
            elif isinstance(m, nn.BatchNorm2d):
                if m.num_features > 3:
                    m.num_features = max(1, int(m.num_features * self.width_mult))
                    m.weight.data = m.weight.data[:m.num_features]
                    m.bias.data = m.bias.data[:m.num_features]

    def forward(self, x: torch.Tensor, return_all: bool = False) -> Dict[str, torch.Tensor]:
        return super().forward(x, return_all)