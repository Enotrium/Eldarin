"""
Event Stream Encoder (FPGA-Compatible)
=======================================
Adapted from FPGA-Event-Based-encode: https://github.com/Enotrium/FPGA-Event-Based-encode

Processes event camera streams into sparse, SNN-friendly representations.
Optimized for low-latency, high-temporal-resolution UAV processing.

Supports multiple event representations:
  - Voxel grid (sparse 3D convolution compatible)
  - Event frame (2D histogram accumulation)
  - Time surface (per-pixel latest timestamp)
  - Sparse spike tensors (direct SNN input)


Paper: https://arxiv.org/pdf/2411.13607
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class EventVoxelGrid(nn.Module):
    """
    Convert event stream to voxel grid representation.
    Events (x, y, t, p) → sparse 3D tensor [B, T_bins, H, W].

    FPGA-optimized: Uses integer indexing and sparse accumulation.
    Compatible with sparse 3D convolutions (e.g., MinkowskiEngine).
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        num_bins: int = 10,
        normalize: bool = True,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.normalize = normalize

    def forward(
        self, events: Tuple[torch.Tensor, ...], duration_us: float
    ) -> torch.Tensor:
        """
        Convert events to voxel grid.

        Args:
            events: (x, y, t, p) tuple, each [N]
            duration_us: Total time window in microseconds

        Returns:
            Voxel grid [B, num_bins, H, W]
        """
        x, y, t, p = events
        device = x.device
        N = x.shape[0]

        if N == 0:
            return torch.zeros(1, self.num_bins, self.height, self.width, device=device)

        # Normalize time to [0, 1]
        t_norm = (t - t.min()) / (duration_us + 1e-6)

        # Assign to bins
        bin_idx = (t_norm * self.num_bins).long().clamp(0, self.num_bins - 1)

        # Separate polarity
        pos_mask = p > 0
        neg_mask = ~pos_mask

        voxel = torch.zeros(
            1, self.num_bins, self.height, self.width, device=device
        )

        # Accumulate (FPGA-friendly: sparse indexed writes)
        for b in range(self.num_bins):
            mask = bin_idx == b
            if mask.any():
                pos_count = pos_mask[mask].float().sum()
                neg_count = neg_mask[mask].float().sum()
                bin_val = pos_count - neg_count
                # Scatter to spatial positions
                coords = torch.stack([y[mask], x[mask]], dim=-1)
                unique_coords, indices = torch.unique(coords, dim=0, return_inverse=True)
                for i, (cy, cx) in enumerate(unique_coords):
                    if 0 <= cy < self.height and 0 <= cx < self.width:
                        voxel[0, b, cy, cx] += (indices == i).float().sum() * (1 if pos_mask[mask].any() else -1)

        if self.normalize:
            voxel = voxel / (voxel.abs().max() + 1e-6)

        return voxel


class EventFrame(nn.Module):
    """
    Simple 2D event frame accumulation.
    Fast, FPGA-friendly: sum positive and negative events into 2 channels.
    """

    def __init__(self, height: int = 480, width: int = 640):
        super().__init__()
        self.height = height
        self.width = width

    def forward(
        self, events: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Convert events to 2-channel frame [B, 2, H, W].

        Args:
            events: (x, y, t, p) tuple, each [N]

        Returns:
            Event frame [1, 2, H, W]
        """
        x, y, t, p = events
        N = x.shape[0]

        frame = torch.zeros(1, 2, self.height, self.width, device=x.device)

        if N == 0:
            return frame

        pos_mask = p > 0
        neg_mask = ~pos_mask

        # Accumulate by polarity
        for px, py in zip(x[pos_mask].long(), y[pos_mask].long()):
            if 0 <= py < self.height and 0 <= px < self.width:
                frame[0, 0, py, px] += 1

        for nx, ny in zip(x[neg_mask].long(), y[neg_mask].long()):
            if 0 <= ny < self.height and 0 <= nx < self.width:
                frame[0, 1, ny, nx] += 1

        # Log normalization
        frame = torch.log(1 + frame)
        return frame


class TimeSurface(nn.Module):
    """
    Time surface representation: each pixel stores the timestamp
    of the most recent event. Captures motion direction via polarity.
    """

    def __init__(self, height: int = 480, width: int = 640):
        super().__init__()
        self.height = height
        self.width = width

    def forward(
        self, events: Tuple[torch.Tensor, ...], duration_us: float
    ) -> torch.Tensor:
        """
        Create time surface [1, 2, H, W] (pos/neg poles).

        Args:
            events: (x, y, t, p) tuple
            duration_us: Window duration

        Returns:
            Time surface [1, 2, H, W]
        """
        x, y, t, p = events
        device = x.device

        t_norm = (t.float() - t.min()) / (duration_us + 1e-6)

        surface = torch.zeros(1, 2, self.height, self.width, device=device)

        # For each event, update time surface (latest wins)
        for i in range(len(x)):
            px, py = x[i].long().item(), y[i].long().item()
            if 0 <= py < self.height and 0 <= px < self.width:
                ch = 0 if p[i] > 0 else 1
                surface[0, ch, py, px] = max(
                    surface[0, ch, py, px], t_norm[i]
                )

        return surface


class EventEncoder(nn.Module):
    """
    Event stream encoder for Eldarin.
    Integrates FPGA-Event-Based-encode principles:
      - Sparse event accumulation
      - Low-latency representation
      - SNN-compatible output

    Architecture:
        Event representation (voxel/frame/surface) → CNN backbone → feature maps
        Output: multi-scale features + global descriptor for VSA/HDC

    Args:
        height/width: Sensor resolution
        representation: "voxel_grid", "event_frame", "time_surface", "spike_tensor"
        num_bins: Time bins for voxel grid
        out_dim: Output feature dimension
        backbone_channels: Conv backbone channels
        use_sparse: Use sparse convolution (FPGA-optimized)
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        representation: str = "voxel_grid",
        num_bins: int = 10,
        out_dim: int = 512,
        backbone_channels: List[int] = [32, 64, 128, 256],
        use_sparse: bool = True,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.representation_type = representation
        self.out_dim = out_dim
        self.use_sparse = use_sparse

        # Event representation module
        if representation == "voxel_grid":
            self.representation = EventVoxelGrid(height, width, num_bins)
            in_channels = num_bins
        elif representation == "event_frame":
            self.representation = EventFrame(height, width)
            in_channels = 2
        elif representation == "time_surface":
            self.representation = TimeSurface(height, width)
            in_channels = 2
        else:
            raise ValueError(f"Unknown event representation: {representation}")

        # CNN backbone (FPGA-friendly: small kernels, few channels)
        self.backbone = self._build_backbone(in_channels, backbone_channels)

        # Feature pyramid for multi-scale
        self.pyramid_layers = nn.ModuleList([
            nn.Conv2d(c, out_dim // 4, 1)
            for c in backbone_channels[-3:] if len(backbone_channels) >= 3
        ])

        # Global descriptor
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(backbone_channels[-1], out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

        self._init_weights()

    def _build_backbone(
        self, in_channels: int, channels: List[int]
    ) -> nn.ModuleList:
        """Build lightweight CNN backbone for event features."""
        layers = []
        c_prev = in_channels
        stride = 1

        for c in channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_prev, c, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                )
            )
            stride = 2  # Downsample after first layer
            c_prev = c

        return nn.ModuleList(layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        events: Tuple[torch.Tensor, ...],
        duration_us: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for event stream.

        Args:
            events: (x, y, t, p) tuple
            duration_us: Time window in microseconds (for voxel/surface)

        Returns:
            dict with "global" and "multiscale" features
        """
        # Convert events to representation
        if self.representation_type in ("voxel_grid", "time_surface"):
            if duration_us is None:
                duration_us = 10000  # Default 10ms
            feat = self.representation(events, duration_us)
        else:
            feat = self.representation(events)

        # CNN backbone
        x = feat
        backbone_features = []
        for layer in self.backbone:
            x = layer(x)
            backbone_features.append(x)

        # Multi-scale features (last 3 levels)
        multiscale = []
        for feat, proj in zip(backbone_features[-3:], self.pyramid_layers):
            multiscale.append(proj(feat))

        # Global descriptor
        global_feat = self.global_pool(backbone_features[-1])
        B = global_feat.shape[0]
        global_feat = global_feat.view(B, -1)
        global_feat = self.projection(global_feat)

        return {
            "global": global_feat,
            "multiscale": multiscale,
            "raw": feat if not self.use_sparse else None,
        }

    def to_spikes(
        self, features: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Convert continuous features to binary spikes for SNN input.
        Threshold-based rate coding.

        Args:
            features: Continuous features [B, C, H, W]
            threshold: Binary threshold

        Returns:
            Binary spike tensor [B, C, H, W]
        """
        return (features > threshold).float()


class SparseEventEncoder(EventEncoder):
    """
    Sparse event encoder optimized for FPGA deployment.
    Uses quantized representations and sparse operations.
    Compatible with FPGA-Event-Based-encode HLS kernels.

    Reference: https://github.com/Enotrium/FPGA-Event-Based-encode
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        out_dim: int = 256,
        quantize_bits: int = 8,
    ):
        super().__init__(
            height=height,
            width=width,
            representation="voxel_grid",
            num_bins=5,
            out_dim=out_dim,
            backbone_channels=[16, 32, 64, 128],
            use_sparse=True,
        )
        self.quantize_bits = quantize_bits

    def quantize_features(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to fixed-point for FPGA."""
        scale = 2 ** (self.quantize_bits - 1) - 1
        return (x * scale).round().clamp(-scale, scale) / scale

    def forward(
        self,
        events: Tuple[torch.Tensor, ...],
        duration_us: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        result = super().forward(events, duration_us)
        # Quantize for FPGA
        if self.training:
            result["global"] = self.quantize_features(result["global"])
            result["multiscale"] = [
                self.quantize_features(f) for f in result["multiscale"]
            ]
        return result