"""
Hierarchy Module — Cascading High-to-Low Feature Fusion with VSA/HDC
======================================================================
Adapted from VioPose's hierarchy module that cascades high-level semantics
down to low-level features, preserving both global context and fine detail.

Enhanced with VSA/HDC binding (from arthedain-1) for role-filler representations:
  - High-level features (object class, scene context) act as "roles"
  - Low-level features (edges, motion, texture) act as "fillers"
  - Binding creates robust hyperdimensional representations

This preserves VioPose's strength in handling subtle/fast motions and
occlusions by maintaining multi-scale feature alignment.

Original VioPose: https://github.com/SeongJong-Yoo/VioPose
Paper: https://arxiv.org/pdf/2411.13607
VSA/HDC: https://github.com/Enotrium/arthedain-1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from .vsa_hdc import VSAHDC


class CascadedFusionBlock(nn.Module):
    """
    A single level of the hierarchy: fuses high-level context with
    low-level features, optionally using VSA binding.

    High-level → upsampled → fused with low-level → refined.
    """

    def __init__(
        self,
        high_dim: int,
        low_dim: int,
        out_dim: int,
        use_vsa: bool = True,
        hd_dim: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_vsa = use_vsa
        self.out_dim = out_dim

        # Feature alignment
        self.high_proj = nn.Conv2d(high_dim, out_dim, 1)
        self.low_proj = nn.Conv2d(low_dim, out_dim, 1)

        # Fusion (with or without VSA)
        if use_vsa:
            self.vsa = VSAHDC(
                hd_dim=hd_dim,
                input_dim=out_dim,
                dtype="bipolar",
                binding="circular",
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            # Standard attention fusion
            self.attn = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )

        self.dropout = nn.Dropout2d(dropout)

    def forward(
        self,
        high_feat: torch.Tensor,
        low_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse high-level into low-level features.

        Args:
            high_feat: [B, C_high, H, W] high-level feature map
            low_feat: [B, C_low, H', W'] low-level feature map

        Returns:
            [B, out_dim, H', W'] fused feature map
        """
        # Project both to same channel dim
        high_proj = self.high_proj(high_feat)
        low_proj = self.low_proj(low_feat)

        # Upsample high to match low spatial size
        if high_proj.shape[-2:] != low_proj.shape[-2:]:
            high_proj = F.interpolate(
                high_proj, size=low_proj.shape[-2:], mode="bilinear", align_corners=False
            )

        if self.use_vsa:
            # VSA binding: high (role) ⊗ low (filler)
            B, C, H, W = high_proj.shape
            high_flat = high_proj.permute(0, 2, 3, 1).reshape(B * H * W, C)
            low_flat = low_proj.permute(0, 2, 3, 1).reshape(B * H * W, C)

            high_hd = self.vsa.encode(high_flat)
            low_hd = self.vsa.encode(low_flat)
            bound = self.vsa.bind(high_hd, low_hd)

            # Project back to spatial feature space
            bound_feat = self.vsa.projection.T.to(bound.dtype)  # hd_dim → out_dim
            fused = bound @ bound_feat
            fused = fused.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            fused = self.fusion(fused)
        else:
            # Standard attention-gated fusion
            cat = torch.cat([high_proj, low_proj], dim=1)
            gate = self.attn(cat)
            fused = self.fusion(cat)
            fused = gate * fused + (1 - gate) * low_proj

        fused = self.dropout(fused)
        return fused


class HierarchyModule(nn.Module):
    """
    Hierarchical feature fusion module.
    Cascades features from high-level (semantic) to low-level (detailed),
    preserving VioPose's architecture of progressive refinement.

    The hierarchy flows: Level 0 (coarsest) → Level 1 → ... → Level N (finest).
    Each level fuses the previous (higher/coarser) level with the current
    (lower/finer) features.

    Args:
        level_dims: Channel dimensions for each hierarchy level
        modality_dims: Dict mapping modality names to their feature dimensions
        use_vsa_binding: Enable VSA/HDC binding at each level
        hd_dim: Hyperdimensional dimension for VSA
        dropout: Dropout rate
    """

    def __init__(
        self,
        level_dims: List[int] = [2048, 1024, 512, 256],
        use_vsa_binding: bool = True,
        hd_dim: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_levels = len(level_dims)
        self.level_dims = level_dims
        self.use_vsa_binding = use_vsa_binding
        self.hd_dim = hd_dim

        # Initial high-level processor (Level 0)
        self.initial_processor = nn.Sequential(
            nn.Conv2d(level_dims[0], level_dims[0], 3, padding=1),
            nn.BatchNorm2d(level_dims[0]),
            nn.ReLU(inplace=True),
        )

        # Cascaded fusion blocks for levels 1..N
        self.cascade_blocks = nn.ModuleList()
        for i in range(1, self.num_levels):
            self.cascade_blocks.append(
                CascadedFusionBlock(
                    high_dim=level_dims[i - 1],
                    low_dim=level_dims[i],
                    out_dim=level_dims[i],
                    use_vsa=use_vsa_binding,
                    hd_dim=hd_dim,
                    dropout=dropout,
                )
            )

        # Final refinement (Level N output)
        self.final_processor = nn.Sequential(
            nn.Conv2d(level_dims[-1], level_dims[-1], 3, padding=1),
            nn.BatchNorm2d(level_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(level_dims[-1], level_dims[-1], 1),
        )

        # VSA encoder for global HD representation (optional)
        if use_vsa_binding:
            self.vsa_global = VSAHDC(
                hd_dim=hd_dim,
                input_dim=level_dims[0],
                dtype="bipolar",
                binding="circular",
            )

    def forward(
        self,
        multiscale_features: List[torch.Tensor],
        global_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Cascade high→low features through the hierarchy.

        Args:
            multiscale_features: List of feature maps from coarsest to finest
                                [feat_l0, feat_l1, ..., feat_lN]
            global_features: Optional global descriptor [B, D]

        Returns:
            dict with:
                - "fused_features": Final fused feature map
                - "level_features": List of features at each level
                - "hd_global": HD global representation (if VSA enabled)
        """
        assert len(multiscale_features) == self.num_levels, (
            f"Expected {self.num_levels} feature levels, got {len(multiscale_features)}"
        )

        level_features = []

        # Level 0: Initial processing (highest semantic level)
        l0 = self.initial_processor(multiscale_features[0])
        level_features.append(l0)

        # Cascade through levels 1..N
        current = l0
        for i, block in enumerate(self.cascade_blocks):
            low_feat = multiscale_features[i + 1]
            current = block(current, low_feat)
            level_features.append(current)

        # Final refinement
        fused = self.final_processor(current)

        result = {
            "fused_features": fused,
            "level_features": level_features,
        }

        # VSA global encoding
        if self.use_vsa_binding and global_features is not None:
            hd_global = self.vsa_global.encode(global_features)
            result["hd_global"] = hd_global

        return result

    def reconstruct_from_hd(
        self,
        hd_representation: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Reconstruct approximate feature map from HD representation.
        Demonstrates VSA memory capability — robust to noise/sparsity.

        Args:
            hd_representation: [B, hd_dim] hyperdimensional vector
            target_size: (H, W) target spatial dimensions

        Returns:
            [B, level_dims[0], H, W] approximate reconstruction
        """
        B = hd_representation.shape[0]
        # Project HD back to feature space
        feat = hd_representation @ self.vsa_global.projection.T
        feat = feat.reshape(B, self.level_dims[0], 1, 1)
        feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
        return feat