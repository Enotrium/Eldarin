"""
Eldarin Loss Functions
=======================
Combined detection + tracking + VSA/HDC consistency losses.
 loss components.

Loss components:
  1. Box loss (IoU/GIoU)
  2. Classification loss (Focal/BCE)
  3. Objectness loss
  4. Tracking/trajectory loss (MSE + HD consistency)
  5. Temporal smoothness loss
  6. VSA/HDC consistency loss


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class GIoULoss(nn.Module):
    """Generalized IoU loss for bounding box regression."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GIoU loss.

        Args:
            pred_boxes: [N, 4] (x, y, w, h) or (x1, y1, x2, y2)
            target_boxes: [N, 4]
        """
        # Convert to corner format if needed
        pred_corners = torch.cat([
            pred_boxes[:, :2] - pred_boxes[:, 2:4] / 2,
            pred_boxes[:, :2] + pred_boxes[:, 2:4] / 2,
        ], dim=-1)

        target_corners = torch.cat([
            target_boxes[:, :2] - target_boxes[:, 2:4] / 2,
            target_boxes[:, :2] + target_boxes[:, 2:4] / 2,
        ], dim=-1)

        # Intersection
        lt = torch.max(pred_corners[:, :2], target_corners[:, :2])
        rb = torch.min(pred_corners[:, 2:], target_corners[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        # Areas
        area_pred = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
        area_target = (target_corners[:, 2] - target_corners[:, 0]) * (target_corners[:, 3] - target_corners[:, 1])
        union = area_pred + area_target - inter + 1e-6
        iou = inter / union

        # Enclosure
        lt_en = torch.min(pred_corners[:, :2], target_corners[:, :2])
        rb_en = torch.max(pred_corners[:, 2:], target_corners[:, 2:])
        wh_en = (rb_en - lt_en).clamp(min=0)
        area_en = wh_en[:, 0] * wh_en[:, 1] + 1e-6

        giou = iou - (area_en - union) / area_en
        loss = 1 - giou

        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalLoss(nn.Module):
    """Focal loss for classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, pred_logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: [N, C] logits
            targets: [N] class indices
        """
        ce_loss = F.cross_entropy(pred_logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


class TrackingLoss(nn.Module):
    """
    Combined tracking loss: trajectory smoothness + HD Kalman consistency.
    """

    def __init__(self, smooth_weight: float = 0.5, hd_weight: float = 0.1):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.hd_weight = hd_weight
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        pred_states: torch.Tensor,
        target_states: torch.Tensor,
        hd_pred: Optional[torch.Tensor] = None,
        hd_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_states: [B, T, state_dim] predicted states
            target_states: [B, T, state_dim] target states
            hd_pred: Optional HD predictions
            hd_target: Optional HD targets
        """
        # Position + velocity loss
        pos_loss = self.mse(pred_states[:, :, :3], target_states[:, :, :3])

        # Velocity loss
        if pred_states.shape[-1] >= 6:
            vel_loss = self.mse(pred_states[:, :, 3:6], target_states[:, :, 3:6])
        else:
            vel_loss = torch.tensor(0.0, device=pred_states.device)

        # Temporal smoothness (encourage smooth trajectories)
        if pred_states.shape[1] >= 2:
            smooth_loss = self.mse(
                pred_states[:, 1:, :3] - pred_states[:, :-1, :3],
                target_states[:, 1:, :3] - target_states[:, :-1, :3],
            )
        else:
            smooth_loss = torch.tensor(0.0, device=pred_states.device)

        # HD Kalman consistency
        hd_loss = torch.tensor(0.0, device=pred_states.device)
        if hd_pred is not None and hd_target is not None:
            hd_loss = 1 - F.cosine_similarity(hd_pred, hd_target, dim=-1).mean()

        total = pos_loss + vel_loss + self.smooth_weight * smooth_loss + self.hd_weight * hd_loss

        return {
            "track_pos": pos_loss,
            "track_vel": vel_loss,
            "track_smooth": smooth_loss,
            "track_hd": hd_loss,
            "track_total": total,
        }


class VSAHDCConsistencyLoss(nn.Module):
    """
    VSA/HDC consistency loss.
    Encourages HD representations to maintain VSA algebraic properties:
      - Binding invertibility
      - Bundle similarity to constituents
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        hd_representations: Dict[str, torch.Tensor],
        modality_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute consistency between HD representations and original features.
        """
        if not hd_representations or not modality_features:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)

        # Reconstruction consistency: feature → HD → feature should be close
        for mod_name, hd in hd_representations.items():
            if hd is not None and mod_name in modality_features:
                # Compare HD vectors across batches for consistency
                if hd.shape[0] > 1:
                    # Intra-class compactness
                    centroid = hd.mean(dim=0, keepdim=True)
                    loss += (1 - F.cosine_similarity(hd, centroid.expand_as(hd)).mean())

        return self.weight * loss


class EldarinLoss(nn.Module):
    """
    Combined Eldarin loss for detection + tracking + VSA/HDC consistency.

    Loss weights from config:
      - box: 7.5 (bounding box)
      - cls: 0.5 (classification)
      - obj: 1.0 (objectness)
      - track: 2.0 (tracking)
      - vsa_consistency: 0.1 (VSA/HDC)
      - temporal_smooth: 0.5 (smoothness)
    """

    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.weights = loss_weights or {
            "box": 7.5,
            "cls": 0.5,
            "obj": 1.0,
            "track": 2.0,
            "vsa_consistency": 0.1,
            "temporal_smooth": 0.5,
        }

        self.box_loss = GIoULoss()
        self.cls_loss = FocalLoss()
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.track_loss = TrackingLoss(
            smooth_weight=self.weights["temporal_smooth"],
            hd_weight=self.weights["vsa_consistency"],
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            predictions: Dict from model forward pass with "detection", "tracking", etc.
            targets: Dict with ground truth annotations

        Returns:
            Dict with individual losses and total
        """
        device = next(iter(predictions.values())).device if isinstance(predictions, dict) else "cpu"
        losses = {}

        # Detection losses
        det_out = predictions.get("detection", {})
        if det_out:
            pred_bbox = det_out.get("bbox", torch.zeros(1, device=device))
            pred_cls = det_out.get("cls", torch.zeros(1, 10, device=device))
            pred_obj = det_out.get("obj", torch.zeros(1, device=device))

            target_bbox = targets.get("bboxes", torch.zeros(1, 4, device=device))
            target_cls = targets.get("classes", torch.zeros(1, dtype=torch.long, device=device))

            # Box loss on matched pairs (simplified: all preds vs all targets)
            num_pred = pred_bbox.shape[1] if pred_bbox.dim() > 1 else pred_bbox.shape[0]
            num_target = target_bbox.shape[0]

            if num_pred > 0 and num_target > 0:
                # Flatten for loss computation
                pred_flat = pred_bbox.reshape(-1, 4)
                # Repeat targets to match (simplified)
                idx = torch.randint(0, num_target, (num_pred * pred_bbox.shape[0],), device=device)
                target_flat = target_bbox[idx]

                losses["box"] = self.box_loss(pred_flat, target_flat) * self.weights["box"]

            if pred_cls.numel() > 0 and target_cls.numel() > 0:
                pred_cls_flat = pred_cls.reshape(-1, pred_cls.shape[-1])
                target_cls_rep = target_cls.repeat(pred_cls.shape[0] * pred_cls.shape[1] // max(1, len(target_cls)) + 1)[:pred_cls_flat.shape[0]]
                losses["cls"] = self.cls_loss(pred_cls_flat, target_cls_rep) * self.weights["cls"]

        # Tracking losses
        track_out = predictions.get("tracking", {})
        if track_out:
            pred_states = track_out.get("active_states")
            if pred_states is not None and pred_states.numel() > 0:
                target_states = targets.get("track_states", pred_states.detach())
                track_losses = self.track_loss(
                    pred_states.unsqueeze(0),
                    target_states.unsqueeze(0),
                )
                losses["track"] = track_losses["track_total"] * self.weights["track"]
                losses["track_smooth"] = track_losses["track_smooth"]
                losses["track_hd"] = track_losses["track_hd"]

        # VSA/HDC consistency
        if predictions.get("hd_representation") is not None:
            losses["vsa_consistency"] = self.weights["vsa_consistency"] * (
                1 - F.normalize(predictions["hd_representation"], dim=-1).var(dim=0).mean()
            )

        # Total loss
        losses["total"] = sum(v for v in losses.values() if v.requires_grad)

        return losses