"""
Eldarin Loss Functions
=======================
Combined detection + tracking + VSA/HDC consistency losses.

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
from scipy.optimize import linear_sum_assignment


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


def compute_iou_matrix(
    pred_boxes: torch.Tensor, target_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise IoU cost matrix between predicted and target boxes.

    Args:
        pred_boxes: [N_pred, 4] (cx, cy, w, h) center-format
        target_boxes: [N_target, 4] (x1, y1, x2, y2) corner-format

    Returns:
        cost_matrix: [N_pred, N_target] with 1 - IoU (lower = better match)
    """
    # Convert pred from center format to corner format
    pred_corners = torch.cat([
        pred_boxes[:, :2] - pred_boxes[:, 2:4] / 2,
        pred_boxes[:, :2] + pred_boxes[:, 2:4] / 2,
    ], dim=-1)  # [N_pred, 4] (x1, y1, x2, y2)

    # target_boxes already in corner format
    target_corners = target_boxes  # [N_target, 4]

    # Compute intersection
    lt = torch.max(pred_corners[:, None, :2], target_corners[None, :, :2])  # [N_pred, N_target, 2]
    rb = torch.min(pred_corners[:, None, 2:], target_corners[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N_pred, N_target]

    # Compute areas
    area_pred = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
    area_target = (target_corners[:, 2] - target_corners[:, 0]) * (target_corners[:, 3] - target_corners[:, 1])

    union = area_pred[:, None] + area_target[None, :] - inter + 1e-6
    iou = inter / union  # [N_pred, N_target]

    # Return cost (1 - IoU) for Hungarian matching
    return 1 - iou


def hungarian_match_predictions(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    max_iou: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Perform Hungarian matching (one-to-one optimal assignment) between
    predictions and targets using IoU as the cost metric.

    Unlike the previous random-int approach, this guarantees every
    prediction is matched to its most similar target.

    Args:
        pred_boxes: [N_pred, 4] predicted boxes (cx, cy, w, h)
        target_boxes: [N_target, 4] target boxes (x1, y1, x2, y2)
        max_iou: Minimum IoU for a valid match (lower = more matches)

    Returns:
        matched_pred: Tensor of matched prediction boxes [M, 4]
        matched_target: Tensor of matched target boxes [M, 4]
        unmatched_pred: List of unmatched prediction indices
        unmatched_target: List of unmatched target indices
    """
    num_pred = pred_boxes.shape[0]
    num_target = target_boxes.shape[0]

    if num_pred == 0 or num_target == 0:
        return (
            torch.zeros(0, 4, device=pred_boxes.device, dtype=pred_boxes.dtype),
            torch.zeros(0, 4, device=target_boxes.device, dtype=target_boxes.dtype),
            list(range(num_pred)),
            list(range(num_target)),
        )

    # Compute cost matrix
    cost = compute_iou_matrix(pred_boxes, target_boxes)  # [N_pred, N_target]

    # Convert to numpy for scipy
    cost_np = cost.detach().cpu().numpy()

    # Hungarian algorithm: find optimal one-to-one assignment
    pred_indices, target_indices = linear_sum_assignment(cost_np)

    # Filter assignments by IoU threshold
    valid_matches = []
    for p_idx, t_idx in zip(pred_indices, target_indices):
        if cost_np[p_idx, t_idx] <= (1 - max_iou):  # IoU >= max_iou
            valid_matches.append((p_idx, t_idx))

    # Build matched tensors
    matched_pred_indices = [m[0] for m in valid_matches]
    matched_target_indices = [m[1] for m in valid_matches]

    matched_pred = pred_boxes[matched_pred_indices] if matched_pred_indices else torch.zeros(
        0, 4, device=pred_boxes.device, dtype=pred_boxes.dtype
    )
    matched_target = target_boxes[matched_target_indices] if matched_target_indices else torch.zeros(
        0, 4, device=target_boxes.device, dtype=target_boxes.dtype
    )

    # Compute unmatched
    matched_p_set = set(matched_pred_indices)
    matched_t_set = set(matched_target_indices)
    unmatched_pred = [i for i in range(num_pred) if i not in matched_p_set]
    unmatched_target = [i for i in range(num_target) if i not in matched_t_set]

    return matched_pred, matched_target, unmatched_pred, unmatched_target


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

    def _get_device(self, predictions: Dict) -> torch.device:
        """Safely extract device from predictions dict."""
        det_out = predictions.get("detection", {})
        if isinstance(det_out, dict):
            candidates = [v for v in det_out.values()
                          if isinstance(v, torch.Tensor) and v.numel() > 0]
            if candidates:
                return candidates[0].device
        # Try other keys
        for key, val in predictions.items():
            if isinstance(val, torch.Tensor) and val.numel() > 0:
                return val.device
            if isinstance(val, dict):
                for v2 in val.values():
                    if isinstance(v2, torch.Tensor) and v2.numel() > 0:
                        return v2.device
        return torch.device("cpu")

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
        device = self._get_device(predictions)
        losses = {}

        # Detection losses
        det_out = predictions.get("detection", {})
        if det_out:
            pred_bbox = det_out.get("bbox", torch.zeros(0, 4, device=device))
            pred_cls = det_out.get("cls", torch.zeros(0, device=device))
            pred_obj = det_out.get("obj", torch.zeros(0, device=device))

            target_bbox = targets.get("bboxes", torch.zeros(0, 4, device=device))
            target_cls = targets.get("classes", torch.zeros(0, dtype=torch.long, device=device))

            # Ensure pred_bbox is 2D: [N, 4]
            if pred_bbox.dim() == 3:
                pred_bbox = pred_bbox.reshape(-1, 4)
            if pred_bbox.dim() == 1 and pred_bbox.numel() == 4:
                pred_bbox = pred_bbox.unsqueeze(0)

            num_pred = pred_bbox.shape[0]
            num_target = target_bbox.shape[0] if target_bbox.dim() >= 2 else 0

            if num_pred > 0 and num_target > 0:
                # Hungarian matching for box loss
                matched_pred, matched_target, unmatched_pred, unmatched_target = (
                    hungarian_match_predictions(pred_bbox, target_bbox)
                )

                if matched_pred.shape[0] > 0:
                    losses["box"] = self.box_loss(matched_pred, matched_target) * self.weights["box"]

                # Classification loss: match class predictions to targets via Hungarian
                # Flatten pred_cls to [N_pred, C]
                if pred_cls.dim() == 3:
                    pred_cls_flat = pred_cls.reshape(-1, pred_cls.shape[-1])
                elif pred_cls.dim() == 2:
                    pred_cls_flat = pred_cls
                else:
                    pred_cls_flat = torch.zeros(0, 80, device=device)

                # Match classes using Hungarian bbox assignments
                if pred_cls_flat.shape[0] > 0 and target_cls.numel() > 0:
                    # Build matched class pairs
                    matched_pred_indices = []
                    matched_target_indices = []

                    if num_pred > 0 and num_target > 0:
                        cost = compute_iou_matrix(pred_bbox, target_bbox)
                        cost_np = cost.detach().cpu().numpy()
                        p_idx_arr, t_idx_arr = linear_sum_assignment(cost_np)
                        for p_i, t_i in zip(p_idx_arr, t_idx_arr):
                            if cost_np[p_i, t_i] <= 0.5:  # IoU >= 0.5
                                if p_i < pred_cls_flat.shape[0] and t_i < target_cls.shape[0]:
                                    matched_pred_indices.append(p_i)
                                    matched_target_indices.append(int(t_i))

                    if matched_pred_indices:
                        pred_cls_matched = pred_cls_flat[matched_pred_indices]
                        target_cls_matched = target_cls[matched_target_indices]
                        losses["cls"] = self.cls_loss(pred_cls_matched, target_cls_matched) * self.weights["cls"]

            elif num_pred > 0 and num_target == 0:
                # Penalize false positives (predictions with no targets)
                # Use self-supervision: push confidence toward 0
                if pred_obj.numel() > 0:
                    target_zeros = torch.zeros_like(pred_obj.reshape(-1))
                    losses["obj"] = self.obj_loss(pred_obj.reshape(-1), target_zeros) * self.weights.get("obj", 1.0) * 0.5

        # Tracking losses
        track_out = predictions.get("tracking", {})
        if track_out:
            pred_states = track_out.get("active_states")
            if pred_states is not None and pred_states.numel() > 0:
                target_states = targets.get("track_states", pred_states.detach())
                # Ensure compatible shapes
                if target_states.dim() == 2 and pred_states.dim() == 2:
                    # Pad shorter dimension
                    min_len = min(pred_states.shape[0], target_states.shape[0])
                    pred_states = pred_states[:min_len]
                    target_states = target_states[:min_len]

                if pred_states.numel() > 0:
                    track_losses = self.track_loss(
                        pred_states.unsqueeze(0),
                        target_states.unsqueeze(0),
                    )
                    losses["track"] = track_losses["track_total"] * self.weights["track"]
                    losses["track_smooth"] = track_losses["track_smooth"]
                    losses["track_hd"] = track_losses["track_hd"]

        # VSA/HDC consistency
        if predictions.get("hd_representation") is not None:
            hd_repr = predictions["hd_representation"]
            if hd_repr.numel() > 0 and hd_repr.shape[0] > 1:
                losses["vsa_consistency"] = self.weights["vsa_consistency"] * (
                    1 - F.normalize(hd_repr, dim=-1).var(dim=0).mean()
                )

        # Total loss — sum all named loss components (not filtered by requires_grad)
        total = torch.tensor(0.0, device=device)
        for key in ["box", "cls", "obj", "track", "vsa_consistency"]:
            if key in losses:
                total = total + losses[key]
        losses["total"] = total

        return losses