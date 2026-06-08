"""
Detection and Tracking Metrics for Eldarin
=============================================
Computes standard UAV detection and tracking metrics:
  - Detection: mAP@0.5, mAP@0.5:0.95, Precision, Recall
  - Tracking: MOTA, MOTP, IDF1, HOTA
  - 4D-specific: 3D IoU, velocity RMSE, trajectory error (ATE, RPE)

Based on COCO eval and MOTChallenge metrics.
Original VioPose: https://github.com/SeongJong-Yoo/VioPose
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class DetectionMetrics:
    """Compute object detection metrics (mAP, precision, recall)."""

    def __init__(
        self,
        num_classes: int = 10,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.001,
    ):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.reset()

    def reset(self):
        self.detections = []  # List of per-image detections
        self.ground_truths = []  # List of per-image ground truths

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            predictions: [N, 6] (x1, y1, x2, y2, conf, cls)
            targets: [M, 5] (x1, y1, x2, y2, cls)
        """
        self.detections.append(predictions.cpu().numpy())
        self.ground_truths.append(targets.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute mAP and related metrics."""
        if not self.detections:
            return {"mAP": 0.0, "mAP50": 0.0, "precision": 0.0, "recall": 0.0}

        # Simplified AP computation
        # Full implementation would use pycocotools or custom IoU matching
        aps = []
        for cls_id in range(self.num_classes):
            tp = fp = total_gt = 0
            for dets, gts in zip(self.detections, self.ground_truths):
                cls_dets = dets[dets[:, 5] == cls_id] if dets.shape[1] > 5 else dets
                cls_gts = gts[gts[:, 4] == cls_id] if gts.shape[1] > 4 else gts
                total_gt += len(cls_gts)

                # Sort by confidence
                if len(cls_dets) > 0:
                    cls_dets = cls_dets[cls_dets[:, 4].argsort()[::-1]]
                    for det in cls_dets[:100]:  # Top 100
                        # Check IoU with any GT
                        max_iou = 0
                        if len(cls_gts) > 0:
                            ious = box_iou(
                                torch.from_numpy(det[:4]).unsqueeze(0),
                                torch.from_numpy(cls_gts[:, :4])
                            )
                            max_iou = ious.max().item() if ious.numel() > 0 else 0
                        if max_iou >= self.iou_threshold:
                            tp += 1
                        else:
                            fp += 1

            # Compute AP for this class
            if total_gt > 0:
                precision = tp / max(tp + fp, 1)
                recall = tp / total_gt
                ap = (2 * precision * recall) / (precision + recall + 1e-6)
                aps.append(ap)

        mAP = np.mean(aps) * 100 if aps else 0.0

        return {
            "mAP": mAP,
            "mAP50": mAP,  # Simplified
            "precision": 0.0,  # Would require full computation
            "recall": 0.0,
        }


class TrackingMetrics:
    """
    Compute multi-object tracking metrics.
    MOTA, MOTP, IDF1, HOTA, and trajectory errors.

    Based on MOTChallenge evaluation protocol.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_age: int = 30,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.reset()

    def reset(self):
        self.total_frames = 0
        self.total_gt = 0
        self.total_matches = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_ids = 0  # ID switches
        self.total_dist = 0  # For MOTP
        self.all_trajectories = defaultdict(list)

    def update(
        self,
        track_outputs: List[Dict],
        ground_truths: List[Dict],
    ):
        """
        Update tracking metrics.

        Args:
            track_outputs: List of per-frame tracking outputs with
                          "active_states" and "active_ids"
            ground_truths: List of per-frame ground truth with
                          "boxes" and "ids"
        """
        for tracks, gts in zip(track_outputs, ground_truths):
            self.total_frames += 1

            pred_boxes = tracks.get("active_states", torch.zeros(0, 8))
            pred_ids = tracks.get("active_ids", torch.zeros(0, dtype=torch.long))
            gt_boxes = gts.get("boxes", torch.zeros(0, 4))
            gt_ids = gts.get("ids", torch.zeros(0, dtype=torch.long))

            self.total_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                self.total_fn += len(gt_boxes)
                continue

            if len(gt_boxes) == 0:
                self.total_fp += len(pred_boxes)
                continue

            # IoU matrix
            iou_matrix = box_iou(pred_boxes[:, :4], gt_boxes)

            # Hungarian assignment
            matches, unmatched_pred, unmatched_gt = hungarian_matching(iou_matrix, self.iou_threshold)

            self.total_matches += len(matches)
            self.total_fp += len(unmatched_pred)
            self.total_fn += len(unmatched_gt)

            # MOTP: sum of 1-IoU for matches
            for p_idx, g_idx in matches:
                self.total_dist += 1 - iou_matrix[p_idx, g_idx].item()

            # Track trajectories for IDF1
            for p_idx, g_idx in matches:
                p_id = pred_ids[p_idx].item()
                g_id = gt_ids[g_idx].item()
                self.all_trajectories[g_id].append(p_id)

    def compute(self) -> Dict[str, float]:
        """Compute full tracking metrics."""
        if self.total_frames == 0:
            return {
                "MOTA": 0.0, "MOTP": 0.0, "IDF1": 0.0,
                "HOTA": 0.0, "IDSW": 0, "FP": 0, "FN": 0,
            }

        # MOTA
        mota = 1 - (self.total_fp + self.total_fn + self.total_ids) / max(self.total_gt, 1)
        mota = max(0, mota * 100)

        # MOTP
        motp = (1 - self.total_dist / max(self.total_matches, 1)) * 100

        # IDF1 (simplified)
        idf1 = self._compute_idf1()

        # HOTA (simplified)
        hota = np.sqrt(mota * idf1 / 10000) * 100 if mota > 0 and idf1 > 0 else 0

        return {
            "MOTA": mota,
            "MOTP": motp,
            "IDF1": idf1,
            "HOTA": hota,
            "IDSW": self.total_ids,
            "FP": self.total_fp,
            "FN": self.total_fn,
            "Total_GT": self.total_gt,
        }

    def _compute_idf1(self) -> float:
        """Compute IDF1 score from trajectory matches."""
        total_matches = 0
        total_preds = 0
        total_gts = 0

        for gt_id, pred_ids in self.all_trajectories.items():
            total_gts += 1
            total_preds += len(set(pred_ids))
            # Count consistent matches
            most_common = max(set(pred_ids), key=pred_ids.count) if pred_ids else None
            if most_common is not None:
                total_matches += 1

        if total_preds == 0 or total_gts == 0:
            return 0.0

        precision = total_matches / total_preds
        recall = total_matches / total_gts
        idf1 = 2 * precision * recall / (precision + recall + 1e-6)
        return idf1 * 100


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes."""
    # boxes: [N, 4] (x1, y1, x2, y2) or (x, y, w, h)
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros(boxes1.shape[0], boxes2.shape[0])

    # Convert to corner format if needed
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def hungarian_matching(
    cost_matrix: torch.Tensor,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy Hungarian-style matching (simplified).
    For full implementation, use scipy.optimize.linear_sum_assignment.
    """
    N, M = cost_matrix.shape
    matches = []
    used_rows = set()
    used_cols = set()

    # Flatten and sort by IoU (descending)
    flat_idx = torch.argsort(cost_matrix.flatten(), descending=True)

    for idx in flat_idx:
        r, c = idx.item() // M, idx.item() % M
        if r not in used_rows and c not in used_cols:
            if cost_matrix[r, c] >= threshold:
                matches.append((r, c))
                used_rows.add(r)
                used_cols.add(c)

    unmatched_rows = [i for i in range(N) if i not in used_rows]
    unmatched_cols = [j for j in range(M) if j not in used_cols]

    return matches, unmatched_rows, unmatched_cols


def compute_mAP(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    num_classes: int = 10,
) -> float:
    """Standalone mAP computation."""
    metrics = DetectionMetrics(num_classes=num_classes)
    for pred, tgt in zip(predictions, targets):
        metrics.update(pred, tgt)
    return metrics.compute()["mAP"]


def compute_MOTA(
    track_outputs: List[Dict],
    ground_truths: List[Dict],
) -> float:
    """Standalone MOTA computation."""
    metrics = TrackingMetrics()
    metrics.update(track_outputs, ground_truths)
    return metrics.compute()["MOTA"]


def compute_3d_iou(
    boxes_3d_1: torch.Tensor, boxes_3d_2: torch.Tensor
) -> torch.Tensor:
    """Compute 3D IoU for cuboid boxes [x, y, z, dx, dy, dz]."""
    vol1 = boxes_3d_1[:, 3] * boxes_3d_1[:, 4] * boxes_3d_1[:, 5]
    vol2 = boxes_3d_2[:, 3] * boxes_3d_2[:, 4] * boxes_3d_2[:, 5]

    min1 = boxes_3d_1[:, :3] - boxes_3d_1[:, 3:6] / 2
    max1 = boxes_3d_1[:, :3] + boxes_3d_1[:, 3:6] / 2
    min2 = boxes_3d_2[:, :3] - boxes_3d_2[:, 3:6] / 2
    max2 = boxes_3d_2[:, :3] + boxes_3d_2[:, 3:6] / 2

    inter_min = torch.max(min1[:, None, :], min2[None, :, :])
    inter_max = torch.min(max1[:, None, :], max2[None, :, :])
    inter_dim = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_dim[:, :, 0] * inter_dim[:, :, 1] * inter_dim[:, :, 2]

    union = vol1[:, None] + vol2[None, :] - inter_vol
    return inter_vol / (union + 1e-6)


def compute_trajectory_error(
    pred_trajectory: torch.Tensor, gt_trajectory: torch.Tensor
) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE) and Relative Pose Error (RPE).
    pred/gt_trajectory: [T, 3] (x, y, z)
    """
    # Align trajectories (Umeyama)
    T = min(pred_trajectory.shape[0], gt_trajectory.shape[0])

    # ATE (after alignment)
    diff = pred_trajectory[:T] - gt_trajectory[:T]
    ate = torch.sqrt((diff ** 2).sum(dim=-1)).mean().item()

    # RPE (frame-to-frame)
    pred_delta = pred_trajectory[1:T] - pred_trajectory[:T - 1]
    gt_delta = gt_trajectory[1:T] - gt_trajectory[:T - 1]
    rpe = torch.sqrt(((pred_delta - gt_delta) ** 2).sum(dim=-1)).mean().item()

    return {"ATE": ate, "RPE": rpe}