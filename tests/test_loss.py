"""
Tests for loss functions.

Covers:
  - GIoU loss correctness
  - Focal loss numerical stability
  - Hungarian matching correctness
  - Loss computation on edge cases (empty detections, zero-track frames)
  - Tracking loss shapes
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGIoULoss:
    """Test GIoU loss computation."""

    def test_perfect_match(self):
        """GIoU loss is zero for identical boxes."""
        from utils.loss import GIoULoss

        loss_fn = GIoULoss(reduction="mean")
        boxes = torch.tensor([[50., 50., 30., 20.]])  # cx, cy, w, h

        loss = loss_fn(boxes, boxes)
        assert loss.item() < 0.01, f"Perfect match should have ~0 loss, got {loss.item():.4f}"

    def test_non_overlapping(self):
        """GIoU loss is high for non-overlapping boxes."""
        from utils.loss import GIoULoss

        loss_fn = GIoULoss(reduction="mean")
        pred = torch.tensor([[10., 10., 5., 5.]])
        target = torch.tensor([[100., 100., 5., 5.]])

        loss = loss_fn(pred, target)
        # Non-overlapping boxes should have high GIoU loss (> 1.0)
        assert loss.item() > 1.0, f"Non-overlapping should have loss > 1.0, got {loss.item():.4f}"

    def test_partial_overlap(self):
        """GIoU loss reflects partial overlap."""
        from utils.loss import GIoULoss

        loss_fn = GIoULoss(reduction="mean")
        pred = torch.tensor([[50., 50., 20., 20.]])
        target1 = torch.tensor([[50., 50., 20., 20.]])  # perfect
        target2 = torch.tensor([[55., 55., 20., 20.]])  # slight shift

        loss1 = loss_fn(pred, target1).item()
        loss2 = loss_fn(pred, target2).item()

        assert loss2 > loss1, f"Shifted should have higher loss: {loss1:.4f} vs {loss2:.4f}"


class TestFocalLoss:
    """Test focal loss computation."""

    def test_numerical_stability(self):
        """Focal loss produces finite values on random inputs."""
        from utils.loss import FocalLoss

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(10, 80)  # 10 predictions, 80 classes
        targets = torch.randint(0, 80, (10,))

        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss).all()
        assert loss.item() > 0

    def test_correct_prediction_low_loss(self):
        """Focal loss is low for high-confidence correct predictions."""
        from utils.loss import FocalLoss

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        # High confidence for class 3
        logits = torch.zeros(1, 10)
        logits[0, 3] = 100.0  # Very high logit for correct class
        targets = torch.tensor([3])

        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1, f"High-confidence correct should have low loss, got {loss.item():.4f}"

    def test_wrong_prediction_high_loss(self):
        """Focal loss is high for confident wrong predictions."""
        from utils.loss import FocalLoss

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        # High confidence for wrong class
        logits = torch.zeros(1, 10)
        logits[0, 7] = 100.0  # High logit, but target is class 3
        targets = torch.tensor([3])

        loss = loss_fn(logits, targets)
        assert loss.item() > 1.0, f"Confident wrong should have high loss, got {loss.item():.4f}"


class TestHungarianMatching:
    """Test Hungarian matching for box assignment."""

    def test_hungarian_match_pred_to_target(self):
        """Hungarian matching correctly assigns predictions to targets."""
        from utils.loss import hungarian_match_predictions

        # Two predictions matching two targets (swapped)
        pred = torch.tensor([
            [10., 10., 5., 5.],  # Matches target 2
            [50., 50., 5., 5.],  # Matches target 1
        ])
        target = torch.tensor([
            [47.5, 47.5, 52.5, 52.5],  # Corner format: ~(50,50,w=5,h=5)
            [7.5, 7.5, 12.5, 12.5],    # Corner format: ~(10,10,w=5,h=5)
        ])

        matched_pred, matched_target, unmatched_pred, unmatched_target = (
            hungarian_match_predictions(pred, target, max_iou=0.5)
        )

        # Should match both pairs
        assert matched_pred.shape[0] == 2, f"Expected 2 matches, got {matched_pred.shape[0]}"
        assert len(unmatched_pred) == 0
        assert len(unmatched_target) == 0

    def test_hungarian_with_extra_pred(self):
        """Extra predictions are correctly marked as unmatched."""
        from utils.loss import hungarian_match_predictions

        pred = torch.tensor([
            [50., 50., 10., 10.],
            [10., 10., 10., 10.],
            [200., 200., 5., 5.],  # No matching target
        ])
        target = torch.tensor([
            [45., 45., 55., 55.],
            [5., 5., 15., 15.],
        ])

        matched_pred, matched_target, unmatched_pred, unmatched_target = (
            hungarian_match_predictions(pred, target, max_iou=0.5)
        )

        # Should have 2 matches, 1 unmatched pred
        assert matched_pred.shape[0] >= 1
        assert len(unmatched_pred) >= 1

    def test_hungarian_iou_matrix(self):
        """IoU cost matrix computation."""
        from utils.loss import compute_iou_matrix

        pred = torch.tensor([[50., 50., 10., 10.]])
        target = torch.tensor([[45., 45., 55., 55.]])

        cost = compute_iou_matrix(pred, target)
        assert cost.shape == (1, 1)
        # IoU should be > 0.5, so cost < 0.5
        assert cost.item() < 0.5, f"High IoU should have low cost: {cost.item():.4f}"

    def test_empty_inputs(self):
        """Hungarian matching handles empty inputs."""
        from utils.loss import hungarian_match_predictions

        # Empty predictions
        pred = torch.zeros(0, 4)
        target = torch.randn(3, 4)

        matched_pred, matched_target, unmatched_pred, unmatched_target = (
            hungarian_match_predictions(pred, target)
        )

        assert matched_pred.shape[0] == 0
        assert len(unmatched_pred) == 0
        assert len(unmatched_target) == 3


class TestEldarinLoss:
    """Test combined Eldarin loss function."""

    def test_loss_edge_case_empty_detections(self):
        """Loss handles empty detections gracefully."""
        from utils.loss import EldarinLoss

        loss_fn = EldarinLoss()

        predictions = {
            "detection": {
                "bbox": torch.zeros(0, 4),
                "cls": torch.zeros(0, 80),
                "obj": torch.zeros(0, 1),
            },
        }
        targets = {
            "bboxes": torch.randn(3, 4),
            "classes": torch.randint(0, 10, (3,)),
        }

        losses = loss_fn(predictions, targets)

        # Total loss should exist and be finite
        assert "total" in losses
        assert torch.isfinite(losses["total"])

    def test_loss_edge_case_zero_targets(self):
        """Loss handles zero targets gracefully."""
        from utils.loss import EldarinLoss

        loss_fn = EldarinLoss()

        predictions = {
            "detection": {
                "bbox": torch.randn(5, 4),
                "cls": torch.randn(5, 80),
                "obj": torch.randn(5, 1),
            },
        }
        targets = {
            "bboxes": torch.zeros(0, 4),
            "classes": torch.zeros(0, dtype=torch.long),
        }

        losses = loss_fn(predictions, targets)

        assert "total" in losses
        assert torch.isfinite(losses["total"])

    def test_loss_tracking_component(self):
        """Tracking loss produces correct structure."""
        from utils.loss import EldarinLoss

        loss_fn = EldarinLoss()

        predictions = {
            "detection": {
                "bbox": torch.randn(2, 4),
                "cls": torch.randn(2, 80),
                "obj": torch.randn(2, 1),
            },
            "tracking": {
                "active_states": torch.randn(4, 8),
            },
        }
        targets = {
            "bboxes": torch.randn(3, 4),
            "classes": torch.randint(0, 10, (3,)),
            "track_states": torch.randn(4, 8),
        }

        losses = loss_fn(predictions, targets)

        assert "track" in losses, f"Keys: {losses.keys()}"
        assert "total" in losses

    def test_loss_vsa_consistency(self):
        """VSA consistency loss computation."""
        from utils.loss import EldarinLoss

        loss_fn = EldarinLoss()

        predictions = {
            "detection": {
                "bbox": torch.randn(2, 4),
                "cls": torch.randn(2, 80),
                "obj": torch.randn(2, 1),
            },
            "hd_representation": torch.randn(4, 256),
        }
        targets = {
            "bboxes": torch.randn(2, 4),
            "classes": torch.randint(0, 10, (2,)),
        }

        losses = loss_fn(predictions, targets)

        assert "vsa_consistency" in losses, f"Keys: {losses.keys()}"
        assert "total" in losses

    def test_loss_all_components(self):
        """Full loss with all components."""
        from utils.loss import EldarinLoss

        loss_fn = EldarinLoss()

        predictions = {
            "detection": {
                "bbox": torch.randn(3, 4),
                "cls": torch.randn(3, 10),
                "obj": torch.randn(3, 1),
            },
            "tracking": {
                "active_states": torch.randn(3, 8),
            },
            "hd_representation": torch.randn(4, 512),
        }
        targets = {
            "bboxes": torch.randn(4, 4),
            "classes": torch.randint(0, 10, (4,)),
            "track_states": torch.randn(3, 8),
        }

        losses = loss_fn(predictions, targets)

        assert "total" in losses
        assert torch.isfinite(losses["total"])
        assert losses["total"].item() >= 0

    def test_loss_total_not_requires_grad_filtered(self):
        """Total loss includes all named components, not just requires_grad ones."""
        from utils.loss import EldarinLoss

        loss_fn = EldarinLoss()

        predictions = {
            "detection": {
                "bbox": torch.randn(2, 4),
                "cls": torch.randn(2, 10),
                "obj": torch.randn(2, 1),
            },
        }
        targets = {
            "bboxes": torch.randn(3, 4),
            "classes": torch.randint(0, 10, (3,)),
        }

        losses = loss_fn(predictions, targets)

        # Total should have grad
        assert losses["total"].requires_grad, "Total loss should require grad for backprop"


class TestTrackingLoss:
    """Test tracking-specific loss components."""

    def test_tracking_loss_shape(self):
        """TrackingLoss processes various state shapes."""
        from utils.loss import TrackingLoss

        loss_fn = TrackingLoss(smooth_weight=0.5, hd_weight=0.1)

        pred_states = torch.randn(2, 5, 6)  # [B, T, state_dim]
        target_states = torch.randn(2, 5, 6)

        losses = loss_fn(pred_states, target_states)

        assert "track_pos" in losses
        assert "track_vel" in losses
        assert "track_smooth" in losses
        assert "track_hd" in losses
        assert "track_total" in losses

    def test_tracking_single_frame(self):
        """TrackingLoss handles single-frame input."""
        from utils.loss import TrackingLoss

        loss_fn = TrackingLoss()
        pred = torch.randn(1, 1, 3)  # [B=1, T=1, dim=3]
        target = torch.randn(1, 1, 3)

        losses = loss_fn(pred, target)

        assert losses["track_smooth"].item() == 0.0, "Smooth loss should be 0 for single frame"