"""
Tests for model forward pass shapes across modality combinations.

Covers:
  - DetectionHead forward pass shapes
  - TrackingHead forward pass shapes
  - VisualOdometryVSA step output structure
  - Eldarin model creation with various modality combos
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.detection
class TestDetectionHead:
    """Test DetectionHead forward pass shapes."""

    def test_detection_forward_shape(self):
        """DetectionHead produces correct output shapes."""
        from model.heads import DetectionHead

        head = DetectionHead(in_channels=256, num_classes=10, num_anchors=3, use_3d=True)
        features = torch.randn(2, 256, 20, 20)

        out = head(features)

        B = features.shape[0]
        expected_N = 3 * 20 * 20  # anchors * H * W

        assert out["bbox"].shape == (B, expected_N, 4), f"bbox shape: {out['bbox'].shape}"
        assert out["obj"].shape == (B, expected_N, 1), f"obj shape: {out['obj'].shape}"
        assert out["cls"].shape == (B, expected_N, 10), f"cls shape: {out['cls'].shape}"
        assert out["d3"].shape == (B, expected_N, 3), f"d3 shape: {out['d3'].shape}"
        assert out["vel"].shape == (B, expected_N, 3), f"vel shape: {out['vel'].shape}"

    def test_detection_no_3d(self):
        """DetectionHead without 3D output."""
        from model.heads import DetectionHead

        head = DetectionHead(in_channels=128, num_classes=5, use_3d=False)
        features = torch.randn(4, 128, 32, 32)

        out = head(features)

        assert out["d3"] is None, "d3 should be None when use_3d=False"
        assert out["bbox"].shape[2] == 4
        assert out["cls"].shape[2] == 5

    def test_detection_raw_shape(self):
        """DetectionHead raw output has correct channels."""
        from model.heads import DetectionHead

        head = DetectionHead(in_channels=256, num_classes=10, num_anchors=3, use_3d=True)
        features = torch.randn(1, 256, 10, 10)

        out = head(features)
        raw = out["raw"]

        expected_channels = 3 * (4 + 1 + 10 + 3 + 3)  # 3 anchors * 21 channels
        assert raw.shape[1] == expected_channels, (
            f"raw channels: {raw.shape[1]}, expected {expected_channels}"
        )


@pytest.mark.tracking
class TestTrackingHead:
    """Test TrackingHead forward pass."""

    def test_tracking_forward_shape(self):
        """TrackingHead produces correct output structure."""
        from model.heads import TrackingHead

        head = TrackingHead(state_dim=8, hd_dim=256, feature_dim=64)

        detections = torch.randn(5, 8)  # 5 detections, 8D state
        det_features = torch.randn(5, 64)
        reid_features = torch.randn(5, 64)

        result = head(detections, det_features, reid_features, frame_id=0)

        assert "tracks" in result
        assert "active_states" in result
        assert "active_ids" in result
        assert "matches" in result
        assert "new_tracks" in result

        # New tracks should be 5 (all detections are new with no existing tracks)
        assert result["new_tracks"] == 5

        # Each track should have required keys
        for track in result["tracks"]:
            assert "id" in track
            assert "state" in track
            assert "hd_state" in track
            assert "hits" in track
            assert "age" in track

    def test_tracking_empty_detections(self):
        """TrackingHead handles empty detection input."""
        from model.heads import TrackingHead

        head = TrackingHead(state_dim=8, hd_dim=256, feature_dim=64)

        detections = torch.zeros(0, 8)
        det_features = torch.zeros(0, 64)
        reid_features = torch.zeros(0, 64)

        result = head(detections, det_features, reid_features, frame_id=0)

        # Should return empty results, not crash
        assert result["active_states"].shape[0] == 0
        assert result["active_ids"].shape[0] == 0

    def test_tracking_across_frames(self):
        """Tracking across multiple frames maintains IDs."""
        from model.heads import TrackingHead

        head = TrackingHead(state_dim=8, hd_dim=256, feature_dim=64, min_hits=1)

        tracks = None
        # Simulate one object moving across 3 frames
        for frame in range(3):
            detections = torch.tensor([[100.0 + frame, 100.0, 50.0, 30.0, 0, 0, 0, 0]])
            det_features = torch.randn(1, 64)
            reid_features = torch.randn(1, 64)

            result = head(detections, det_features, reid_features, tracks=tracks, frame_id=frame)
            tracks = result["tracks"]

            # Should have exactly 1 track
            assert len(tracks) == 1, f"Frame {frame}: expected 1 track, got {len(tracks)}"

        # All frames should have same track ID
        assert tracks[0]["hits"] == 3, f"Track should have 3 hits, got {tracks[0]['hits']}"

    def test_reset_tracking(self):
        """TrackingHead reset clears all tracks."""
        from model.heads import TrackingHead

        head = TrackingHead(state_dim=8, hd_dim=256, feature_dim=64)

        detections = torch.randn(3, 8)
        det_features = torch.randn(3, 64)
        reid_features = torch.randn(3, 64)

        result = head(detections, det_features, reid_features, frame_id=0)
        assert len(result["tracks"]) > 0

        head.reset()
        # After reset, next_id should be 0
        assert head.next_id.item() == 0


@pytest.mark.vo
class TestVisualOdometry:
    """Test VisualOdometryVSA pipeline."""

    def test_vo_step_output_structure(self):
        """VO step returns expected keys."""
        from model.vo import VisualOdometryVSA

        vo = VisualOdometryVSA(
            image_height=32,
            image_width=32,
            hd_dim=256,
            cartesian_bins=32,
            n_rotations=16,
            dtype="bipolar",
        )

        test_image = torch.zeros(32, 32)
        test_image[10:20, 10:20] = 1.0

        result = vo.step(image=test_image, num_iterations=3)

        assert "pose" in result
        assert "encoded" in result
        assert "map" in result
        assert "translation_hd" in result
        assert "is_tracking" in result
        assert "map_initialized" in result

        assert isinstance(result["pose"]["h"], float)
        assert isinstance(result["pose"]["v"], float)
        assert isinstance(result["pose"]["r"], float)

    def test_vo_sequence_processing(self):
        """VO processes a sequence of frames."""
        from model.vo import VisualOdometryVSA

        vo = VisualOdometryVSA(
            image_height=16,
            image_width=16,
            hd_dim=128,
            cartesian_bins=16,
            n_rotations=8,
            dtype="bipolar",
        )

        # Generate moving square sequence
        sequence = torch.zeros(5, 16, 16)
        for t in range(5):
            sequence[t, 4 + t:8 + t, 4 + t:8 + t] = 1.0

        result = vo.process_sequence(frames=sequence, num_iterations=2)

        assert result["trajectory"].shape == (5, 4)
        assert result["num_steps"] == 5
        assert len(result["all_outputs"]) == 5

    def test_vo_with_events(self):
        """VO encodes event data."""
        from model.vo import VisualOdometryVSA

        vo = VisualOdometryVSA(
            image_height=16,
            image_width=16,
            hd_dim=128,
            dtype="bipolar",
        )

        x = torch.randint(0, 16, (50,))
        y = torch.randint(0, 16, (50,))
        t = torch.rand(50)
        p = torch.ones(50)

        encoded = vo.encode_frame(events=(x, y, t, p))

        assert encoded.shape == (1, 128)
        assert torch.isfinite(encoded).all()

    def test_vo_memory_growth(self):
        """VO trajectory list doesn't grow unbounded."""
        from model.vo import VisualOdometryVSA

        vo = VisualOdometryVSA(
            image_height=16,
            image_width=16,
            hd_dim=128,
            dtype="bipolar",
            _max_trajectory_frames=50,
        )

        # Simulate 200 frames
        sequence = torch.zeros(200, 16, 16)
        for t in range(200):
            sequence[t, 4:8, 4:8] = 1.0

        result = vo.process_sequence(frames=sequence, num_iterations=1)

        # Trajectory should be bounded
        assert result["trajectory"].shape[0] == 200  # process_sequence stores all separately
        # But the internal list should be bounded
        assert len(vo._trajectory_list) <= 100  # (50 from cap, + steps over max trigger flush)


@pytest.mark.integration
class TestEldarinModel:
    """Test Eldarin model creation."""

    def test_model_creation(self):
        """Model creates without error."""
        from model.eldarin_model import create_eldarin

        config = {
            "data": {"modalities": "rgb"},
            "model": {
                "hd_dim": 256,
                "feature_dim": 128,
                "hidden_dim": 256,
                "tracking": False,  # Disable tracking for faster test
            },
        }

        model = create_eldarin(config_dict=config)
        assert model is not None

    def test_model_forward_rgb(self):
        """Model forward pass with RGB frames."""
        from model.eldarin_model import create_eldarin

        config = {
            "data": {"modalities": "rgb"},
            "model": {
                "hd_dim": 256,
                "feature_dim": 128,
                "hidden_dim": 256,
                "tracking": False,
            },
        }

        model = create_eldarin(config_dict=config)
        model.eval()

        frames = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            output = model(frames=frames)

        assert "detection" in output, f"Keys: {output.keys()}"
        det = output["detection"]
        assert "bbox" in det, f"Detection keys: {det.keys()}"

    def test_model_forward_multimodal(self):
        """Model forward pass with RGB + events."""
        from model.eldarin_model import create_eldarin

        config = {
            "data": {"modalities": "rgb+event"},
            "model": {
                "hd_dim": 256,
                "feature_dim": 128,
                "hidden_dim": 256,
                "tracking": False,
            },
        }

        model = create_eldarin(config_dict=config)
        model.eval()

        frames = torch.randn(1, 3, 64, 64)
        events = (torch.randint(0, 64, (10,)), torch.randint(0, 64, (10,)),
                  torch.rand(10), torch.ones(10))

        with torch.no_grad():
            output = model(frames=frames, events=events)

        assert "detection" in output