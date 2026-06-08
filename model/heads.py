"""
Detection + 4D Tracking Heads
===============================
Adapted from VioPose's pose regression head → object detection + tracking.
The original VioPose regresses 3D joint positions over time; Eldarin outputs:
  - Bounding boxes (xyxy format)
  - Class probabilities
  - 3D positions (x, y, z in camera/world frame)
  - Velocities (dx, dy, dz)
  - Object IDs (tracking)

Uses a YOLO-style detection head with hyperdimensional Kalman filter
for tracking, integrated with VSA/HDC representations.

Original VioPose: https://github.com/SeongJong-Yoo/VioPose
Paper: https://arxiv.org/pdf/2411.13607
HD Kalman: from arthedain-1 VSA/HDC repo (https://github.com/Enotrium/arthedain-1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from .vsa_hdc import HDCKalmanFilter, VSAHDC


class ConvBlock(nn.Module):
    """Conv-BN-SiLU block for detection head."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    """
    YOLO-style detection head adapted for UAV multi-object detection.
    Outputs bounding boxes and class probabilities at multiple scales.

    Modified from VioPose: Instead of 3D joint positions, predicts:
      - xyxy bounding boxes
      - Objectness score
      - Class probabilities
      - (Optional) 3D position

    Args:
        in_channels: Input feature channels from hierarchy/mixing
        num_classes: Number of object classes
        num_anchors: Anchors per scale (default: 3)
        use_3d: Whether to predict 3D position
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        num_anchors: int = 3,
        use_3d: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.use_3d = use_3d

        # Output channels per anchor: 4 (bbox) + 1 (obj) + num_classes + 3 (3D) + 3 (velocity)
        self.bbox_dim = 4  # x, y, w, h
        self.obj_dim = 1   # objectness
        self.cls_dim = num_classes
        self.d3_dim = 3 if use_3d else 0  # x_world, y_world, z_world
        self.vel_dim = 3  # vx, vy, vz

        out_per_anchor = self.bbox_dim + self.obj_dim + self.cls_dim + self.d3_dim + self.vel_dim
        self.out_channels = out_per_anchor * num_anchors

        # Detection head (two conv layers + final prediction)
        self.stem = nn.Sequential(
            ConvBlock(in_channels, in_channels * 2, 3),
            ConvBlock(in_channels * 2, in_channels, 3),
        )

        self.pred = nn.Conv2d(in_channels, self.out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize prediction bias for stable training
        if hasattr(self.pred, 'bias') and self.pred.bias is not None:
            self.pred.bias.data.zero_()

    def forward(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: [B, C, H, W] fused features from hierarchy/mixing

        Returns:
            dict with:
                - "raw": Raw predictions [B, out_channels, H, W]
                - "bbox": Bounding box predictions [B, num_anchors*H*W, 4]
                - "obj": Objectness [B, num_anchors*H*W, 1]
                - "cls": Class logits [B, num_anchors*H*W, num_classes]
                - "d3": 3D position [B, num_anchors*H*W, 3] (if use_3d)
                - "vel": Velocity [B, num_anchors*H*W, 3]
        """
        B, C, H, W = features.shape

        x = self.stem(features)
        pred = self.pred(x)  # [B, out_channels, H, W]

        # Reshape to [B, num_anchors, out_per_anchor, H, W]
        pred = pred.view(B, self.num_anchors, -1, H, W)

        # Split predictions
        idx = 0
        bbox = pred[:, :, idx : idx + self.bbox_dim]
        idx += self.bbox_dim
        obj = pred[:, :, idx : idx + self.obj_dim]
        idx += self.obj_dim
        cls_logits = pred[:, :, idx : idx + self.cls_dim]
        idx += self.cls_dim

        if self.use_3d:
            d3 = pred[:, :, idx : idx + self.d3_dim]
            idx += self.d3_dim
        else:
            d3 = None

        vel = pred[:, :, idx : idx + self.vel_dim]

        # Reshape to flattened spatial dims
        def flatten_batch(t):
            return t.permute(0, 1, 3, 4, 2).reshape(B, -1, t.shape[2])

        bbox = flatten_batch(bbox)
        obj = flatten_batch(obj)
        cls_logits = flatten_batch(cls_logits)
        vel = flatten_batch(vel)
        if d3 is not None:
            d3 = flatten_batch(d3)

        return {
            "raw": pred,
            "bbox": bbox,
            "obj": obj,
            "cls": cls_logits,
            "d3": d3,
            "vel": vel,
        }


class TrackingHead(nn.Module):
    """
    4D Tracking head with hyperdimensional Kalman filter.
    Maintains tracklets over time using VSA/HDC representations
    for robust association and state estimation.

    Adapted from VioPose: The temporal pose estimation becomes
    multi-object trajectory tracking in HD space.

    The HD Kalman filter replaces standard matrix-inversion Kalman
    with hyperdimensional operations that are:
      - More robust to noise
      - Hardware-efficient (bitwise on FPGA)
      - Naturally handle missing data

    Args:
        state_dim: State vector dimension (8: x,y,z,w,h,dx,dy,dz or 8: x,y,z,dx,dy,dz,vx,vy)
        hd_dim: Hyperdimensional dimension
        feature_dim: Feature dimension for ReID
        max_age: Maximum frames to keep lost tracks
        min_hits: Minimum detections before track is confirmed
        iou_threshold: IoU threshold for association
    """

    def __init__(
        self,
        state_dim: int = 8,
        hd_dim: int = 8192,
        feature_dim: int = 512,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hd_dim = hd_dim
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        # HD Kalman filter for state estimation
        self.hd_kalman = HDCKalmanFilter(
            state_dim=state_dim,
            hd_dim=hd_dim,
        )

        # ReID feature extractor
        self.reid_proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # VSA for tracklet matching
        self.vsa_matcher = VSAHDC(
            hd_dim=hd_dim,
            input_dim=128,
            dtype="bipolar",
            binding="circular",
        )

        # Track management
        self.register_buffer("next_id", torch.tensor(0, dtype=torch.long))

    def extract_appearance(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract appearance embedding for ReID.

        Args:
            features: [N, feature_dim] per-detection features

        Returns:
            [N, 128] appearance embeddings
        """
        return self.reid_proj(features)

    def associate(
        self,
        detections: torch.Tensor,
        detection_features: torch.Tensor,
        track_states: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Associate detections to existing tracks using HD similarity.

        Args:
            detections: [N_det, state_dim] current detections
            detection_features: [N_det, 128] appearance features
            track_states: List of track dicts with "state", "hd_state", "features"

        Returns:
            (matches, unmatched_dets, unmatched_tracks) indices
        """
        N_det = detections.shape[0]
        N_trk = len(track_states)

        if N_trk == 0:
            return (
                torch.zeros(0, 2, dtype=torch.long, device=detections.device),
                torch.arange(N_det, device=detections.device),
                torch.zeros(0, dtype=torch.long, device=detections.device),
            )

        if N_det == 0:
            return (
                torch.zeros(0, 2, dtype=torch.long, device=detections.device),
                torch.zeros(0, dtype=torch.long, device=detections.device),
                torch.arange(N_trk, device=detections.device),
            )

        # Encode detections to HD
        det_hd = self.vsa_matcher.encode(detection_features)  # [N_det, hd_dim]

        # Get track HD states
        trk_hd = torch.stack([t["hd_state"] for t in track_states], dim=0)  # [N_trk, hd_dim]

        # HD similarity matrix
        sim_matrix = self.vsa_matcher.similarity(
            det_hd.unsqueeze(1),  # [N_det, 1, hd_dim]
            trk_hd.unsqueeze(0),  # [1, N_trk, hd_dim]
        )  # [N_det, N_trk]

        # Hungarian algorithm for optimal assignment
        # (simplified greedy for now; replace with scipy.optimize.linear_sum_assignment)
        matched_indices = []
        used_dets = set()
        used_trks = set()

        # Sort by similarity
        flat_idx = torch.argsort(sim_matrix.flatten(), descending=True)
        det_idx = flat_idx // N_trk
        trk_idx = flat_idx % N_trk

        for d, t in zip(det_idx.tolist(), trk_idx.tolist()):
            if d not in used_dets and t not in used_trks:
                if sim_matrix[d, t] > 0.3:  # Similarity threshold
                    matched_indices.append([d, t])
                    used_dets.add(d)
                    used_trks.add(t)

        matched = torch.tensor(matched_indices, dtype=torch.long, device=detections.device)
        unmatched_dets = torch.tensor(
            [d for d in range(N_det) if d not in used_dets],
            dtype=torch.long, device=detections.device,
        )
        unmatched_trks = torch.tensor(
            [t for t in range(N_trk) if t not in used_trks],
            dtype=torch.long, device=detections.device,
        )

        return matched, unmatched_dets, unmatched_trks

    def update_tracks(
        self,
        matches: torch.Tensor,
        unmatched_dets: torch.Tensor,
        unmatched_trks: torch.Tensor,
        detections: torch.Tensor,
        detection_features: torch.Tensor,
        tracks: List[Dict],
        frame_id: int,
    ) -> List[Dict]:
        """
        Update track states based on associations.

        Args:
            matches: [M, 2] matched (det_idx, trk_idx)
            unmatched_dets: Unmatched detection indices
            unmatched_trks: Unmatched track indices
            detections: [N_det, state_dim]
            detection_features: [N_det, 128]
            tracks: Existing track list
            frame_id: Current frame ID

        Returns:
            Updated track list
        """
        device = detections.device

        # Update matched tracks
        for d_idx, t_idx in matches:
            d_idx, t_idx = d_idx.item(), t_idx.item()
            track = tracks[t_idx]

            # HD Kalman update
            det_hd = self.vsa_matcher.encode(detection_features[d_idx : d_idx + 1])
            posterior_hd, _ = self.hd_kalman(track["hd_state"].unsqueeze(0), det_hd)
            track["hd_state"] = posterior_hd.squeeze(0)

            # Update state
            track["state"] = (
                0.7 * track["state"] + 0.3 * detections[d_idx]
            )
            track["features"] = (
                0.8 * track["features"] + 0.2 * detection_features[d_idx]
            )
            track["hits"] += 1
            track["age"] = 0
            track["last_frame"] = frame_id

        # Add new tracks for unmatched detections
        for d_idx in unmatched_dets:
            d_idx = d_idx.item()
            det_hd = self.vsa_matcher.encode(detection_features[d_idx : d_idx + 1])
            _, pred_hd = self.hd_kalman(det_hd)

            tracks.append({
                "id": self.next_id.item(),
                "state": detections[d_idx].clone(),
                "hd_state": pred_hd.squeeze(0),
                "features": detection_features[d_idx].clone(),
                "hits": 1,
                "age": 0,
                "last_frame": frame_id,
                "velocity": detections[d_idx, 4:7].clone() if detections.shape[1] >= 7 else torch.zeros(3, device=device),
                "trajectory": [detections[d_idx, :3].clone()],
            })
            self.next_id += 1

        # Update age for unmatched tracks
        for t_idx in unmatched_trks:
            t_idx = t_idx.item()
            track = tracks[t_idx]
            track["age"] += 1
            # Predict forward
            track["hd_state"], _ = self.hd_kalman(track["hd_state"].unsqueeze(0))
            track["hd_state"] = track["hd_state"].squeeze(0)

        # Remove expired tracks
        tracks = [t for t in tracks if t["age"] <= self.max_age]

        return tracks

    def forward(
        self,
        detections: torch.Tensor,
        detection_features: torch.Tensor,
        features_for_reid: torch.Tensor,
        tracks: Optional[List[Dict]] = None,
        frame_id: int = 0,
    ) -> Dict:
        """
        Full tracking forward pass.

        Args:
            detections: [N_det, 8] state vectors (x,y,z,w,h,vx,vy,vz or x,y,z,dx,dy,dz,vx,vy)
            detection_features: [N_det, D] per-detection features
            features_for_reid: [N_det, D_reid] features for ReID
            tracks: Optional existing track list
            frame_id: Frame counter

        Returns:
            dict with "tracks" and "active_tracks"
        """
        if tracks is None:
            tracks = []

        # Extract appearance features
        app_features = self.extract_appearance(features_for_reid)

        # Associate
        # Build detection state
        if detections.shape[1] < self.state_dim:
            padding = torch.zeros(
                detections.shape[0],
                self.state_dim - detections.shape[1],
                device=detections.device,
            )
            det_state = torch.cat([detections, padding], dim=1)
        else:
            det_state = detections[:, :self.state_dim]

        matched, unmatched_dets, unmatched_trks = self.associate(
            det_state, app_features, tracks
        )

        # Update tracks
        tracks = self.update_tracks(
            matched, unmatched_dets, unmatched_trks,
            det_state, app_features, tracks, frame_id,
        )

        # Filter active (confirmed) tracks
        active_tracks = [t for t in tracks if t["hits"] >= self.min_hits and t["age"] == 0]

        active_states = torch.stack(
            [t["state"] for t in active_tracks]
        ) if active_tracks else torch.zeros(0, self.state_dim, device=detections.device)

        active_ids = torch.tensor(
            [t["id"] for t in active_tracks],
            dtype=torch.long, device=detections.device,
        ) if active_tracks else torch.zeros(0, dtype=torch.long, device=detections.device)

        return {
            "tracks": tracks,
            "active_states": active_states,
            "active_ids": active_ids,
            "matches": matched,
            "new_tracks": len(unmatched_dets),
        }

    def reset(self):
        """Reset tracking state."""
        self.next_id.zero_()