"""
IMU / Auxiliary Sensor Encoder
================================
 auxiliary feature paths.
Processes IMU (accelerometer, gyroscope, magnetometer) + optional GPS/pose data
for motion-aware and position-aware feature extraction.

Provides:
  - Ego-motion estimation (UAV self-motion compensation)
  - Camera pose embedding
  - Temporal dynamics from IMU sequences


Paper: https://arxiv.org/pdf/2411.13607
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class IMUEncoder(nn.Module):
    """
    Encodes IMU and auxiliary sensor data for the multimodal pipeline.

    IMU channels (9-DoF): ax, ay, az, gx, gy, gz, mx, my, mz
    Optional GPS: lat, lon, alt, heading
    Optional pose: qx, qy, qz, qw, tx, ty, tz

    Architecture: 1D CNN + LSTM for temporal dynamics → global descriptor.

    Args:
        imu_dim: IMU input dimension (9 for standard 9-DoF)
        aux_dim: Auxiliary sensor dimension (GPS/pose)
        hidden_dim: Hidden dimension for LSTM
        out_dim: Output feature dimension
        num_layers: LSTM layers
    """

    def __init__(
        self,
        imu_dim: int = 9,
        aux_dim: int = 0,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.imu_dim = imu_dim
        self.aux_dim = aux_dim

        # 1D CNN for IMU feature extraction
        self.imu_cnn = nn.Sequential(
            nn.Conv1d(imu_dim, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=hidden_dim + aux_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Global projection
        lstm_out = hidden_dim * 2  # bidirectional
        self.projection = nn.Sequential(
            nn.Linear(lstm_out, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

        # Motion embedding (for cross-modal priors)
        self.motion_embed = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(
        self,
        imu_data: torch.Tensor,
        aux_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            imu_data: [B, T, imu_dim] IMU time series
            aux_data: Optional [B, T, aux_dim] auxiliary data

        Returns:
            dict with:
                - "global": Global IMU descriptor [B, out_dim]
                - "motion": Motion embedding [B, out_dim]
                - "features": Raw LSTM features
        """
        B, T, D = imu_data.shape

        # 1D CNN over IMU sequence
        imu_t = imu_data.transpose(1, 2)  # [B, D, T]
        imu_feat = self.imu_cnn(imu_t)     # [B, hidden_dim, T]
        imu_feat = imu_feat.transpose(1, 2)  # [B, T, hidden_dim]

        # Concatenate auxiliary data if available
        if aux_data is not None:
            combined = torch.cat([imu_feat, aux_data], dim=-1)
        else:
            combined = imu_feat

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(combined)  # [B, T, hidden_dim*2]

        # Use last hidden state as global descriptor
        # Bidirectional: concatenate forward and backward final states
        forward_h = h_n[-2, :, :]  # Last forward layer
        backward_h = h_n[-1, :, :]  # Last backward layer
        global_feat = torch.cat([forward_h, backward_h], dim=-1)  # [B, hidden_dim*2]

        # Project
        global_out = self.projection(global_feat)

        # Motion embedding
        motion_embed = self.motion_embed(global_feat)

        return {
            "global": global_out,
            "motion": motion_embed,
            "features": lstm_out,
        }


class LightweightIMUEncoder(IMUEncoder):
    """Lightweight IMU encoder for UAV onboard deployment."""

    def __init__(self, imu_dim: int = 9, out_dim: int = 64):
        super().__init__(
            imu_dim=imu_dim,
            aux_dim=0,
            hidden_dim=64,
            out_dim=out_dim,
            num_layers=1,
        )
        # Simplify CNN
        self.imu_cnn = nn.Sequential(
            nn.Conv1d(imu_dim, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )


class PoseEncoder(nn.Module):
    """
    Optional camera/vehicle pose encoder.
    Encodes 6-DoF pose (translation + rotation quaternion) for
    ego-motion compensation and multi-view fusion.
    """

    def __init__(
        self,
        out_dim: int = 128,
    ):
        super().__init__()
        # Input: quaternion(4) + translation(3) = 7
        self.encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: [B, 7] or [B, T, 7] pose data

        Returns:
            [B, out_dim] pose embedding
        """
        if pose.dim() == 3:
            B, T, D = pose.shape
            pose = pose.reshape(B * T, D)
            emb = self.encoder(pose)
            emb = emb.reshape(B, T, -1).mean(dim=1)  # Pool over time
        else:
            emb = self.encoder(pose)
        return emb