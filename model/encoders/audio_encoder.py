"""
Audio Encoder for UAV environmental sounds
============================================
Adapted from VioPose audio modality encoder (originally for violin audio).
Now processes propeller noise, environmental sounds, and other acoustic cues
for cross-modal object type and motion priors.

Uses CNN + attention over mel-spectrograms with optional pretrained models.

Original VioPose: https://github.com/SeongJong-Yoo/VioPose
Paper: https://arxiv.org/pdf/2411.13607

Cross-modal causal cues: Audio can inform object type (e.g., vehicle engine,
drone propeller) and motion state (approaching/receding via Doppler shifts).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class MelSpectrogram(nn.Module):
    """Mel spectrogram extraction for audio input."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Mel filterbank
        f_max = f_max or sample_rate / 2
        mel_fb = self._create_mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)
        self.register_buffer("mel_filterbank", mel_fb)

    @staticmethod
    def _create_mel_filterbank(
        n_mels: int, n_fft: int, sample_rate: int, f_min: float, f_max: float
    ) -> torch.Tensor:
        """Create mel filterbank matrix."""
        mel_min = 2595 * math.log10(1 + f_min / 700)
        mel_max = 2595 * math.log10(1 + f_max / 700)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
        fb = torch.zeros(n_mels, n_fft // 2 + 1)

        for m in range(n_mels):
            f_m = bin_points[m]
            f_mp1 = bin_points[m + 1]
            f_mp2 = bin_points[m + 2]

            if f_mp1 > f_m:
                fb[m, f_m:f_mp1] = torch.linspace(0, 1, f_mp1 - f_m)
            if f_mp2 > f_mp1:
                fb[m, f_mp1:f_mp2] = torch.linspace(1, 0, f_mp2 - f_mp1)

        return fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram.

        Args:
            waveform: [B, T] audio waveform

        Returns:
            Mel spectrogram [B, n_mels, time_frames]
        """
        # STFT
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

        # Power spectrogram
        power = stft.abs() ** 2  # [B, freq_bins, time]

        # Apply mel filterbank
        mel = torch.einsum("mf,bft->bmt", self.mel_filterbank, power)

        # Log compression
        mel = torch.log(mel + 1e-6)

        return mel


class AudioCNNBackbone(nn.Module):
    """2D CNN backbone for mel spectrogram features."""

    def __init__(
        self, in_channels: int = 1, channels: list = None
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        self.convs = nn.ModuleList()
        c_prev = in_channels
        for c in channels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(c_prev, c, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                )
            )
            c_prev = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        return x


class AudioEncoder(nn.Module):
    """
    Audio encoder for UAV environmental sounds.
    Processes audio waveform → mel spectrogram → CNN → global + multi-scale features.

    Captures:
      - Propeller/engine signatures for object type classification
      - Doppler shifts for motion direction/velocity estimation
      - Environmental context (urban, rural, etc.)

    Args:
        sample_rate: Audio sample rate
        n_mels: Mel frequency bins
        out_dim: Output feature dimension
        temporal_pool: Pooling over time dimension
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        out_dim: int = 512,
        temporal_pool: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.out_dim = out_dim

        # Mel spectrogram
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512,
        )

        # CNN backbone
        self.backbone = AudioCNNBackbone(in_channels=1)

        # Temporal attention (cross-modal cue extraction)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=4, batch_first=True
        )

        # Global descriptor
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

        # Motion feature extractor (Doppler analysis)
        self.motion_proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # Approaching/receding/stationary logits
        )

        self.temporal_pool = temporal_pool

    def forward(
        self, waveform: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            waveform: [B, T] audio waveform

        Returns:
            dict with:
                - "global": Global audio descriptor [B, out_dim]
                - "motion": Motion logits [B, 3]
                - "features": Raw CNN features
        """
        B = waveform.shape[0]

        # Mel spectrogram [B, n_mels, time]
        mel = self.mel(waveform)

        # Add channel dim [B, 1, n_mels, time]
        mel = mel.unsqueeze(1)

        # CNN backbone [B, 512, H', W']
        features = self.backbone(mel)

        # Temporal attention
        B, C, H, W = features.shape
        features_flat = features.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        attended, _ = self.temporal_attn(features_flat, features_flat, features_flat)
        features = attended.transpose(1, 2).view(B, C, H, W)

        # Global descriptor
        global_feat = self.global_pool(features).view(B, -1)
        global_feat = self.projection(global_feat)

        # Motion prediction (causal cross-modal cue)
        motion_feat = features.mean(dim=[-2, -1])
        motion_logits = self.motion_proj(motion_feat)

        return {
            "global": global_feat,
            "motion": motion_logits,
            "features": features if not self.temporal_pool else None,
        }


class LightweightAudioEncoder(AudioEncoder):
    """Lightweight variant for UAV onboard deployment."""

    def __init__(self, out_dim: int = 256):
        super().__init__(
            sample_rate=16000,
            n_mels=64,
            out_dim=out_dim,
            temporal_pool=True,
        )
        # Smaller backbone
        self.backbone = AudioCNNBackbone(in_channels=1, channels=[16, 32, 64, 128])
        self.projection = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )