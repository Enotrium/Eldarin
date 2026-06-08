"""
Event Data Processing Utilities (FPGA-Compatible)
===================================================
Adapted from FPGA-Event-Based-encode: https://github.com/Enotrium/FPGA-Event-Based-encode

Functions for loading, processing, and converting event camera data.
Optimized for FPGA streaming and SNN input.

Event format: (x, y, timestamp, polarity) per event.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class EventProcessor:
    """
    Process event camera streams for Eldarin input.
    Supports multiple representations and FPGA-optimized formats.

    Reference: https://github.com/Enotrium/FPGA-Event-Based-encode
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        representation: str = "voxel_grid",
        num_bins: int = 10,
    ):
        self.height = height
        self.width = width
        self.representation = representation
        self.num_bins = num_bins

    def events_to_voxel_grid(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        duration_us: float,
    ) -> torch.Tensor:
        """Convert events to voxel grid [num_bins, H, W]."""
        N = x.shape[0]
        device = x.device

        if N == 0:
            return torch.zeros(self.num_bins, self.height, self.width, device=device)

        t_norm = (t - t.min()) / (duration_us + 1e-6)
        bin_idx = (t_norm * self.num_bins).long().clamp(0, self.num_bins - 1)

        voxel = torch.zeros(self.num_bins, self.height, self.width, device=device)

        for b in range(self.num_bins):
            mask = bin_idx == b
            if mask.any():
                pos = p[mask] > 0
                neg = ~pos
                for xi, yi in zip(x[mask][pos].long(), y[mask][pos].long()):
                    if 0 <= yi < self.height and 0 <= xi < self.width:
                        voxel[b, yi, xi] += 1
                for xi, yi in zip(x[mask][neg].long(), y[mask][neg].long()):
                    if 0 <= yi < self.height and 0 <= xi < self.width:
                        voxel[b, yi, xi] -= 1

        voxel = voxel / (voxel.abs().max() + 1e-6)
        return voxel

    def events_to_frame(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Convert events to 2-channel frame [2, H, W]."""
        frame = torch.zeros(2, self.height, self.width, device=x.device)
        N = x.shape[0]
        if N == 0:
            return frame

        pos = p > 0
        neg = ~pos
        for xi, yi in zip(x[pos].long(), y[pos].long()):
            if 0 <= yi < self.height and 0 <= xi < self.width:
                frame[0, yi, xi] += 1
        for xi, yi in zip(x[neg].long(), y[neg].long()):
            if 0 <= yi < self.height and 0 <= xi < self.width:
                frame[1, yi, xi] += 1

        return torch.log(1 + frame)

    def events_to_spike_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        time_window_us: float = 1000,
    ) -> torch.Tensor:
        """Convert events to spike tensor [T_steps, 2, H, W] for SNN input."""
        # Discretize time
        t_norm = (t - t.min()) / (time_window_us + 1e-6)
        steps = min(100, max(1, int(t_norm.max().item() * 100) + 1))
        t_idx = (t_norm * steps).long().clamp(0, steps - 1)

        spikes = torch.zeros(steps, 2, self.height, self.width, device=x.device)

        pos = p > 0
        for ti, xi, yi in zip(t_idx[pos], x[pos].long(), y[pos].long()):
            if 0 <= yi < self.height and 0 <= xi < self.width:
                spikes[ti, 0, yi, xi] = 1.0

        neg = ~pos
        for ti, xi, yi in zip(t_idx[neg], x[neg].long(), y[neg].long()):
            if 0 <= yi < self.height and 0 <= xi < self.width:
                spikes[ti, 1, yi, xi] = 1.0

        return spikes

    def process(
        self,
        events: Tuple[torch.Tensor, ...],
        duration_us: Optional[float] = None,
    ) -> torch.Tensor:
        """Process events into desired representation."""
        x, y, t, p = events
        if duration_us is None:
            duration_us = (t.max() - t.min()).item() + 1e-6

        if self.representation == "voxel_grid":
            return self.events_to_voxel_grid(x, y, t, p, duration_us)
        elif self.representation == "event_frame":
            return self.events_to_frame(x, y, t, p)
        elif self.representation == "spike_tensor":
            return self.events_to_spike_tensor(x, y, t, p, duration_us)
        return self.events_to_frame(x, y, t, p)

    def fpga_encode(
        self,
        events: Tuple[torch.Tensor, ...],
        quantize_bits: int = 8,
    ) -> torch.Tensor:
        """
        FPGA-optimized event encoding.
        Quantizes to fixed-point and uses sparse representation.

        Compatible with FPGA-Event-Based-encode HLS kernels.
        """
        frame = self.events_to_frame(events[0], events[1], events[2], events[3])
        scale = 2 ** (quantize_bits - 1) - 1
        quantized = (frame * scale).round().clamp(-scale, scale)
        return quantized


def load_events_from_file(
    file_path: str,
    max_events: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load events from file (npy, npz, or raw format).

    Args:
        file_path: Path to event file
        max_events: Maximum events to load

    Returns:
        (x, y, t, p) tuple
    """
    if file_path.endswith('.npy'):
        events = np.load(file_path)
    elif file_path.endswith('.npz'):
        events = np.load(file_path)['events']
    else:
        raise ValueError(f"Unsupported event format: {file_path}")

    if max_events:
        idx = np.random.choice(len(events), min(max_events, len(events)), replace=False)
        events = events[idx]

    return (
        torch.from_numpy(events[:, 0].astype(np.float32)),
        torch.from_numpy(events[:, 1].astype(np.float32)),
        torch.from_numpy(events[:, 2].astype(np.float32)),
        torch.from_numpy(events[:, 3].astype(np.float32)),
    )


def events_to_npy(
    events: Tuple[torch.Tensor, ...],
    output_path: str,
):
    """Save events to npy file."""
    x, y, t, p = events
    events_array = torch.stack([x, y, t, p], dim=1).cpu().numpy()
    np.save(output_path, events_array)