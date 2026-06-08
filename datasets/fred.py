"""
FRED (Florence RGB-Event Drone) Dataset Loader
=================================================
RGB + event camera streams for drone detection, tracking, and forecasting.
Challenging conditions: rain, low light, fast motion.

Source: https://github.com/francesco-p/FRED

Adapted for Eldarin's event-based encoding pipeline (FPGA-Event-Based-encode).

Reference FPGA event encode: https://github.com/Enotrium/FPGA-Event-Based-encode
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional


class FREDDataset(Dataset):
    """
    FRED dataset: RGB frames + event streams for drone detection.

    Event format: (x, y, timestamp_us, polarity)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (640, 640),
        event_window_us: float = 10000,
        augment: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.event_window_us = event_window_us
        self.augment = augment and split == "train"

        self.rgb_dir = self.data_root / split / "rgb"
        self.event_dir = self.data_root / split / "events"
        self.ann_dir = self.data_root / split / "annotations"

        self.rgb_files = sorted(list(self.rgb_dir.glob("*.png")) + list(self.rgb_dir.glob("*.jpg")))
        self.event_files = sorted(list(self.event_dir.glob("*.npy")))

        # Map RGB frames to event windows
        self.samples = self._build_sample_index()

    def _build_sample_index(self) -> list:
        """Match RGB frames with corresponding event data windows."""
        samples = []
        for rgb_path in self.rgb_files:
            event_path = self.event_dir / f"{rgb_path.stem}.npy"
            ann_path = self.ann_dir / f"{rgb_path.stem}.txt"
            samples.append({
                "rgb": rgb_path,
                "event": event_path if event_path.exists() else None,
                "ann": ann_path if ann_path.exists() else None,
            })
        return samples

    def _load_events(self, event_path: Path) -> Tuple[torch.Tensor, ...]:
        """Load event data as (x, y, t, p) tuple."""
        events = np.load(str(event_path))
        # events: [N, 4] -> x, y, t_us, polarity
        return (
            torch.from_numpy(events[:, 0].astype(np.float32)),
            torch.from_numpy(events[:, 1].astype(np.float32)),
            torch.from_numpy(events[:, 2].astype(np.float32)),
            torch.from_numpy(events[:, 3].astype(np.float32)),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # RGB frame
        img = cv2.cvtColor(cv2.imread(str(sample["rgb"])), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, self.img_size[::-1])
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        data = {
            "frames": img_tensor,
            "image_id": sample["rgb"].stem,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

        # Event data
        if sample["event"] is not None:
            data["events"] = self._load_events(sample["event"])
            data["event_duration_us"] = self.event_window_us

        # Annotations
        if sample["ann"] is not None:
            anns = np.loadtxt(sample["ann"], delimiter=",", dtype=np.float32)
            if anns.ndim == 1:
                anns = anns.reshape(1, -1)
            # Simple bbox format: class, x, y, w, h
            if anns.shape[1] >= 5:
                classes = anns[:, 0].astype(np.int64)
                boxes = anns[:, 1:5]
                boxes[:, [0, 2]] /= orig_w
                boxes[:, [1, 3]] /= orig_h
                data["targets"] = {
                    "bboxes": torch.from_numpy(boxes).float(),
                    "classes": torch.from_numpy(classes).long(),
                }
            else:
                data["targets"] = {
                    "bboxes": torch.zeros(0, 4),
                    "classes": torch.zeros(0, dtype=torch.long),
                }
        else:
            data["targets"] = {
                "bboxes": torch.zeros(0, 4),
                "classes": torch.zeros(0, dtype=torch.long),
            }

        return data