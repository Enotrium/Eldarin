"""
UAVDT Dataset Loader
=====================
~80,000 frames from 10 hours of UAV video, focused on vehicle detection/tracking.
Rich attributes: weather, altitude, occlusion, camera view.

Source: https://datasetninja.com/uavdt

Adapted for Eldarin hierarchical multimodal 4D tracking.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple

UAVDT_CLASSES = ["car", "truck", "bus"]


class UAVDTDataset(Dataset):
    """UAVDT dataset for vehicle detection and tracking."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"

        self.img_dir = self.data_root / split / "images"
        self.ann_dir = self.data_root / split / "annotations"
        self.image_files = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        ann_path = self.ann_dir / f"{img_path.stem}.txt"

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, self.img_size[::-1])
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        if ann_path.exists():
            anns = np.loadtxt(ann_path, delimiter=",", dtype=np.float32)
            if anns.ndim == 1:
                anns = anns.reshape(1, -1)
            # UAVDT format: frame_id, track_id, x1, y1, w, h, class, ...
            classes = (anns[:, 6].astype(np.int64) - 1).clip(0, 2)
            boxes = anns[:, 2:6].copy()
            boxes[:, [0, 2]] /= orig_w
            boxes[:, [1, 3]] /= orig_h
            track_ids = anns[:, 1].astype(np.int64)
            targets = {
                "bboxes": torch.from_numpy(boxes).float(),
                "classes": torch.from_numpy(classes).long(),
                "track_ids": torch.from_numpy(track_ids).long(),
            }
        else:
            targets = {"bboxes": torch.zeros(0, 4), "classes": torch.zeros(0, dtype=torch.long)}

        return {
            "frames": img_tensor,
            "targets": targets,
            "image_id": img_path.stem,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }