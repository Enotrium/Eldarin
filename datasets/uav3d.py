"""
UAV3D Dataset Loader
=====================
Large-scale 3D perception benchmark. RGB + 3D boxes.
~500k RGB images, 3.3M 3D boxes, nuScenes-style format.

Supports single-UAV and collaborative multi-UAV 3D object detection + tracking.
Source: https://uav3d.github.io/ | NeurIPS 2024

Adapted for Eldarin's 4D tracking (3D position + velocity/trajectory).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple
import json


class UAV3DDataset(Dataset):
    """UAV3D: RGB + 3D boxes for 3D/4D tracking."""

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
        self.ann_file = self.data_root / split / "annotations.json"

        with open(self.ann_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_files = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        self._build_index()

    def _build_index(self):
            """Build image-to-annotation index."""
            ann_by_image = {}
            for ann in self.annotations.get("annotations", []):
                img_id = ann["image_id"]
                ann_by_image.setdefault(img_id, []).append(ann)
            self.ann_by_image = ann_by_image

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        img_id = img_path.stem

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, self.img_size[::-1])
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        anns = self.ann_by_image.get(img_id, [])
        if anns:
            boxes_2d = []
            boxes_3d = []
            classes = []
            track_ids = []
            velocities = []

            for ann in anns:
                bbox = ann.get("bbox", [0, 0, 0, 0])  # [x, y, w, h]
                boxes_2d.append([
                    bbox[0] / orig_w, bbox[1] / orig_h,
                    (bbox[0] + bbox[2]) / orig_w, (bbox[1] + bbox[3]) / orig_h,
                ])
                d3 = ann.get("d3", [0, 0, 0])  # [x, y, z] in world coords
                boxes_3d.append(d3)
                classes.append(ann.get("category_id", 0))
                track_ids.append(ann.get("track_id", 0))
                velocities.append(ann.get("velocity", [0, 0, 0]))

            targets = {
                "bboxes": torch.tensor(boxes_2d, dtype=torch.float32),
                "d3": torch.tensor(boxes_3d, dtype=torch.float32),
                "classes": torch.tensor(classes, dtype=torch.long),
                "track_ids": torch.tensor(track_ids, dtype=torch.long),
                "velocities": torch.tensor(velocities, dtype=torch.float32),
            }
        else:
            targets = {
                "bboxes": torch.zeros(0, 4),
                "d3": torch.zeros(0, 3),
                "classes": torch.zeros(0, dtype=torch.long),
                "track_ids": torch.zeros(0, dtype=torch.long),
                "velocities": torch.zeros(0, 3),
            }

        return {
            "frames": img_tensor,
            "targets": targets,
            "image_id": img_id,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }