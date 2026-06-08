"""
VisDrone Dataset Loader
========================
Industry-standard UAV benchmark. 288 video clips, 10k+ images,
>2.6M bounding boxes. 10 object classes.

Supports: Detection, single/multi-object tracking, crowd counting.
Source: https://github.com/VisDrone/VisDrone-Dataset

Adapted for Eldarin hierarchical multimodal 4D tracking.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

VISDRONE_CLASSES = [
    "ignored", "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


class VisDroneDataset(Dataset):
    """
    VisDrone dataset for object detection and tracking.

    Annotation format per frame:
      <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

    For tracking (MOT format):
      <frame>,<id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

    Args:
        data_root: Root VisDrone directory
        split: "train", "val", "test-dev", "test-challenge"
        task: "det" (detection) or "mot" (multi-object tracking)
        img_size: Target image resolution [H, W]
        augment: Enable data augmentation
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        task: str = "det",
        img_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.task = task
        self.img_size = img_size
        self.augment = augment and split == "train"

        # Dataset paths
        task_dir = "VisDrone2019-DET" if task == "det" else "VisDrone2019-MOT"
        split_name = {
            "train": "train",
            "val": "val",
            "test-dev": "test-dev",
            "test-challenge": "test-challenge",
        }.get(split, split)

        self.img_dir = self.data_root / task_dir / f"{task}-{split_name}" / "images"
        self.ann_dir = self.data_root / task_dir / f"{task}-{split_name}" / "annotations"

        # Find all image files
        self.image_files = sorted(list(self.img_dir.glob("*.jpg")))

        # Load annotations
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> Dict[str, np.ndarray]:
        """Load VisDrone annotations."""
        annotations = {}

        for img_path in self.image_files:
            ann_file = self.ann_dir / f"{img_path.stem}.txt"
            if ann_file.exists():
                anns = np.loadtxt(ann_file, delimiter=",", dtype=np.float32)
                if anns.ndim == 1:
                    anns = anns.reshape(1, -1)
                # Filter valid annotations (score > 0 and not ignored class)
                if self.task == "det":
                    valid = (anns[:, 4] > 0) & (anns[:, 5] > 0)  # score > 0, not ignored
                    anns = anns[valid]
            else:
                anns = np.zeros((0, 8), dtype=np.float32)
            annotations[img_path.stem] = anns

        return annotations

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        anns = self.annotations[img_path.stem]

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load: {img_path}")
        orig_h, orig_w = img.shape[:2]

        # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size[::-1])
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Parse annotations
        targets = self._parse_visdrone_anns(anns, orig_w, orig_h)

        return {
            "frames": img_tensor,
            "targets": targets,
            "image_id": img_path.stem,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

    def _parse_visdrone_anns(
        self, anns: np.ndarray, img_w: int, img_h: int
    ) -> Dict[str, torch.Tensor]:
        """Parse VisDrone annotations to Eldarin format."""
        if len(anns) == 0:
            return {
                "bboxes": torch.zeros(0, 4),
                "classes": torch.zeros(0, dtype=torch.long),
                "track_ids": torch.zeros(0, dtype=torch.long),
            }

        # VisDrone columns: bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
        x1 = anns[:, 0] / img_w
        y1 = anns[:, 1] / img_h
        w = anns[:, 2] / img_w
        h = anns[:, 3] / img_h
        classes = anns[:, 5].astype(np.int64) - 1  # Subtract 1 (class 0 is ignored)

        # Filter valid classes
        valid = classes >= 0
        x1, y1, w, h = x1[valid], y1[valid], w[valid], h[valid]
        classes = classes[valid]

        bboxes = np.stack([x1, y1, x1 + w, y1 + h], axis=1)

        # Track IDs (column 1 in MOT format, not available in DET)
        track_ids = np.zeros(len(bboxes), dtype=np.int64)
        if self.task == "mot" and anns.shape[1] >= 9:
            track_ids = anns[valid][:, 1].astype(np.int64)

        return {
            "bboxes": torch.from_numpy(bboxes).float(),
            "classes": torch.from_numpy(classes).long(),
            "track_ids": torch.from_numpy(track_ids).long(),
        }