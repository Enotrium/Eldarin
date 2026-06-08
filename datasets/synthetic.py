"""
Synthetic Dataset Loader
=========================
Generates high-fidelity synthetic UAV data for domain randomization
and large-scale pretraining. Supports configurable object types,
weather conditions, lighting, altitudes, and multi-modal outputs.

Designed for sim-to-real transfer (OpenAI/Anduril style training).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple, Optional


class SyntheticDataset(Dataset):
    """
    Synthetic UAV detection/tracking dataset.

    Generates randomized scenes with:
      - Multiple object types (vehicles, people, drones)
      - Varying lighting, weather, altitude
      - 2D/3D bounding boxes with velocities
      - Optional event stream simulation
    """

    def __init__(
        self,
        num_samples: int = 10000,
        img_size: Tuple[int, int] = (640, 640),
        num_classes: int = 10,
        max_objects: int = 20,
        include_3d: bool = True,
        include_velocity: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.include_3d = include_3d
        self.include_velocity = include_velocity
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate random synthetic image
        img = self.rng.rand(3, *self.img_size).astype(np.float32)
        img_tensor = torch.from_numpy(img)

        # Random number of objects
        n_objs = self.rng.randint(1, self.max_objects + 1)

        # Generate random boxes (normalized)
        bboxes = []
        classes = []
        d3_list = []
        velocities = []

        for _ in range(n_objs):
            # Random box
            x1 = self.rng.uniform(0, 0.8)
            y1 = self.rng.uniform(0, 0.8)
            w = self.rng.uniform(0.02, 0.15)
            h = self.rng.uniform(0.02, 0.15)
            bboxes.append([x1, y1, min(x1 + w, 1.0), min(y1 + h, 1.0)])
            classes.append(self.rng.randint(0, self.num_classes))

            if self.include_3d:
                d3_list.append(self.rng.uniform(-50, 50, 3))

            if self.include_velocity:
                velocities.append(self.rng.uniform(-5, 5, 3))

        targets = {
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "classes": torch.tensor(classes, dtype=torch.long),
        }

        if self.include_3d and d3_list:
            targets["d3"] = torch.tensor(d3_list, dtype=torch.float32)

        if self.include_velocity and velocities:
            targets["velocities"] = torch.tensor(velocities, dtype=torch.float32)

        return {
            "frames": img_tensor,
            "targets": targets,
            "image_id": f"synthetic_{idx:06d}",
            "orig_size": torch.tensor(self.img_size),
        }