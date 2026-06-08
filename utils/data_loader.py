"""
Eldarin Data Loader
====================
Multi-modal data loading for Eldarin training and inference.
Supports RGB frames, event streams, audio, IMU, and pose data from:
  - VisDrone (https://github.com/VisDrone/VisDrone-Dataset)
  - UAVDT, UAV3D, FRED datasets
  - Custom event-based streams

Original VioPose: https://github.com/SeongJong-Yoo/VioPose
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json


class EldarinDataLoader(Dataset):
    """
    Multi-modal dataset loader for Eldarin.

    Handles:
      - RGB frames with detection/tracking annotations
      - Event camera streams (FRED format or numpy)
      - Audio waveforms (optional)
      - IMU/GPS data (optional)

    Annotation format: COCO-style JSON with tracking IDs.

    Args:
        data_root: Root directory of dataset
        split: "train", "val", or "test"
        modalities: List of modality strings ["rgb", "event", "audio", "imu"]
        img_size: Target image size (H, W)
        augment: Whether to apply augmentations
        max_samples: Maximum samples to load (for debugging)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        modalities: List[str] = None,
        img_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        max_samples: Optional[int] = None,
        dataset_format: str = "visdrone",
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.modalities = modalities or ["rgb"]
        self.img_size = tuple(img_size)
        self.augment = augment and split == "train"
        self.dataset_format = dataset_format

        # Load annotation file
        self.annotations = self._load_annotations()
        if max_samples:
            self.annotations = self.annotations[:max_samples]

        self.image_files = self._find_image_files()

    def _load_annotations(self) -> List[Dict]:
        """Load COCO-style annotations."""
        ann_file = self.data_root / f"annotations/{self.split}.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                coco = json.load(f)
            # Flatten to per-image entries
            images = {img['id']: img for img in coco['images']}
            annotations = {}
            for ann in coco['annotations']:
                img_id = ann['image_id']
                if img_id not in annotations:
                    annotations[img_id] = []
                annotations[img_id].append(ann)
            # Build sample list
            samples = []
            for img_id, img_info in images.items():
                samples.append({
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'annotations': annotations.get(img_id, []),
                })
            return samples
        else:
            # Fallback: directory listing
            print(f"Warning: No annotation file found at {ann_file}")
            return []

    def _find_image_files(self) -> List[Path]:
        """Find all image files in the dataset directory."""
        img_dir = self.data_root / "images" / self.split
        if img_dir.exists():
            return sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        return []

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess an RGB image."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size[::-1])
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def _load_event_data(self, image_id: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """Load event data for a frame."""
        event_file = self.data_root / "events" / self.split / f"{image_id:06d}.npy"
        if event_file.exists():
            events = np.load(event_file)  # [N, 4]: x, y, t, p
            return (
                torch.from_numpy(events[:, 0]).float(),
                torch.from_numpy(events[:, 1]).float(),
                torch.from_numpy(events[:, 2]).float(),
                torch.from_numpy(events[:, 3]).float(),
            )
        return None

    def _load_audio(self, image_id: int) -> Optional[torch.Tensor]:
        """Load audio waveform for a frame."""
        audio_file = self.data_root / "audio" / self.split / f"{image_id:06d}.wav"
        if audio_file.exists():
            import torchaudio
            waveform, sr = torchaudio.load(str(audio_file))
            return waveform
        return None

    def _parse_annotations(self, ann_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Parse annotations into tensor format.

        Returns dict with:
          - bboxes: [N, 4] (x, y, w, h) normalized
          - classes: [N] class indices
          - track_ids: [N] tracking IDs
          - d3: [N, 3] 3D positions (optional)
        """
        if not ann_list:
            return {
                'bboxes': torch.zeros(0, 4),
                'classes': torch.zeros(0, dtype=torch.long),
                'track_ids': torch.zeros(0, dtype=torch.long),
            }

        bboxes = []
        classes = []
        track_ids = []
        d3 = []

        for ann in ann_list:
            # COCO bbox: [x, y, w, h]
            bbox = ann['bbox']
            bboxes.append(bbox)
            classes.append(ann['category_id'])
            track_ids.append(ann.get('track_id', -1))
            if 'd3' in ann:
                d3.append(ann['d3'])

        result = {
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'classes': torch.tensor(classes, dtype=torch.long),
            'track_ids': torch.tensor(track_ids, dtype=torch.long),
        }

        if d3:
            result['d3'] = torch.tensor(d3, dtype=torch.float32)

        return result

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.annotations[idx]
        img_path = self.data_root / "images" / self.split / sample['file_name']

        data = {}

        # RGB frame
        if "rgb" in self.modalities or "visual" in self.modalities:
            data['frames'] = self._load_image(img_path)

        # Event data
        if "event" in self.modalities:
            events = self._load_event_data(sample['image_id'])
            if events is not None:
                data['events'] = events
                data['event_duration_us'] = 10000.0

        # Audio
        if "audio" in self.modalities:
            audio = self._load_audio(sample['image_id'])
            if audio is not None:
                data['audio'] = audio

        # Annotations
        targets = self._parse_annotations(sample['annotations'])
        data['targets'] = targets

        # Image metadata
        data['image_id'] = sample['image_id']
        data['orig_size'] = torch.tensor([sample['height'], sample['width']])

        return data


def create_dataloader(
    config: dict,
    split: str = "train",
) -> DataLoader:
    """
    Create a DataLoader from config.

    Args:
        config: Configuration dictionary
        split: Dataset split

    Returns:
        PyTorch DataLoader
    """
    data_cfg = config.get('data', {})

    dataset = EldarinDataLoader(
        data_root=data_cfg.get('data_root', '/path/to/dataset'),
        split=split,
        modalities=data_cfg.get('modalities', ['rgb']),
        img_size=tuple(data_cfg.get('img_size', [640, 640])),
        augment=split == 'train',
        max_samples=data_cfg.get('max_samples'),
        dataset_format=data_cfg.get('dataset', 'visdrone'),
    )

    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get('batch_size', 8),
        shuffle=split == 'train',
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_memory', True),
        collate_fn=collate_fn,
        drop_last=split == 'train',
    )

    return loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for multi-modal data."""
    collated = {}

    # Stack frames
    if 'frames' in batch[0]:
        collated['frames'] = torch.stack([b['frames'] for b in batch])

    # Collect events (keep as list of tuples)
    if 'events' in batch[0]:
        collated['events'] = [b.get('events') for b in batch]
        collated['event_duration_us'] = [b.get('event_duration_us') for b in batch]

    # Stack audio
    if 'audio' in batch[0]:
        max_len = max(b['audio'].shape[-1] for b in batch if 'audio' in b)
        audio_batch = []
        for b in batch:
            if 'audio' in b:
                audio = b['audio']
                if audio.shape[-1] < max_len:
                    audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1]))
                audio_batch.append(audio)
        if audio_batch:
            collated['audio'] = torch.stack(audio_batch)

    # Targets
    collated['targets'] = [b['targets'] for b in batch]
    collated['image_id'] = [b['image_id'] for b in batch]

    return collated