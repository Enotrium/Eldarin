#!/usr/bin/env python3
"""
VisDrone Dataset Preprocessing
================================
Converts VisDrone annotations to COCO JSON format for Eldarin training.
Also generates basic statistics and data splits.

VisDrone: https://github.com/VisDrone/VisDrone-Dataset
Eldarin adaptation of 
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

VISDRONE_CLASSES = [
    "ignored", "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def parse_visdrone_annotation(ann_path: str) -> list:
    """Parse a single VisDrone annotation file."""
    annotations = []
    if not os.path.exists(ann_path):
        return annotations

    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            bbox_left, bbox_top, bbox_width, bbox_height = map(float, parts[:4])
            score = float(parts[4])
            category = int(parts[5])
            truncation = int(parts[6]) if len(parts) > 6 else 0
            occlusion = int(parts[7]) if len(parts) > 7 else 0

            # Skip ignored and low-confidence
            if category == 0 or score <= 0:
                continue

            annotations.append({
                "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                "category_id": category - 1,  # 0-indexed (shift from VisDrone 1-indexed)
                "score": score,
                "truncation": truncation,
                "occlusion": occlusion,
            })
    return annotations


def convert_to_coco(data_dir: str, split: str, output_path: str):
    """Convert VisDrone annotations to COCO format."""
    img_dir = Path(data_dir) / f"VisDrone2019-DET-{split}" / "images"
    ann_dir = Path(data_dir) / f"VisDrone2019-DET-{split}" / "annotations"

    if not img_dir.exists():
        print(f"Warning: Image directory not found: {img_dir}")
        return

    images = []
    annotations = []
    ann_id = 0

    img_files = sorted(list(img_dir.glob("*.jpg")))
    print(f"Processing {len(img_files)} images for {split}...")

    for img_id, img_path in enumerate(img_files):
        # Read image dimensions
        try:
            import cv2
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
        except:
            w, h = 1920, 1080  # Default

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        # Parse annotations
        ann_file = ann_dir / f"{img_path.stem}.txt"
        anns = parse_visdrone_annotation(str(ann_file))

        for ann in anns:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": 0,
                "score": ann.get("score", 1.0),
                "truncation": ann.get("truncation", 0),
                "occlusion": ann.get("occlusion", 0),
            })
            ann_id += 1

    # Build COCO dict
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": name} for i, name in enumerate(VISDRONE_CLASSES[1:])
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    # Statistics
    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    if annotations:
        cats = defaultdict(int)
        for ann in annotations:
            cats[VISDRONE_CLASSES[ann["category_id"] + 1]] += 1
        print(f"  Class distribution: {dict(cats)}")

    output_size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved to {output_path} ({output_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess VisDrone for Eldarin")
    parser.add_argument("--data_root", type=str, default="data/VisDrone2019-DET",
                        help="Root directory of VisDrone dataset")
    parser.add_argument("--output_dir", type=str, default="data/VisDrone2019-DET/annotations",
                        help="Output directory for COCO JSON files")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        print(f"\n--- Processing {split} ---")
        output_path = Path(args.output_dir) / f"{split}.json"
        convert_to_coco(args.data_root, split, str(output_path))

    print("\nPreprocessing complete!")
    print(f"Run training with: python main.py --config config/train_visdrone.yaml --data_root {args.data_root}")


if __name__ == "__main__":
    main()