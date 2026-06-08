"""
Visualization utilities for Eldarin
=====================================
Detection bounding boxes, tracking trajectories, and debug visualizations.


"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# VisDrone class names
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# UAVDT class names
UAVDT_CLASSES = ["car", "truck", "bus"]

# Class colors (BGR for OpenCV)
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0),
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection bounding boxes on image.

    Args:
        image: BGR image [H, W, 3]
        boxes: [N, 4] (x1, y1, x2, y2)
        class_ids: [N]
        confidences: Optional [N]
        class_names: Class name list
        line_thickness: Box line thickness
    """
    if class_names is None:
        class_names = VISDRONE_CLASSES

    img = image.copy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(class_ids[i]) % len(CLASS_COLORS)
        color = CLASS_COLORS[cls_id]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # Label
        label = class_names[int(class_ids[i])] if int(class_ids[i]) < len(class_names) else f"cls_{int(class_ids[i])}"
        if confidences is not None and i < len(confidences):
            label += f" {confidences[i]:.2f}"

        # Text background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - baseline - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def draw_tracks(
    image: np.ndarray,
    tracks: List[Dict],
    class_names: Optional[List[str]] = None,
    trail_length: int = 30,
) -> np.ndarray:
    """
    Draw tracking trajectories and boxes.

    Args:
        image: BGR image
        tracks: List of track dicts with "state", "id", "trajectory", "cls"
        class_names: Class names
        trail_length: Trajectory trail length
    """
    if class_names is None:
        class_names = VISDRONE_CLASSES

    img = image.copy()

    for track in tracks:
        track_id = track.get("id", 0)
        state = track.get("state", np.zeros(8))
        trajectory = track.get("trajectory", [])
        cls_id = track.get("cls", 0)

        color = CLASS_COLORS[track_id % len(CLASS_COLORS)]

        # Draw current box
        if len(state) >= 4:
            x1, y1, x2, y2 = state[:4].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # ID label
            label = f"ID:{track_id}"
            if cls_id < len(class_names):
                label = f"{class_names[int(cls_id)]} {label}"

            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw trajectory trail
        if trajectory and len(trajectory) > 1:
            trail = trajectory[-trail_length:]
            for j in range(1, len(trail)):
                pt1 = tuple(trail[j - 1][:2].astype(int))
                pt2 = tuple(trail[j][:2].astype(int))
                alpha = j / len(trail)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(img, pt1, pt2, trail_color, 1)

        # Draw velocity arrow
        if len(state) >= 8:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vx, vy = state[6], state[7]  # Velocity components
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0:
                vx_norm = int(vx / speed * 30)
                vy_norm = int(vy / speed * 30)
                cv2.arrowedLine(img, (cx, cy), (cx + vx_norm, cy + vy_norm), (0, 0, 255), 2)

    return img


def visualize_detections(
    image_path: str,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Load image and draw detections."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    result = draw_detections(img, boxes, class_ids, confidences, class_names)

    if output_path:
        cv2.imwrite(str(output_path), result)

    return result


def visualize_tracks(
    image_path: str,
    tracks: List[Dict],
    output_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Load image and draw tracks."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    result = draw_tracks(img, tracks, class_names)

    if output_path:
        cv2.imwrite(str(output_path), result)

    return result


def plot_trajectories(
    tracks: List[Dict],
    output_path: Optional[str] = None,
    title: str = "Object Trajectories",
    figsize: Tuple[int, int] = (10, 8),
):
    """Plot 2D top-down trajectory view using matplotlib."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for track in tracks:
        trajectory = track.get("trajectory", [])
        if trajectory and len(trajectory) > 1:
            traj = np.array(trajectory)
            track_id = track.get("id", 0)
            color = [c / 255 for c in CLASS_COLORS[track_id % len(CLASS_COLORS)][::-1]]  # BGR to RGB

            ax.plot(traj[:, 0], traj[:, 1], '-o', color=color, markersize=2, linewidth=1, label=f"ID {track_id}")
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=6)  # Start
            ax.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=6)  # End

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_video_visualization(
    frame_paths: List[str],
    all_detections: List[np.ndarray],
    all_class_ids: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    class_names: Optional[List[str]] = None,
):
    """Create visualization video from frames and detections."""
    if not frame_paths:
        return

    first_frame = cv2.imread(str(frame_paths[0]))
    H, W = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for i, frame_path in enumerate(frame_paths):
        if i < len(all_detections):
            img = visualize_detections(
                frame_path,
                all_detections[i],
                all_class_ids[i] if i < len(all_class_ids) else np.array([]),
                class_names=class_names,
            )
        else:
            img = cv2.imread(str(frame_path))

        writer.write(img)

    writer.release()
    print(f"Video saved to {output_path}")


def draw_hd_similarity_matrix(
    similarity_matrix: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "HD Similarity Matrix",
):
    """Visualize HD similarity matrix (detections vs tracks)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(similarity_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xlabel("Tracks")
    ax.set_ylabel("Detections")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()