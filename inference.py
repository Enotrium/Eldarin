#!/usr/bin/env python3
"""
Eldarin Inference Script
==========================
Real-time UAV inference for detection, 4D tracking, and visualization.
Supports RGB video, event streams, and multi-modal input.


Integrations:
  - FPGA-Event-Based-encode: https://github.com/Enotrium/FPGA-Event-Based-encode
  - arthedain-1 VSA/HDC: https://github.com/Enotrium/arthedain-1

Usage:
    python inference.py --config config/inference.yaml --checkpoint checkpoints/best_model.pth --input video.mp4
    python inference.py --checkpoint checkpoints/eldarin_v1.pth --input /path/to/video.mp4 --modality rgb+event --output results/
    python inference.py --checkpoint checkpoints/eldarin_v1.pth --input 0  # Webcam
"""

import argparse
import yaml
import torch
import numpy as np
import cv2
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from model.eldarin_model import create_eldarin
from utils.visualization import draw_detections, draw_tracks, plot_trajectories, VISDRONE_CLASSES
from utils.event_utils import EventProcessor, load_events_from_file
from utils.metrics import box_iou

logger = logging.getLogger(__name__)


class EldarinInference:
    """
    Real-time inference engine for Eldarin.
    Handles:
      - Model loading and device management
      - Multi-modal input processing (RGB, event, audio, IMU)
      - Detection post-processing (NMS)
      - 4D tracking via model's built-in TrackingHead (HD Kalman filter)
      - Visualization
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        conf_thresh: float = 0.25,
        nms_iou: float = 0.45,
        max_det: int = 300,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf_thresh = conf_thresh
        self.nms_iou = nms_iou
        self.max_det = max_det

        # Load model
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        self.model = create_eldarin(config_dict=config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Class names
        self.class_names = VISDRONE_CLASSES

        # Delegate tracking to the model's built-in TrackingHead
        # This uses the HD Kalman filter + Hungarian association (fixed)
        self._tracks = []  # Mirror of TrackingHead tracks for visualization
        self.frame_id = 0

    def preprocess_frame(
        self, frame: np.ndarray, img_size: Tuple[int, int] = (640, 640)
    ) -> torch.Tensor:
        """Preprocess RGB frame to tensor."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size[::-1])
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def postprocess_detections(
        self, predictions: Dict[str, torch.Tensor], orig_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process raw model predictions.

        Returns:
            boxes: [N, 4] (x1, y1, x2, y2) in original image coords
            class_ids: [N]
            confidences: [N]
        """
        det_out = predictions.get("detection", {})
        if not det_out:
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0)

        bbox_pred = det_out.get("bbox", None)
        cls_pred = det_out.get("cls", None)
        obj_pred = det_out.get("obj", None)

        if bbox_pred is None or bbox_pred.numel() == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0)

        # Compute confidence
        if obj_pred is not None and obj_pred.numel() > 0:
            conf = torch.sigmoid(obj_pred.squeeze(-1))
        else:
            conf = torch.ones(bbox_pred.shape[0], 1, device=bbox_pred.device)

        cls_ids_out = np.zeros(0, dtype=np.int64)
        if cls_pred is not None and cls_pred.numel() > 0:
            cls_flat = cls_pred.reshape(-1, cls_pred.shape[-1])
            cls_conf, cls_ids = cls_flat.softmax(-1).max(-1)
            if conf.numel() == cls_conf.numel():
                conf = conf * cls_conf.unsqueeze(-1) if conf.dim() < 2 else conf.squeeze(-1) * cls_conf

        conf_np = conf.cpu().numpy().flatten() if torch.is_tensor(conf) else np.array(conf)

        # Filter by confidence
        mask = conf_np > self.conf_thresh
        boxes_raw = bbox_pred.reshape(-1, 4).cpu().numpy()

        if mask.sum() == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0)

        boxes = boxes_raw[mask]
        conf_filtered = conf_np[mask]

        if cls_pred is not None and cls_pred.numel() > 0:
            cls_flat = cls_pred.reshape(-1, cls_pred.shape[-1])
            if cls_flat.shape[0] == len(mask):
                _, cls_ids = cls_flat.softmax(-1).max(-1)
                cls_ids_out = cls_ids[mask].cpu().numpy()
            else:
                cls_ids_out = np.zeros(boxes.shape[0], dtype=np.int64)

        # Rescale to original image size
        orig_h, orig_w = orig_size
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h

        # NMS
        keep = self._nms(boxes, conf_filtered)
        if len(keep) == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0)

        boxes = boxes[keep][:self.max_det]
        conf_filtered = conf_filtered[keep][:self.max_det]
        cls_ids_out = cls_ids_out[keep][:self.max_det] if len(cls_ids_out) > 0 else np.zeros(boxes.shape[0], dtype=np.int64)

        return boxes, cls_ids_out, conf_filtered

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= self.nms_iou)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int64)

    def update_tracking(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        features: torch.Tensor,
    ):
        """
        Update 4D tracking state using the model's built-in TrackingHead
        with HD Kalman filter and Hungarian association.

        This replaces the previous duplicate IoU-based logic in inference.py
        that bypassed the HD Kalman filter entirely.
        """
        if not hasattr(self.model, 'tracking_head') or self.model.tracking_head is None:
            # Fallback: simple IoU-based tracking if model doesn't have tracking head
            self._update_tracking_fallback(boxes, class_ids, confidences)
            return

        # Convert boxes to detections tensor for TrackingHead
        if len(boxes) == 0:
            # Just age tracks through the tracking head
            det_state = torch.zeros(0, 8, device=self.device)
            det_features = torch.zeros(0, 128, device=self.device)
        else:
            det_state = np.zeros((len(boxes), 8), dtype=np.float32)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                det_state[i] = [cx, cy, w, h, 0, 0, 0, 0]

            det_state = torch.from_numpy(det_state).to(self.device)

            # Use provided features for ReID, fall back to zeros
            if features.numel() > 0:
                n_feat = min(features.shape[0], len(boxes))
                det_features = torch.zeros(len(boxes), 128, device=self.device)
                det_features[:n_feat] = features[:n_feat]
            else:
                det_features = torch.zeros(len(boxes), 128, device=self.device)

        # Delegate to TrackingHead (uses Hungarian + HD Kalman)
        with torch.no_grad():
            tracking_output = self.model.tracking_head(
                detections=det_state,
                detection_features=det_features,
                features_for_reid=det_features,
                tracks=self._tracks,
                frame_id=self.frame_id,
            )

        self._tracks = tracking_output["tracks"]

    def _update_tracking_fallback(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
    ):
        """Fallback IoU-based tracking when TrackingHead is not available."""
        if len(boxes) == 0:
            for track in self._tracks:
                track["age"] += 1
            self._tracks = [t for t in self._tracks if t["age"] <= 30]
            return

        det_states = np.zeros((len(boxes), 8), dtype=np.float32)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            det_states[i] = [cx, cy, w, h, 0, 0, 0, 0]

        if self._tracks:
            track_boxes = np.array([t["state"][:4] for t in self._tracks])
            iou_matrix = box_iou(
                torch.from_numpy(det_states[:, :4]),
                torch.from_numpy(track_boxes)
            ).numpy()

            matched = set()
            for d_idx in range(len(det_states)):
                if iou_matrix.shape[1] > 0:
                    best_t = iou_matrix[d_idx].argmax()
                    if iou_matrix[d_idx, best_t] > 0.3:
                        self._tracks[best_t]["state"] = (
                            0.7 * self._tracks[best_t]["state"] + 0.3 * det_states[d_idx]
                        )
                        self._tracks[best_t]["age"] = 0
                        self._tracks[best_t]["hits"] += 1
                        self._tracks[best_t].setdefault("trajectory", []).append(det_states[d_idx][:2])
                        matched.add(d_idx)

            for d_idx in range(len(det_states)):
                if d_idx not in matched:
                    self._tracks.append({
                        "id": self.frame_id * 1000 + len(self._tracks),
                        "state": det_states[d_idx],
                        "age": 0,
                        "hits": 1,
                        "trajectory": [det_states[d_idx][:2]],
                        "cls": class_ids[d_idx] if d_idx < len(class_ids) else 0,
                    })

            for track in self._tracks:
                if track.get("age", 0) > 0:
                    track["age"] += 1 if "age" in track else 1
        else:
            for d_idx in range(len(det_states)):
                self._tracks.append({
                    "id": self.frame_id * 1000 + d_idx,
                    "state": det_states[d_idx],
                    "age": 0,
                    "hits": 1,
                    "trajectory": [det_states[d_idx][:2]],
                    "cls": class_ids[d_idx] if d_idx < len(class_ids) else 0,
                })

        self._tracks = [t for t in self._tracks if t.get("age", 0) <= 30]

    def process_frame(
        self,
        frame: np.ndarray,
        events: Optional[Tuple[torch.Tensor, ...]] = None,
        audio: Optional[torch.Tensor] = None,
        visualize: bool = True,
    ) -> Dict:
        """Process a single frame through the full pipeline."""
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        frames_tensor = self.preprocess_frame(frame).to(self.device)

        # Model inference
        with torch.no_grad():
            predictions = self.model(
                frames=frames_tensor,
                events=events,
                audio=audio.to(self.device) if audio is not None else None,
            )

        # Post-process detections
        boxes, class_ids, confidences = self.postprocess_detections(
            predictions, (orig_h, orig_w)
        )

        # Extract features for tracking
        features = predictions.get("fused_features", torch.zeros(1, 256))
        if features.dim() == 4:
            # Average over spatial dims: [B, C, H, W] -> [B, C]
            features = features.mean(dim=[-2, -1])
        elif features.dim() == 3:
            features = features.mean(dim=1)

        # Update tracking via model's TrackingHead
        self.update_tracking(boxes, class_ids, confidences, features)
        self.frame_id += 1

        result = {
            "boxes": boxes,
            "class_ids": class_ids,
            "confidences": confidences,
            "tracks": self._tracks,
        }

        # Visualization
        if visualize:
            vis_frame = draw_detections(frame, boxes, class_ids, confidences, self.class_names)
            active_tracks = [t for t in self._tracks if t.get("age", 0) == 0 and t.get("hits", 0) >= 3]
            if active_tracks:
                vis_frame = draw_tracks(vis_frame, active_tracks, self.class_names)
            result["visualization"] = vis_frame

        return result

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_display: bool = False,
        trail_plot_every: int = 100,
    ):
        """Process entire video file."""
        cap = cv2.VideoCapture(video_path if video_path != "0" else 0)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {W}x{H} @ {fps:.0f}fps, {total_frames} frames")

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        frame_count = 0
        processing_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()

            result = self.process_frame(frame, visualize=True)
            processing_times.append(time.time() - t_start)

            vis_frame = result.get("visualization", frame)

            # FPS counter
            fps_str = f"FPS: {1.0 / max(np.mean(processing_times[-30:]), 0.001):.1f}"
            cv2.putText(vis_frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Track count
            n_tracks = len([t for t in result["tracks"] if t.get("age", 0) == 0 and t.get("hits", 0) >= 3])
            cv2.putText(vis_frame, f"Tracks: {n_tracks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if writer:
                writer.write(vis_frame)

            if show_display:
                cv2.imshow("Eldarin", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                avg_fps = 1.0 / max(np.mean(processing_times[-100:]), 0.001)
                logger.info(f"Frame {frame_count}/{total_frames} | Avg FPS: {avg_fps:.1f} | Tracks: {n_tracks}")

            # Periodic trajectory plot
            if trail_plot_every > 0 and frame_count % trail_plot_every == 0 and output_path:
                plot_path = Path(output_path).parent / f"trajectories_{frame_count:06d}.png"
                active_tracks = [t for t in result["tracks"] if len(t.get("trajectory", [])) > 2]
                if active_tracks:
                    plot_trajectories(active_tracks, str(plot_path))

        cap.release()
        if writer:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()

        avg_fps = 1.0 / max(np.mean(processing_times), 0.001)
        logger.info(f"\nDone! Processed {frame_count} frames in {np.sum(processing_times):.1f}s")
        logger.info(f"Average FPS: {avg_fps:.1f}")

    def reset_tracks(self):
        """Reset tracking state."""
        self._tracks = []
        self.frame_id = 0
        if hasattr(self.model, 'tracking_head') and self.model.tracking_head is not None:
            self.model.tracking_head.reset()


def main():
    parser = argparse.ArgumentParser(description="Eldarin Inference")
    parser.add_argument("--config", type=str, default=None, help="Config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Video path or '0' for webcam")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms_iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--show", action="store_true", help="Show display window")
    parser.add_argument("--trail_plot_every", type=int, default=0, help="Plot trajectories every N frames")

    args = parser.parse_args()

    engine = EldarinInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        conf_thresh=args.conf_thresh,
        nms_iou=args.nms_iou,
    )

    engine.process_video(
        video_path=args.input,
        output_path=args.output,
        show_display=args.show,
        trail_plot_every=args.trail_plot_every,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()