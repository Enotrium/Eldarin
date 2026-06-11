#!/usr/bin/env python3
"""
Eldarin Training Script
=========================
Main entry point for training the hierarchical multimodal 4D detection
and tracking model for UAV applications.


Integrations:
  - FPGA-Event-Based-encode: https://github.com/Enotrium/FPGA-Event-Based-encode
  - arthedain-1 VSA/HDC: https://github.com/Enotrium/arthedain-1

Usage:
    # Single GPU training with VisDrone
    python main.py --config config/train_visdrone.yaml --data_root /path/to/VisDrone

    # Multi-modal training
    python main.py --config config/train_multimodal.yaml --modality rgb+event

    # Distributed training
    python -m torch.distributed.launch --nproc_per_node=4 main.py --distributed

    # Resume from checkpoint
    python main.py --config config/train_visdrone.yaml --resume checkpoints/last_model.pth
"""

import argparse
import yaml
import torch
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.eldarin_model import create_eldarin
from model.vsa_hdc import test_vsa_operations
from utils.data_loader import create_dataloader
from utils.loss import EldarinLoss
from utils.metrics import DetectionMetrics, compute_mAP
from utils.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eldarin: Hierarchical Multimodal 4D Detection & Tracking for UAVs"
    )

    # Config
    parser.add_argument("--config", type=str, default="config/base.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override dataset root path")
    parser.add_argument("--modality", type=str, default=None,
                        help="Override modalities (e.g., 'rgb+event')")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")

    # Training mode
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda / cpu)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints and logs")

    # Validation
    parser.add_argument("--val_only", action="store_true",
                        help="Run validation only")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint for validation")

    # SNN / FPGA
    parser.add_argument("--convert_snn", action="store_true",
                        help="Convert model to SNN after training")
    parser.add_argument("--test_vsa", action="store_true",
                        help="Run VSA/HDC algebra tests")

    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (fewer samples)")

    return parser.parse_args()


def run_validation(model, val_loader, device, num_classes: int = 10):
    """
    Run a proper validation pass computing mAP and MOTA metrics.

    This replaces the previous placeholder that used torch.rand.
    """
    from utils.metrics import DetectionMetrics, compute_mAP

    det_metrics = DetectionMetrics(num_classes=num_classes)

    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        # Move data to device
        frames = batch.get("frames")
        if isinstance(frames, torch.Tensor):
            frames = frames.to(device)

        events = batch.get("events")
        if isinstance(events, tuple):
            events = tuple(e.to(device) if isinstance(e, torch.Tensor) else e for e in events)

        with torch.no_grad():
            predictions = model(
                frames=frames,
                events=events,
            )

        # Extract detections from predictions
        det_out = predictions.get("detection", {})
        if not det_out:
            continue

        bbox_pred = det_out.get("bbox", None)
        cls_pred = det_out.get("cls", None)
        obj_pred = det_out.get("obj", None)

        if bbox_pred is None or bbox_pred.numel() == 0:
            continue

        # Build detection tensor for metrics
        # Format: [N, 6] (x1, y1, x2, y2, conf, cls)
        bbox_np = bbox_pred.reshape(-1, 4).cpu().numpy()
        conf_np = torch.sigmoid(obj_pred.reshape(-1)).cpu().numpy() if obj_pred is not None and obj_pred.numel() > 0 else np.ones(bbox_np.shape[0])

        if cls_pred is not None and cls_pred.numel() > 0:
            cls_flat = cls_pred.reshape(-1, cls_pred.shape[-1])
            cls_ids = cls_flat.argmax(-1).cpu().numpy()
        else:
            cls_ids = np.zeros(bbox_np.shape[0], dtype=np.int64)

        # Filter confident detections
        conf_thresh = 0.25
        keep = conf_np > conf_thresh

        dets = np.concatenate([
            bbox_np[keep],
            conf_np[keep, np.newaxis],
            cls_ids[keep, np.newaxis].astype(np.float32),
        ], axis=1) if keep.sum() > 0 else np.zeros((0, 6))

        # Get targets
        targets_data = batch.get("targets", {})
        if isinstance(targets_data, dict):
            gt_bboxes = targets_data.get("bboxes", torch.zeros(0, 4))
            gt_classes = targets_data.get("classes", torch.zeros(0, dtype=torch.long))
            if gt_bboxes.numel() > 0:
                gts = torch.cat([
                    gt_bboxes.cpu(),
                    gt_classes.cpu().float().unsqueeze(-1),
                ], dim=1)
            else:
                gts = torch.zeros(0, 5)
        else:
            gts = torch.zeros(0, 5)

        det_metrics.update(
            torch.from_numpy(dets),
            gts,
        )

        if batch_idx % 50 == 0:
            logger.info(f"Validation batch {batch_idx}/{len(val_loader)}")

    results = det_metrics.compute()
    logger.info(f"Validation Results: mAP={results['mAP']:.2f}%, "
                f"Precision={results['precision']:.2f}%, "
                f"Recall={results['recall']:.2f}%")
    return results


def main():
    args = parse_args()

    # Test VSA operations if requested
    if args.test_vsa:
        test_vsa_operations()
        return

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.data_root:
        config.setdefault("data", {})["data_root"] = args.data_root
    if args.modality:
        config.setdefault("data", {})["modalities"] = args.modality
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("training", {})["lr"] = args.lr
    if args.debug:
        config.setdefault("data", {})["max_samples"] = 100

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    # Create model
    logger.info("Building Eldarin model...")
    model = create_eldarin(config_dict=config)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count/1e6:.2f}M total, {trainable_count/1e6:.2f}M trainable")

    # Data loaders
    logger.info("Loading datasets...")
    train_loader = create_dataloader(config, split="train")

    val_loader = None
    data_cfg = config.get("data", {})
    val_root = data_cfg.get("val_root", data_cfg.get("data_root"))
    if val_root:
        val_config = config.copy()
        val_config["data"]["data_root"] = val_root
        try:
            val_loader = create_dataloader(val_config, split="val")
            logger.info(f"Validation samples: {len(val_loader.dataset)}")
        except Exception as e:
            logger.warning(f"No validation data found: {e}")

    # Validation-only mode — proper evaluation with real metrics
    if args.val_only:
        import numpy as np  # needed for eval
        checkpoint_path = args.checkpoint or args.resume
        if checkpoint_path is None:
            logger.error("Error: --checkpoint required for --val_only")
            sys.exit(1)

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        eval_loader = val_loader or train_loader
        if eval_loader is None:
            logger.error("No data loader available for validation")
            sys.exit(1)

        num_classes = config.get("data", {}).get("num_classes", 10)
        results = run_validation(model, eval_loader, device, num_classes)
        return

    # Loss function
    loss_weights = config.get("training", {}).get("loss", {})
    loss_fn = EldarinLoss(loss_weights)

    # Optimizer
    train_cfg = config.get("training", {})
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg.get("lr", 0.001),
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        nesterov=True,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("epochs", 100) - train_cfg.get("warmup_epochs", 5),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )

    # Resume
    resume_from = args.resume or args.checkpoint
    if resume_from:
        logger.info(f"Resuming from {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Train
    try:
        trainer.train(resume_from=resume_from if not resume_from else None)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint("interrupted.pth")
        logger.info("Checkpoint saved.")

    # SNN conversion
    if args.convert_snn:
        logger.info("Converting to SNN...")
        model.to_snn()
        trainer.save_checkpoint("eldarin_snn.pth")
        logger.info("SNN model saved to checkpoints/eldarin_snn.pth")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()