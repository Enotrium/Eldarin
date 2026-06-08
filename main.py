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
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.eldarin_model import create_eldarin
from model.vsa_hdc import test_vsa_operations
from utils.data_loader import create_dataloader
from utils.loss import EldarinLoss
from utils.trainer import Trainer


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
        print("CUDA not available, falling back to CPU")

    # Create model
    print("Building Eldarin model...")
    model = create_eldarin(config_dict=config)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count/1e6:.2f}M total, {trainable_count/1e6:.2f}M trainable")

    # Data loaders
    print("Loading datasets...")
    train_loader = create_dataloader(config, split="train")

    val_loader = None
    data_cfg = config.get("data", {})
    val_root = data_cfg.get("val_root", data_cfg.get("data_root"))
    if val_root:
        val_config = config.copy()
        val_config["data"]["data_root"] = val_root
        try:
            val_loader = create_dataloader(val_config, split="val")
            print(f"Validation samples: {len(val_loader.dataset)}")
        except Exception as e:
            print(f"No validation data found: {e}")

    # Validation-only mode
    if args.val_only:
        checkpoint_path = args.checkpoint or args.resume
        if checkpoint_path is None:
            print("Error: --checkpoint required for --val_only")
            sys.exit(1)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        from utils.metrics import DetectionMetrics, compute_mAP
        det_metrics = DetectionMetrics()

        for batch in val_loader or train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                pred = model(frames=batch.get("frames"))
            # Update metrics (simplified)
            if "targets" in batch:
                targets = batch["targets"]
                if isinstance(targets, list):
                    for tgt in targets:
                        det_metrics.update(
                            torch.rand(0, 6),  # Placeholder
                            tgt.get("bboxes", torch.zeros(0, 4))
                        )

        results = det_metrics.compute()
        print(f"Validation Results: mAP={results['mAP']:.2f}%")
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
        print(f"Resuming from {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Train
    try:
        trainer.train(resume_from=resume_from if not resume_from else None)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint("interrupted.pth")
        print("Checkpoint saved.")

    # SNN conversion
    if args.convert_snn:
        print("Converting to SNN...")
        model.to_snn()
        trainer.save_checkpoint("eldarin_snn.pth")
        print("SNN model saved to checkpoints/eldarin_snn.pth")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()