"""
Eldarin Trainer
=================
Training loop with mixed precision, checkpointing, and evaluation.
 training pipeline.


"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Tuple
import os
import time
from pathlib import Path
from tqdm import tqdm
import json
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training manager for Eldarin model.

    Handles:
      - Training loop with mixed precision (AMP)
      - Validation loop
      - Checkpoint saving/loading
      - Learning rate scheduling
      - Metrics logging (TensorBoard / wandb)
      - SNN calibration (optional)

    Args:
        model: Eldarin model
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        device: Device string
        output_dir: Checkpoint/log output directory
        use_wandb: Enable Weights & Biases logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "checkpoints",
        use_wandb: bool = False,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.grad_clip = train_cfg.get("gradient_clip", 10.0)
        self.save_interval = train_cfg.get("save_interval", 5)
        self.eval_interval = train_cfg.get("eval_interval", 1)
        self.use_amp = train_cfg.get("amp", True)

        # Loss
        if loss_fn is None:
            from .loss import EldarinLoss
            loss_weights = train_cfg.get("loss", {})
            loss_fn = EldarinLoss(loss_weights)
        self.loss_fn = loss_fn.to(device)

        # Optimizer
        if optimizer is None:
            lr = train_cfg.get("lr", 0.001)
            weight_decay = train_cfg.get("weight_decay", 0.0005)
            momentum = train_cfg.get("momentum", 0.937)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
            )
        self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs - self.warmup_epochs
            )
        self.scheduler = scheduler

        # Mixed precision
        self.scaler = GradScaler(enabled=self.use_amp)

        # Metrics
        self.best_loss = float("inf")
        self.current_epoch = 0
        self.global_step = 0

        # Wandb
        if self.use_wandb:
            import wandb
            wandb.init(project="Eldarin", config=config)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = self._to_device(batch)

            # Warmup
            if self.current_epoch < self.warmup_epochs:
                warmup_factor = (self.current_epoch * num_batches + batch_idx + 1) / (
                    self.warmup_epochs * num_batches
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * warmup_factor

            # Forward pass
            with autocast(enabled=self.use_amp):
                predictions = self.model(
                    frames=batch.get("frames"),
                    events=batch.get("events"),
                    event_duration_us=batch.get("event_duration_us"),
                    audio=batch.get("audio"),
                    imu_data=batch.get("imu_data"),
                )

                losses = self.loss_fn(predictions, batch.get("targets", {}))

            # Backward
            total_loss = losses.get("total")
            if total_loss is None or not torch.isfinite(total_loss):
                logger.warning(f"Non-finite loss at batch {batch_idx}, skipping")
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(total_loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Accumulate losses
            for k, v in losses.items():
                if k == "total":
                    epoch_losses[k] = epoch_losses.get(k, 0) + v.item()

            # Update progress bar
            batch_total = total_loss.item()
            pbar.set_postfix({
                "loss": f"{batch_total:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })

            self.global_step += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {}
        num_batches = len(self.val_loader)

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._to_device(batch)

            with autocast(enabled=self.use_amp):
                predictions = self.model(
                    frames=batch.get("frames"),
                    events=batch.get("events"),
                )
                losses = self.loss_fn(predictions, batch.get("targets", {}))

            for k, v in losses.items():
                if k == "total":
                    val_losses[k] = val_losses.get(k, 0) + v.item()

        for k in val_losses:
            val_losses[k] /= max(num_batches, 1)

        return val_losses

    def train(self, resume_from: Optional[str] = None):
        """
        Full training loop.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Resume
        if resume_from:
            self.load_checkpoint(resume_from)

        self.model.to(self.device)

        logger.info(f"Starting training: {self.epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = {}
            if self.val_loader and (epoch + 1) % self.eval_interval == 0:
                val_losses = self.validate()

            # Scheduler step
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Logging
            elapsed = time.time() - start_time
            log_str = f"Epoch {epoch + 1}/{self.epochs} | Time: {elapsed:.1f}s | Train Loss: {train_losses.get('total', 0):.4f}"
            if val_losses:
                log_str += f" | Val Loss: {val_losses.get('total', 0):.4f}"
            logger.info(log_str)

            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train": train_losses,
                    "val": val_losses,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

            # Checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

            # Best model
            current_loss = val_losses.get("total", train_losses.get("total", float("inf")))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint("best_model.pth")
                logger.info(f"  -> Best model saved (loss: {current_loss:.4f})")

        logger.info(f"Training complete! Best loss: {self.best_loss:.4f}")
        self.save_checkpoint("last_model.pth")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def _to_device(self, batch) -> Dict:
        """
        Move batch to device, handling nested structures.

        Uses iterative approach to avoid infinite recursion on
        deeply-nested structures such as event tuples (x, y, t, p)
        which must be preserved as tuples of tensors.
        """
        # Handle dicts
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}

        # Handle lists (non-recursive for tensors list)
        if isinstance(batch, list):
            return [
                self._to_device(b) if isinstance(b, (dict, tuple))
                else b.to(self.device) if isinstance(b, torch.Tensor)
                else b
                for b in batch
            ]

        # Handle tuples — preserve structure, move inner tensors
        # Event data tuples like (x, y, t, p) must stay as tuples
        if isinstance(batch, tuple):
            return tuple(
                self._to_device(b) if isinstance(b, (dict, list))
                else b.to(self.device) if isinstance(b, torch.Tensor)
                else b
                for b in batch
            )

        # Handle individual tensors
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)

        # Passthrough for everything else (strings, None, numbers)
        return batch

    def export_onnx(self, output_path: str, input_shape: Tuple = (1, 3, 640, 640)):
        """Export model to ONNX format."""
        self.model.eval()
        dummy_input = torch.randn(*input_shape).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["frames"],
            output_names=["detections", "hd_representation"],
            dynamic_axes={
                "frames": {0: "batch"},
                "detections": {0: "num_detections"},
            },
            opset_version=14,
        )
        logger.info(f"ONNX model exported to {output_path}")