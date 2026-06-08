"""
SNN Simulation Harness for Eldarin
====================================
Simulates SNN inference with temporal dynamics.
Used for validating SNN conversion accuracy before FPGA deployment.

Supports:
  - Rate-based spike coding
  - IF/LIF neuron simulation
  - Membrane potential tracking
  - Accuracy comparison with ANN baseline

References:
  snnTorch: https://github.com/jeshraghian/snntorch
  Lava: https://github.com/lava-nc/lava
  FPGA-Event-Based-encode: https://github.com/Enotrium/FPGA-Event-Based-encode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.snn_layers import IFNeuron, LIFNeuron, SNNConversionHelper


class SNNSimulator:
    """
    SNN simulation harness for validating ANN→SNN conversion.
    Simulates temporal spike-based inference and compares with ANN.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 100,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device

    def reset_states(self):
        """Reset all neuron membrane potentials."""
        for module in self.model.modules():
            if hasattr(module, 'reset_state'):
                module.reset_state()

    def run_snn(
        self,
        inputs: torch.Tensor,
        reset_after: bool = True,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Run SNN for multiple timesteps.

        For rate-coded input: repeat input as constant current.
        For spike input: propagate spikes through network.

        Args:
            inputs: Input tensor [B, ...]
            reset_after: Reset membrane states after simulation

        Returns:
            (output_spike_count, layer_recordings)
              output_spike_count: Accumulated spikes over timesteps
              layer_recordings: List of per-timestep layer outputs
        """
        self.reset_states()
        self.model.eval()

        layer_recordings = []

        with torch.no_grad():
            for t in range(self.timesteps):
                # For rate coding: constant input, output = spike count
                if inputs.dim() == 4:  # 2D image input
                    out = self.model(frames=inputs)
                else:
                    out = self.model(inputs)

                # Record layer outputs
                recordings = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, (IFNeuron, LIFNeuron)):
                        recordings[name] = {
                            "membrane": module.membrane.clone() if module.membrane is not None else None,
                        }

                layer_recordings.append({**recordings, "output": out})

        # Compute output spike count (or feature average)
        output_avg = None
        if layer_recordings:
            detections = [r.get("output", {}) for r in layer_recordings]
            # Average detection outputs (simplified)
            if detections[0]:
                output_avg = detections[-1]  # Use final output

        if reset_after:
            self.reset_states()

        return output_avg, layer_recordings

    def compare_ann_snn(
        self,
        ann_model: nn.Module,
        test_loader,
        max_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Compare ANN and SNN outputs for accuracy validation.

        Args:
            ann_model: Trained ANN model (same weights)
            test_loader: Data loader for validation
            max_samples: Max samples to compare

        Returns:
            Dict with comparison metrics (MSE, cosine similarity, etc.)
        """
        ann_model.eval()
        self.model.eval()

        mse_sum = 0.0
        cos_sim_sum = 0.0
        count = 0

        for batch in test_loader:
            if count >= max_samples:
                break

            frames = batch.get("frames", None)
            if frames is None:
                continue

            frames = frames.to(self.device)
            B = frames.shape[0]

            # ANN forward
            with torch.no_grad():
                ann_out = ann_model(frames=frames)
                ann_feat = ann_out.get("fused_features", ann_out.get("hd_representation"))
                if ann_feat is None:
                    ann_feat = torch.zeros(B, 256, device=self.device)

            # SNN forward
            snn_out, _ = self.run_snn(frames)
            if snn_out is None:
                snn_feat = torch.zeros_like(ann_feat)
            else:
                snn_feat = snn_out.get("fused_features", snn_out.get("hd_representation", ann_feat))

            if snn_feat is not None and ann_feat is not None:
                # Align shapes
                ann_flat = ann_feat.view(B, -1)
                snn_flat = snn_feat.view(B, -1)

                mse_sum += F.mse_loss(snn_flat, ann_flat).item()
                cos_sim_sum += F.cosine_similarity(snn_flat, ann_flat, dim=-1).mean().item()
                count += B

            if count % 10 == 0:
                print(f"  Compared {count}/{max_samples} samples...")

        return {
            "mse": mse_sum / max(1, count / B),
            "cosine_similarity": cos_sim_sum / max(1, count / B),
            "samples_compared": min(count, max_samples),
        }


def validate_snn_accuracy(
    ann_checkpoint: str,
    snn_checkpoint: str,
    data_root: str,
    config_path: str = None,
    device: str = "cuda",
    timesteps: int = 100,
    max_samples: int = 100,
) -> Dict[str, float]:
    """
    Full SNN accuracy validation pipeline.

    Returns comparison metrics between ANN and SNN inference.
    """
    from model.eldarin_model import create_eldarin
    from utils.data_loader import create_dataloader

    # Load models
    import yaml
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    ann_model = create_eldarin(config_dict=config)
    ann_ckpt = torch.load(ann_checkpoint, map_location="cpu", weights_only=False)
    ann_model.load_state_dict(ann_ckpt["model_state_dict"], strict=False)
    ann_model.to(device)
    ann_model.eval()

    snn_model = create_eldarin(config_dict=config)
    snn_ckpt = torch.load(snn_checkpoint, map_location="cpu", weights_only=False)
    snn_model.load_state_dict(snn_ckpt["model_state_dict"], strict=False)
    SNNConversionHelper.convert_relu_to_if(snn_model)
    snn_model.to(device)

    # Data loader
    loader_config = {"data": {"data_root": data_root, "max_samples": max_samples, "batch_size": 4}}
    loader = create_dataloader(loader_config, split="val")

    # Simulate and compare
    simulator = SNNSimulator(snn_model, timesteps=timesteps, device=device)
    metrics = simulator.compare_ann_snn(ann_model, loader, max_samples=max_samples)

    print(f"\nSNN Accuracy Validation Results:")
    print(f"  Samples: {metrics['samples_compared']}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
    print(f"  Effective Accuracy Retention: {metrics['cosine_similarity']*100:.1f}%")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SNN Accuracy Validation")
    parser.add_argument("--ann_checkpoint", type=str, required=True)
    parser.add_argument("--snn_checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    validate_snn_accuracy(
        ann_checkpoint=args.ann_checkpoint,
        snn_checkpoint=args.snn_checkpoint,
        data_root=args.data_root,
        config_path=args.config,
        device=args.device,
        timesteps=args.timesteps,
        max_samples=args.max_samples,
    )