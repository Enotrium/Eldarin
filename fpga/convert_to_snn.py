"""
ANN → SNN Conversion for Eldarin
==================================
Converts trained Eldarin ANN to SNN for neuromorphic/FPGA deployment.
Supports snnTorch, Lava, and custom FPGA HLS paths.

Process:
  1. Load trained checkpoint
  2. Replace ReLU/SiLU with IF/LIF neurons
  3. Calibrate thresholds using validation data
  4. Export as SNN checkpoint

References:
  FPGA-Event-Based-encode: https://github.com/Enotrium/FPGA-Event-Based-encode
  arthedain-1 VSA/HDC: https://github.com/Enotrium/arthedain-1
  
"""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.eldarin_model import create_eldarin
from model.snn_layers import SNNConversionHelper, IFNeuron, LIFNeuron, SNNConv2d, SNNLinear


def convert_to_snn(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    output_path: str = "checkpoints/eldarin_snn.pth",
    threshold: float = 1.0,
    neuron_type: str = "LIF",
    membrane_decay: float = 0.9,
) -> nn.Module:
    """
    Convert ANN model to SNN.

    Args:
        checkpoint_path: Path to trained ANN checkpoint
        config_path: Optional config YAML
        output_path: Output SNN checkpoint path
        threshold: IF/LIF neuron threshold
        neuron_type: "IF" or "LIF"
        membrane_decay: LIF leak factor
    """
    print(f"Loading ANN checkpoint: {checkpoint_path}")

    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {"model": {"snn": {"enabled": True, "neuron_type": neuron_type}}}

    # Create model
    model = create_eldarin(config_dict=config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Convert activations to spiking neurons
    print(f"Converting activations to {neuron_type} neurons...")
    SNNConversionHelper.convert_relu_to_if(model, threshold=threshold)

    # Replace all IF/LIF params
    for name, module in model.named_modules():
        if isinstance(module, IFNeuron):
            module.membrane_decay = membrane_decay if neuron_type == "LIF" else 1.0

    # Save SNN model
    snn_checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "neuron_type": neuron_type,
        "threshold": threshold,
        "membrane_decay": membrane_decay,
        "ann_checkpoint": checkpoint_path,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(snn_checkpoint, output_path)
    print(f"SNN model saved to {output_path}")

    return model


def calibrate_snn(
    model: nn.Module,
    dataloader,
    percentile: float = 99.9,
    device: str = "cuda",
    num_steps: int = 100,
) -> nn.Module:
    """
    Calibrate SNN neuron thresholds to minimize conversion loss.

    Runs the model over calibration data for num_steps timesteps,
    collects activation statistics, and sets thresholds to the
    specified percentile of activation magnitude.
    """
    print(f"Calibrating SNN thresholds (percentile: {percentile}%)...")
    model = SNNConversionHelper.calibrate_thresholds(
        model, dataloader, percentile, device
    )
    print("Calibration complete.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Eldarin ANN → SNN Conversion")
    parser.add_argument("--checkpoint", type=str, required=True, help="ANN checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config YAML")
    parser.add_argument("--output", type=str, default="checkpoints/eldarin_snn.pth")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--neuron_type", type=str, default="LIF", choices=["IF", "LIF"])
    parser.add_argument("--membrane_decay", type=float, default=0.9)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--data_root", type=str, default=None, help="For calibration")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    model = convert_to_snn(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        threshold=args.threshold,
        neuron_type=args.neuron_type,
        membrane_decay=args.membrane_decay,
    )

    if args.calibrate and args.data_root:
        from utils.data_loader import create_dataloader
        calib_config = {"data": {"data_root": args.data_root, "max_samples": 100, "batch_size": 4}}
        loader = create_dataloader(calib_config, split="train")
        model = calibrate_snn(model, loader, device=args.device)
        torch.save({"model_state_dict": model.state_dict()}, args.output)
        print(f"Calibrated SNN saved to {args.output}")

    print("SNN conversion complete.")


if __name__ == "__main__":
    main()