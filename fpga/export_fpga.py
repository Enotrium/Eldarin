"""
Eldarin FPGA Export Module
============================
Exports Eldarin model for FPGA deployment via ONNX, TensorRT, and HLS C++.

References:
  FPGA-Event-Based-encode: https://github.com/Enotrium/FPGA-Event-Based-encode
  
"""

import torch
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 640, 640),
    opset_version: int = 14,
) -> str:
    """Export model to ONNX format."""
    model.eval()
    model = model.cpu()

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["frames"],
        output_names=["detections", "hd_representation"],
        dynamic_axes={
            "frames": {0: "batch_size"},
            "detections": {0: "num_detections"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"ONNX exported to {output_path}")
    return output_path


def export_to_tensorrt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    workspace_size: int = 4096,  # MB
) -> str:
    """
    Export ONNX model to TensorRT engine.
    Requires tensorrt Python package.

    Args:
        onnx_path: Input ONNX model
        output_path: Output .engine file
        precision: "fp32", "fp16", or "int8"
        workspace_size: Workspace in MB
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not available. Install with: pip install nvidia-tensorrt")
        print("Skipping TensorRT export.")
        return ""

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        success = parser.parse(f.read())

    if not success:
        for i in range(parser.num_errors):
            print(f"  Error: {parser.get_error(i)}")
        raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size * 1024 * 1024

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision (calibration needed)")

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"TensorRT engine exported to {output_path}")
    return output_path


def generate_hls_config(
    model: torch.nn.Module,
    output_path: str = "fpga/hls_config.json",
    precision: int = 8,
    clock_mhz: int = 200,
    hd_dim_hw: int = 4096,
) -> Dict:
    """
    Generate HLS synthesis configuration for FPGA.

    Outputs JSON with:
      - Layer-by-layer parameters (weights shapes, bit widths)
      - VSA/HDC kernel parameters
      - Memory allocation estimates
      - Timing constraints
    """
    config = {
        "model": "Eldarin",
        "precision": f"int{precision}",
        "clock_mhz": clock_mhz,
        "target_fps": 60,
        "hd_dim": hd_dim_hw,
        "layers": [],
        "memory_estimate": {},
        "vsa_kernel": {
            "hd_dim": hd_dim_hw,
            "binding": "xor",
            "similarity": "hamming",
            "lut_estimate": hd_dim_hw * 2,  # XNOR + popcount
        },
    }

    # Analyze model layers
    total_params = 0
    total_activations = 0

    for name, module in model.named_modules():
        layer_info = {"name": name, "type": type(module).__name__}

        if isinstance(module, torch.nn.Conv2d):
            layer_info.update({
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "weight_shape": list(module.weight.shape),
                "bits": precision,
            })
            total_params += module.weight.numel()

        elif isinstance(module, torch.nn.Linear):
            layer_info.update({
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight_shape": list(module.weight.shape),
                "bits": precision,
            })
            total_params += module.weight.numel()

        elif isinstance(module, torch.nn.BatchNorm2d):
            layer_info.update({
                "num_features": module.num_features,
            })

        config["layers"].append(layer_info)

    # Memory estimate
    bytes_per_param = precision // 8
    weight_memory = total_params * bytes_per_param
    activation_memory = hd_dim_hw * 4  # Estimated

    config["memory_estimate"] = {
        "weights_bytes": weight_memory,
        "weights_mb": weight_memory / 1e6,
        "activation_bytes": activation_memory,
        "total_mb": (weight_memory + activation_memory) / 1e6,
        "dsp_estimate": total_params // 1000 * precision // 8,
        "bram_36kb": int(weight_memory / 4500),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"HLS config saved to {output_path}")
    print(f"  Total params: {total_params/1e6:.2f}M")
    print(f"  Weight memory: {weight_memory/1e6:.2f} MB")
    print(f"  Est. BRAM 36Kb blocks: {config['memory_estimate']['bram_36kb']}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Eldarin FPGA Export")
    parser.add_argument("--config", type=str, default="config/fpga_export.yaml", help="Config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="export/eldarin")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--export_onnx", action="store_true", help="Export ONNX")
    parser.add_argument("--export_tensorrt", action="store_true", help="Export TensorRT")
    parser.add_argument("--generate_hls", action="store_true", help="Generate HLS config")
    parser.add_argument("--hd_dim", type=int, default=4096, help="Hardware HD dim")

    args = parser.parse_args()

    # Load config
    cfg = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    fpga_cfg = cfg.get("fpga_export", {})
    args.precision = args.precision or fpga_cfg.get("precision", "fp16")
    args.hd_dim = args.hd_dim or fpga_cfg.get("hd_dim_hw", 4096)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from model.eldarin_model import create_eldarin
    model = create_eldarin(config_dict=cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    if args.export_onnx:
        export_to_onnx(model, f"{args.output}.onnx")

        if args.export_tensorrt:
            export_to_tensorrt(
                f"{args.output}.onnx",
                f"{args.output}.engine",
                precision=args.precision,
            )

    if args.generate_hls:
        generate_hls_config(
            model,
            f"{args.output}_hls.json",
            precision=8 if args.precision == "int8" else 16,
            hd_dim_hw=args.hd_dim,
        )

    print("FPGA export complete.")


if __name__ == "__main__":
    main()