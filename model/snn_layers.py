"""
SNN-Compatible Layer Definitions
==================================
Spiking Neural Network layers for FPGA deployment.
Supports conversion of ANN layers to SNN equivalents (IF/LIF neurons).

Compatible with:
  - snnTorch (https://github.com/jeshraghian/snntorch)
  - Lava (Intel Loihi) (https://github.com/lava-nc/lava)
  - Custom FPGA HLS implementation using bitwise operations

The VSA/HDC operations (binding, bundling) are naturally SNN-friendly
since they use bitwise/bipolar operations that map directly to spikes.

Original VioPose: https://github.com/SeongJong-Yoo/VioPose
FPGA Event Encode: https://github.com/Enotrium/FPGA-Event-Based-encode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class IFNeuron(nn.Module):
    """
    Integrate-and-Fire neuron for SNN conversion.
    Accumulates input and fires a spike when threshold is exceeded.

    Args:
        threshold: Firing threshold
        reset_mechanism: "subtract" or "zero"
        membrane_decay: Leak factor (< 1 for LIF, = 1 for IF)
    """

    def __init__(
        self,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        membrane_decay: float = 1.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.membrane_decay = membrane_decay
        self.register_buffer("membrane", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input current [B, ...]
        Returns:
            Spike output [B, ...] (binary)
        """
        if self.membrane is None or self.membrane.shape != x.shape:
            self.membrane = torch.zeros_like(x)

        # Leak
        self.membrane = self.membrane * self.membrane_decay

        # Integrate
        self.membrane = self.membrane + x

        # Fire
        spikes = (self.membrane >= self.threshold).float()

        # Reset
        if self.reset_mechanism == "subtract":
            self.membrane = self.membrane - spikes * self.threshold
        elif self.reset_mechanism == "zero":
            self.membrane = self.membrane * (1 - spikes)

        return spikes

    def reset_state(self):
        self.membrane = None


class LIFNeuron(IFNeuron):
    """Leaky Integrate-and-Fire neuron (default: membrane_decay < 1)."""

    def __init__(
        self,
        threshold: float = 1.0,
        membrane_decay: float = 0.9,
        reset_mechanism: str = "subtract",
    ):
        super().__init__(
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            membrane_decay=membrane_decay,
        )


class SurrogateGradientSpike(nn.Module):
    """
    Spike function with surrogate gradient for training.
    Uses fast sigmoid surrogate for backpropagation through the
    non-differentiable spike threshold.

    For use with snnTorch or custom SNN training.
    """

    def __init__(self, threshold: float = 1.0, slope: float = 10.0):
        super().__init__()
        self.threshold = threshold
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Surrogate: fast sigmoid
            surrogate = torch.sigmoid(self.slope * (x - self.threshold))
            # Straight-through estimator
            spike = (x > self.threshold).float()
            return spike + surrogate - surrogate.detach()
        else:
            return (x > self.threshold).float()


class SNNConv2d(nn.Module):
    """
    SNN-compatible 2D convolution layer.
    Wraps standard Conv2d with IF/LIF neuron and optional spike encoding.

    For FPGA deployment, this maps to:
      - Weight matrix in BRAM/DRAM
      - MAC operations → IF accumulation
      - Threshold comparator + spike output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        threshold: float = 1.0,
        neuron_type: str = "LIF",
        membrane_decay: float = 0.9,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if neuron_type == "LIF":
            self.neuron = LIFNeuron(threshold, membrane_decay)
        else:
            self.neuron = IFNeuron(threshold)

        self.spike_fn = SurrogateGradientSpike(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.neuron(x)
        return x

    def reset_state(self):
        self.neuron.reset_state()


class SNNLinear(nn.Module):
    """SNN-compatible linear layer with IF/LIF neuron."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 1.0,
        neuron_type: str = "LIF",
        membrane_decay: float = 0.9,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)

        if neuron_type == "LIF":
            self.neuron = LIFNeuron(threshold, membrane_decay)
        else:
            self.neuron = IFNeuron(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.neuron(x)
        return x

    def reset_state(self):
        self.neuron.reset_state()


class VSAHDCSpiking(nn.Module):
    """
    Spiking implementation of VSA/HDC operations.
    Maps binding, bundling, and permutation to spike-compatible
    binary operations that are efficient on neuromorphic hardware.

    Binding (⊗): Element-wise XNOR of binary vectors → maps to simple logic
    Bundling (⊕): Thresholded sum → population count + comparator
    Similarity: Hamming distance → XNOR + popcount

    Reference: https://github.com/Enotrium/arthedain-1
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.threshold = threshold

    def binary_bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binary binding via element-wise XNOR (spike-compatible)."""
        # For bipolar ±1: a * b
        # For binary 0/1: 1 - (a XOR b) = a == b
        return (a == b).float() * 2 - 1  # Convert to bipolar

    def threshold_bundle(
        self, vectors: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Threshold-based bundling (majority vote with spike threshold)."""
        if weights is not None:
            vectors = vectors * weights.unsqueeze(-1)
        summed = vectors.sum(dim=0)
        return (summed > self.threshold * vectors.shape[0]).float()

    def similarity_popcount(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Similarity via population count.
        For binary vectors: count matching bits / total bits.
        Maps to XNOR + popcount in hardware (very FPGA-efficient).
        """
        matches = (a == b).float()
        return matches.sum(dim=-1) / self.hd_dim

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        operation: str = "bind",
    ) -> torch.Tensor:
        if operation == "bind":
            return self.binary_bind(a, b)
        elif operation == "bundle":
            return self.threshold_bundle(a)
        elif operation == "similarity":
            return self.similarity_popcount(a, b)
        return a


class SNNConversionHelper:
    """
    Utility for converting standard ANN modules to SNN equivalents.

    Process:
    1. Train ANN normally
    2. Replace activation functions with IF/LIF neurons
    3. Calibrate thresholds using training data
    4. Export as spike-compatible model

    For FPGA deployment, the spike operations map directly to
    HLS-synthesizable logic (bitwise ops + accumulators).
    """

    @staticmethod
    def convert_relu_to_if(
        module: nn.Module,
        threshold: float = 1.0,
        timesteps: int = 100,
    ) -> nn.Module:
        """
        Recursively convert ReLU activations to IF neurons.

        Args:
            module: ANN module
            threshold: IF threshold
            timesteps: SNN inference timesteps

        Returns:
            SNN module with IF neurons
        """
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, IFNeuron(threshold=threshold))
            elif isinstance(child, nn.SiLU):
                setattr(module, name, IFNeuron(threshold=threshold))
            else:
                SNNConversionHelper.convert_relu_to_if(child, threshold, timesteps)
        return module

    @staticmethod
    def calibrate_thresholds(
        model: nn.Module,
        dataloader,
        percentile: float = 99.9,
        device: str = "cuda",
    ):
        """
        Calibrate IF neuron thresholds using training data statistics.
        Sets threshold to the (percentile)th activation value to
        minimize spike rate while preserving information.

        Args:
            model: SNN model with IF neurons
            dataloader: Data loader for calibration
            percentile: Activation percentile for threshold
            device: Compute device
        """
        model.eval()
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                activations[name].append(output.detach().cpu().flatten())
            return hook

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (IFNeuron, LIFNeuron)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    model(**inputs)
                else:
                    model(batch.to(device))
                if len(activations) > 100:  # Limit calibration batches
                    break

        for hook in hooks:
            hook.remove()

        # Set thresholds
        for name, module in model.named_modules():
            if isinstance(module, (IFNeuron, LIFNeuron)):
                if name in activations:
                    all_acts = torch.cat(activations[name])
                    module.threshold = torch.quantile(
                        all_acts, percentile / 100.0
                    ).item()

        return model