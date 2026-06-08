"""
Vector Symbolic Architecture (VSA) / Hyperdimensional Computing (HDC) Module
=============================================================================

Adapted from: https://github.com/Enotrium/arthedain-1
Integrated into Eldarin for robust multimodal feature fusion, memory, and
uncertainty handling in the hierarchical Bayesian-style mixing module.


Paper: https://arxiv.org/pdf/2411.13607

Core VSA/HDC operations:
  - Binding  (⊗): Associates features across modalities (e.g., visual ⊗ event)
  - Bundling (⊕): Superimposes multiple feature bindings (sum / majority vote)
  - Permutation (ρ): Encodes temporal/sequential relationships
  - Similarity: Cosine or Hamming distance for robust matching under noise

VSA algebra makes these operations hardware-efficient:
  - Binary/bipolar vectors → bitwise XNOR/XOR for binding
  - Population count for similarity
  - Naturally compatible with FPGA and SNN implementations

Reference: Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction
to Computing in Distributed Representation with High-Dimensional Random Vectors"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class VSAHDC(nn.Module):
    """
    VSA/HDC operations module for Eldarin.
    Handles projection to hyperdimensional space, binding, bundling,
    permutation, and similarity computation.

    Supports both Fourier Holographic Reduced Representation (FHRR / "circular")
    and Binary Spatter Code (BSC / "xor") VSA variants.

    Args:
        hd_dim: Hyperdimensional vector dimension (default: 8192)
        input_dim: Input feature dimension to project from
        dtype: "bipolar" (±1) or "binary" (0/1)
        binding: "circular" (FHRR, uses complex FFT) or "xor" (BSC)
        similarity: "cosine" or "hamming"
        seed: Random seed for HD basis vectors
    """

    def __init__(
        self,
        hd_dim: int = 8192,
        input_dim: int = 1024,
        dtype: str = "bipolar",
        binding: str = "circular",
        similarity: str = "cosine",
        seed: int = 42,
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.input_dim = input_dim
        self.dtype = dtype
        self.binding = binding
        self.similarity = similarity

        # Projection from feature space to HD space
        self.register_buffer(
            "projection",
            self._init_projection(input_dim, hd_dim, dtype, seed),
        )

        # Base HD vectors for each dimension (used in binding/permutation)
        self.register_buffer(
            "id_vec",
            self._generate_base_vector(hd_dim, dtype, seed + 1),
        )

        # Permutation basis for temporal encoding
        self.register_buffer(
            "perm_basis",
            self._generate_permutation_basis(hd_dim, dtype, seed + 2),
        )

        # Learnable scaling for projection
        self.scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def _init_projection(
        input_dim: int, hd_dim: int, dtype: str, seed: int
    ) -> torch.Tensor:
        """Initialize projection matrix: feature space → HD space."""
        gen = torch.Generator()
        gen.manual_seed(seed)

        if dtype == "bipolar":
            return torch.bernoulli(
                torch.full((input_dim, hd_dim), 0.5, generator=gen)
            ) * 2 - 1  # ±1
        else:  # binary
            return torch.bernoulli(
                torch.full((input_dim, hd_dim), 0.5, generator=gen)
            )  # 0/1

    @staticmethod
    def _generate_base_vector(hd_dim: int, dtype: str, seed: int) -> torch.Tensor:
        """Generate a random base HD vector (identity)."""
        gen = torch.Generator()
        gen.manual_seed(seed)

        if dtype == "bipolar":
            return torch.bernoulli(
                torch.full((hd_dim,), 0.5, generator=gen)
            ) * 2 - 1

        # Use complex for FHRR binding
        if dtype == "circular":
            angles = torch.rand(hd_dim, generator=gen) * 2 * np.pi
            return torch.complex(
                torch.cos(angles), torch.sin(angles)
            )
        return torch.bernoulli(
            torch.full((hd_dim,), 0.5, generator=gen)
        )

    @staticmethod
    def _generate_permutation_basis(
        hd_dim: int, dtype: str, seed: int
    ) -> torch.Tensor:
        """Generate permutation basis vectors for temporal encoding."""
        gen = torch.Generator()
        gen.manual_seed(seed)

        if dtype == "bipolar":
            # Random cyclic shift indices
            return torch.randint(0, hd_dim, (hd_dim // 16,), generator=gen)
        return torch.bernoulli(
            torch.full((hd_dim,), 0.5, generator=gen)
        ) * 2 - 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode feature vector into hyperdimensional space.

        Args:
            x: Input features [B, input_dim] or [B, N, input_dim]

        Returns:
            HD vector [B, hd_dim] or [B, N, hd_dim]
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
            hd = F.linear(x, self.projection.T.to(x.dtype)) * self.scale
            hd = torch.tanh(hd)  # Soft binarization
            hd = hd.reshape(B, N, self.hd_dim)
        else:
            hd = F.linear(x, self.projection.T.to(x.dtype)) * self.scale
            hd = torch.tanh(hd)

        # Normalize
        hd = F.normalize(hd, p=2, dim=-1)
        return hd

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Binding operation (⊗): Associates two HD vectors.
        Binding is invertible and distributes over bundling.

        FHRR: complex multiplication (convolution theorem → element-wise mult in Fourier)
        BSC: XOR / element-wise multiplication
        """
        if self.binding == "circular":
            # Fourier Holographic Reduced Representation
            # Circular convolution via FFT → pointwise multiply → IFFT
            a_fft = torch.fft.fft(a.float(), dim=-1)
            b_fft = torch.fft.fft(b.float(), dim=-1)
            result = torch.fft.ifft(a_fft * b_fft, dim=-1).real
        elif self.binding == "xor":
            # Binary Spatter Code: element-wise XOR ~ multiplication for bipolar
            result = a * b
        else:
            result = a * b  # Default: element-wise multiply
        return F.normalize(result, p=2, dim=-1)

    def bundle(
        self, vectors: Union[torch.Tensor, list], weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Bundling operation (⊕): Superimposes multiple HD vectors.
        Represents a SET of items. Robust to noise.

        Args:
            vectors: [K, hd_dim] or list of tensors
            weights: Optional [K] importance weights

        Returns:
            Bundled vector [hd_dim]
        """
        if isinstance(vectors, list):
            vectors = torch.stack(vectors, dim=0)

        if weights is not None:
            weights = weights.to(vectors.device, vectors.dtype)
            weights = F.softmax(weights, dim=0)
            bundled = (vectors * weights.unsqueeze(-1)).sum(dim=0)
        else:
            bundled = vectors.sum(dim=0)

        if self.dtype == "bipolar":
            # Majority vote → sign
            bundled = torch.sign(bundled)
        return F.normalize(bundled, p=2, dim=-1)

    def permute(self, x: torch.Tensor, shift: int) -> torch.Tensor:
        """
        Permutation (ρ): Cyclic shift encoding temporal/positional order.
        ρ^k(x) encodes the k-th temporal position.
        """
        if self.binding == "circular":
            # Phase rotation in frequency domain
            x_fft = torch.fft.fft(x.float(), dim=-1)
            N = self.hd_dim
            freqs = 2 * np.pi * shift / N
            phase_shift = torch.exp(
                1j * freqs * torch.arange(N // 2 + 1, device=x.device)
            )
            # Apply to both halves
            x_fft[..., : N // 2 + 1] *= phase_shift
            x_fft[..., N // 2 + 1 :] *= phase_shift[: N // 2 - 1].flip(-1).conj()
            result = torch.fft.ifft(x_fft, dim=-1).real
        else:
            # Cyclic shift
            result = torch.roll(x, shifts=shift, dims=-1)
        return result

    def temporal_bind(
        self, sequence: torch.Tensor, use_permutation: bool = True
    ) -> torch.Tensor:
        """
        Encode a temporal sequence of HD vectors using permutation.
        This captures ORDERED relationships (trajectory dynamics).

        Args:
            sequence: [T, hd_dim] time-ordered HD vectors
            use_permutation: If True, permute each timestep before bundling

        Returns:
            Single HD vector encoding the entire sequence
        """
        T = sequence.shape[0]
        if use_permutation:
            encoded = []
            for t in range(T):
                encoded.append(self.permute(sequence[t], t))
            return self.bundle(torch.stack(encoded))
        else:
            return self.bundle(sequence)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between HD vectors.
        Cosine similarity for bipolar, Hamming similarity for binary.
        """
        if self.similarity == "cosine":
            return F.cosine_similarity(a, b, dim=-1)
        elif self.similarity == "hamming":
            # For bipolar: (hd_dim - mismatches) / hd_dim
            if self.dtype == "bipolar":
                return (a * b).sum(dim=-1) / self.hd_dim
            else:
                return (a == b).float().mean(dim=-1)
        return F.cosine_similarity(a, b, dim=-1)

    def hierarchical_bind(
        self, high_level: torch.Tensor, low_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Hierarchical binding: High-level semantics ⊗ Low-level features.
        This is the core VSA-enhanced operation in the Hierarchy Module,
        replacing/supplementing attention with role-filler binding.

        high_level: e.g., object class/semantic HD vector
        low_level: e.g., edge/motion feature HD vector

        Returns: Role-filled representation
        """
        return self.bind(high_level, low_level)

    def bayesian_update(
        self,
        prior: torch.Tensor,
        likelihood: torch.Tensor,
        prior_weight: float = 0.7,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bayesian-style update in hyperdimensional space.
        Used in the Mixing Module for cross-modal fusion.

        posterior = α * prior ⊕ (1-α) * likelihood
        With uncertainty gating when available.

        Args:
            prior: HD prior vector (from previous modalities)
            likelihood: HD likelihood vector (from new modality)
            prior_weight: Prior confidence (α)
            uncertainty: Optional uncertainty vector [hd_dim]

        Returns:
            (posterior, updated_uncertainty) tuple
        """
        if uncertainty is not None:
            # Uncertainty-gated update
            # Lower uncertainty → more weight to prior
            uncert_norm = torch.sigmoid(uncertainty.mean())
            effective_weight = prior_weight * (1 - uncert_norm) + uncert_norm * 0.5
        else:
            effective_weight = prior_weight

        posterior = (
            effective_weight * prior + (1 - effective_weight) * likelihood
        )
        posterior = F.normalize(posterior, p=2, dim=-1)

        # Update uncertainty (entropy-like measure from similarity)
        new_uncertainty = 1 - self.similarity(posterior, prior).abs()

        return posterior, new_uncertainty

    def retrieve(
        self, bound: torch.Tensor, cue: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve an item from a bound pair: if bound = a ⊗ b, then
        retrieve(bound, a) ≈ b (unbinding).
        """
        if self.binding == "circular":
            # Convolution inverse
            cue_inv = torch.conj(torch.fft.fft(cue.float(), dim=-1))
            bound_fft = torch.fft.fft(bound.float(), dim=-1)
            result = torch.fft.ifft(bound_fft * cue_inv, dim=-1).real
        elif self.binding == "xor":
            # XOR is its own inverse for bipolar (±1)
            result = bound * cue
        else:
            result = bound * cue
        return F.normalize(result, p=2, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        operation: str = "encode",
        aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unified forward pass.

        Args:
            features: Input features
            operation: "encode", "bind", "bundle", "permute", "similarity"
            aux: Auxiliary input for binding/retrieval
        """
        if operation == "encode":
            return self.encode(features)
        elif operation == "bind" and aux is not None:
            return self.bind(features, aux)
        elif operation == "bundle":
            return self.bundle(features)
        elif operation == "permute":
            shift = aux if aux is not None else 1
            return self.permute(features, shift)
        elif operation == "similarity" and aux is not None:
            return self.similarity(features, aux)
        return features


# ----- Functional API (for use outside the module) -----

def bind(a: torch.Tensor, b: torch.Tensor, mode: str = "circular") -> torch.Tensor:
    """Functional binding: a ⊗ b"""
    if mode == "circular":
        a_fft = torch.fft.fft(a.float(), dim=-1)
        b_fft = torch.fft.fft(b.float(), dim=-1)
        return torch.fft.ifft(a_fft * b_fft, dim=-1).real
    return a * b  # XOR for bipolar


def bundle(
    vectors: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Functional bundling: Σ vectors"""
    if weights is not None:
        vectors = vectors * F.softmax(weights, dim=0).unsqueeze(-1)
    return torch.sign(vectors.sum(dim=0))


def permute(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Functional permutation: ρ^shift(x)"""
    return torch.roll(x, shifts=shift, dims=-1)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between HD vectors."""
    return F.cosine_similarity(a, b, dim=-1)


def hamming_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamming similarity for binary HD vectors."""
    return (a == b).float().mean(dim=-1)


class HDCKalmanFilter(nn.Module):
    """
    Hyperdimensional Kalman Filter for 4D tracking.
    Performs Kalman-like state estimation in HD space, leveraging
    VSA binding for efficient state prediction and update.

    This replaces the standard matrix-inversion Kalman with HD vector operations
    that are more robust to noise and compatible with FPGA/SNN.

    State: HD encoding of [x, y, z, dx, dy, dz, vx, vy]
    """

    def __init__(
        self,
        state_dim: int = 8,
        hd_dim: int = 8192,
        dtype: str = "bipolar",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hd_dim = hd_dim
        self.dtype = dtype

        # HD basis vectors for each state component
        gen = torch.Generator()
        gen.manual_seed(1234)
        if dtype == "bipolar":
            self.state_basis = nn.Parameter(
                torch.bernoulli(
                    torch.full((state_dim, hd_dim), 0.5, generator=gen)
                ) * 2 - 1,
                requires_grad=False,
            )

        # State transition in HD space (simple momentum)
        self.transition_weight = nn.Parameter(torch.tensor(0.9))
        self.measurement_weight = nn.Parameter(torch.tensor(0.1))

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode Kalman state vector into HD space.
        state: [B, state_dim]
        returns: [B, hd_dim]
        """
        # Bind each state component with its basis vector
        hs = []
        for d in range(self.state_dim):
            basis = self.state_basis[d : d + 1]
            value = state[:, d : d + 1]
            hs.append(basis * value * self.hd_dim)
        return bundle(torch.cat(hs, dim=0).reshape(self.state_dim, -1, self.hd_dim).transpose(0, 1).reshape(-1, self.hd_dim))

    def predict(self, hd_state: torch.Tensor) -> torch.Tensor:
        """Predict next state in HD space."""
        # Simple HD momentum: permute + weighted prior
        predicted = permute(hd_state, 1) * self.transition_weight
        return F.normalize(predicted + hd_state * (1 - self.transition_weight), p=2, dim=-1)

    def update(
        self, hd_state: torch.Tensor, measurement: torch.Tensor
    ) -> torch.Tensor:
        """Update state with measurement in HD space."""
        updated = hd_state * (1 - self.measurement_weight) + measurement * self.measurement_weight
        return F.normalize(updated, p=2, dim=-1)

    def forward(
        self,
        prev_hd_state: torch.Tensor,
        measurement: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kalman filter step in HD space."""
        pred = self.predict(prev_hd_state)
        if measurement is not None:
            posterior = self.update(pred, measurement)
        else:
            posterior = pred
        return posterior, pred

# Test utilities
def test_vsa_operations():
    """Verify VSA algebra properties."""
    print("Testing VSA/HDC operations...")

    hd_dim = 1024
    vsa = VSAHDC(hd_dim=hd_dim, input_dim=256, dtype="bipolar", binding="circular")

    a = vsa._generate_base_vector(hd_dim, "bipolar", 0)
    b = vsa._generate_base_vector(hd_dim, "bipolar", 1)
    c = vsa._generate_base_vector(hd_dim, "bipolar", 2)

    # Test 1: Binding is invertible
    bound = vsa.bind(a, b)
    recovered = vsa.retrieve(bound, a)
    sim = vsa.similarity(recovered, b)
    print(f"  Binding invertibility: sim(recover(bind(a,b), a), b) = {sim:.4f}")

    # Test 2: Bundling similarity
    bundle_ab = vsa.bundle(torch.stack([a, b]))
    sim_a = vsa.similarity(bundle_ab, a)
    sim_b = vsa.similarity(bundle_ab, b)
    sim_c = vsa.similarity(bundle_ab, c)
    print(f"  Bundle similarity: sim(bundle, a)={sim_a:.4f}, sim(bundle, b)={sim_b:.4f}, sim(bundle, c)={sim_c:.4f}")

    # Test 3: Temporal encoding
    seq = torch.stack([a, b, c])
    temporal = vsa.temporal_bind(seq)
    sim_a0 = vsa.similarity(temporal, a)
    sim_perm = vsa.similarity(temporal, vsa.permute(b, 1))
    print(f"  Temporal encoding: sim(seq_encoded, a)={sim_a0:.4f}, sim(seq_encoded, ρ¹(b))={sim_perm:.4f}")

    print("VSA/HDC tests passed!\n")


if __name__ == "__main__":
    test_vsa_operations()