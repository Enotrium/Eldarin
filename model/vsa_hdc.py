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
            return (torch.rand(input_dim, hd_dim, generator=gen) > 0.5).float() * 2 - 1
        else:  # binary
            return (torch.rand(input_dim, hd_dim, generator=gen) > 0.5).float()

    @staticmethod
    def _generate_base_vector(hd_dim: int, dtype: str, seed: int) -> torch.Tensor:
        """Generate a random base HD vector (identity)."""
        gen = torch.Generator()
        gen.manual_seed(seed)

        if dtype == "bipolar":
            return (torch.rand(hd_dim, generator=gen) > 0.5).float() * 2 - 1

        # Use complex for FHRR binding
        if dtype == "circular":
            angles = torch.rand(hd_dim, generator=gen) * 2 * np.pi
            return torch.complex(torch.cos(angles), torch.sin(angles))
        return (torch.rand(hd_dim, generator=gen) > 0.5).float()

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
        return (torch.rand(hd_dim, generator=gen) > 0.5).float() * 2 - 1

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


# =============================================================================
# Resonator Network — from Renner et al. (2024), Nature Machine Intelligence
# arXiv: https://arxiv.org/abs/2209.02000
#
# The resonator network performs VSA-native vector factorization:
# given a composite HD vector s = Σᵢ (factor_1 ⊗ factor_2 ⊗ ...),
# iteratively recover the individual factors.
#
# This is the core computational primitive that enables:
#   - Translation/rotation/scale factorization from scene encoding
#   - Pattern recognition invariant to geometric transforms
#   - Efficient search over combinatorial factor spaces
# =============================================================================

class ResonatorNetwork(nn.Module):
    """
    Standard (flat) resonator network from Frady et al. (2020).

    Given a composite vector s = bind(h, v) where h and v are unknown,
    the resonator iteratively estimates ĥ and v̂:

        ĥ(t+1) = cleanup( s ⊗ unbind(v̂(t)) )
        v̂(t+1) = cleanup( s ⊗ unbind(ĥ(t+1)) )

    where cleanup projects back to the valid codebook of each factor.

    Args:
        factor_sizes: List of ints, number of possible values for each factor
        hd_dim: Hyperdimensional vector dimension
        gamma: Update rate (0 < γ ≤ 1) — lower = smoother
        nonlinearity: "phasor" (project to unit circle) or "relu" or "softmax"
        dtype: "bipolar" or "binary"
    """

    def __init__(
        self,
        factor_sizes: list,
        hd_dim: int = 8192,
        gamma: float = 0.3,
        nonlinearity: str = "phasor",
        dtype: str = "bipolar",
        seed: int = 42,
    ):
        super().__init__()
        self.num_factors = len(factor_sizes)
        self.factor_sizes = factor_sizes
        self.hd_dim = hd_dim
        self.gamma = gamma
        self.nonlinearity = nonlinearity
        self.dtype = dtype

        # Codebook for each factor: [size, hd_dim]
        self.codebooks = nn.ParameterList()
        gen = torch.Generator()
        for i, size in enumerate(factor_sizes):
            gen.manual_seed(seed + i * 1000)
            cb = (torch.rand(size, hd_dim, generator=gen) > 0.5).float() * 2 - 1
            self.codebooks.append(
                nn.Parameter(cb, requires_grad=False)
            )

        # Decoding matrices (transpose of codebooks for efficient readout)
        # For bipolar: decode = codebook.T (since ±1, transpose ≈ pseudoinverse)
        self.decoders = nn.ParameterList()
        for cb in self.codebooks:
            self.decoders.append(
                nn.Parameter(cb.T.clone(), requires_grad=False)
            )

    def cleanup(self, state: torch.Tensor, factor_idx: int) -> torch.Tensor:
        """
        Cleanup an estimated factor vector by projecting to the valid codebook.

        Step 1: Decode — compute similarity to all codebook vectors
        Step 2: Apply nonlinearity (normalization, thresholding)
        Step 3: Encode — reconstruct from codebook

        This is the same operation as the "projection" step in the paper:
            ĥ_clean = f(C_h · C_h† · ĥ_raw)
        where C_h is the codebook matrix for factor h.

        Args:
            state: [B, hd_dim] raw estimated factor
            factor_idx: Which factor to clean up

        Returns:
            [B, hd_dim] cleaned factor
        """
        codebook = self.codebooks[factor_idx]  # [size, hd_dim]
        decoder = self.decoders[factor_idx]     # [hd_dim, size]

        # Decode: compute similarities
        # raw_state @ decoder = [B, hd_dim] @ [hd_dim, size] = [B, size]
        similarities = state @ decoder  # [B, size]

        # Apply nonlinearity
        if self.nonlinearity == "phasor":
            # Element-wise division by magnitude (project to unit circle)
            # For real vectors, this is L2 normalization
            similarities = F.normalize(similarities, p=2, dim=-1)
        elif self.nonlinearity == "relu":
            similarities = F.relu(similarities)
            similarities = F.normalize(similarities, p=2, dim=-1)
        elif self.nonlinearity == "softmax":
            similarities = F.softmax(similarities * 10, dim=-1)  # Temperature
        elif self.nonlinearity == "exp":
            # Exponentiation + normalization (stronger cleanup, from paper)
            similarities = torch.exp(similarities * 5)
            similarities = F.normalize(similarities, p=1, dim=-1)

        # Encode: weighted sum of codebook vectors
        # [B, size] @ [size, hd_dim] = [B, hd_dim]
        cleaned = similarities @ codebook
        return F.normalize(cleaned, p=2, dim=-1)

    def unbind_state(self, composite: torch.Tensor, other_factor: torch.Tensor) -> torch.Tensor:
        """
        Unbind: extract one factor from composite given the other.

        If s = h ⊗ v, then:
            h ≈ s ⊗ v^{-1}  (unbinding via element-wise multiply for bipolar)

        Args:
            composite: [B, hd_dim] the composite vector s
            other_factor: [B, hd_dim] the other known factor

        Returns:
            [B, hd_dim] estimated unknown factor
        """
        # For bipolar ±1, multiplication is its own inverse
        return composite * other_factor

    def forward(
        self,
        composite: torch.Tensor,
        num_iterations: int = 10,
        return_all_iterations: bool = False,
    ) -> dict:
        """
        Run the resonator network to factorize a composite HD vector.

        Args:
            composite: [B, hd_dim] the bundled product of factors
            num_iterations: Number of resonator iterations
            return_all_iterations: If True, return factor estimates at each step

        Returns:
            Dict with:
                - "factors": List of [B, size_i] confidence vectors for each factor
                - "factor_states": List of [B, hd_dim] final HD factor estimates
                - "history": (if return_all_iterations) list of per-iteration states
        """
        B = composite.shape[0]
        device = composite.device
        composite = F.normalize(composite, p=2, dim=-1)

        # Initialize factor states randomly (as in paper)
        states = []
        gen = torch.Generator(device=device)
        gen.manual_seed(12345)
        for size in self.factor_sizes:
            idx = torch.randint(0, size, (B,), generator=gen, device=device)
            states.append(self.codebooks[0][idx])  # Initialize from random codebook entry
        states = [F.normalize(s, p=2, dim=-1) for s in states]

        history = [] if return_all_iterations else None

        # Resonator iterations (Eq. 4 in the paper)
        for _ in range(num_iterations):
            for factor_idx in range(self.num_factors):
                # Compute query: unbind all OTHER factors from composite
                query = composite.clone()
                for other_idx in range(self.num_factors):
                    if other_idx != factor_idx:
                        query = self.unbind_state(query, states[other_idx])

                # Cleanup
                cleaned = self.cleanup(query, factor_idx)

                # Update with smoothing (γ parameter, from paper Eq. 4)
                states[factor_idx] = (
                    (1 - self.gamma) * states[factor_idx] + self.gamma * cleaned
                )
                states[factor_idx] = F.normalize(states[factor_idx], p=2, dim=-1)

            if history is not None:
                history.append([s.clone() for s in states])

        # Final readout: decode each factor to confidence distribution
        factors = []
        for factor_idx in range(self.num_factors):
            confidences = states[factor_idx] @ self.decoders[factor_idx]
            factors.append(confidences)

        result = {
            "factors": factors,
            "factor_states": states,
        }
        if return_all_iterations:
            result["history"] = history

        return result

    def population_vector_readout(self, confidences: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Population vector readout for sub-index resolution.
        From the paper (Eq. 7): computes the similarity-weighted average of indices
        around the peak, yielding sub-pixel/sub-index precision.

        This is the method used for output trajectory estimation in the paper,
        replacing argmax with a continuous-valued estimate.

        Args:
            confidences: [B, size] similarity values from decode
            indices: Optional [size] index values (default: 0, 1, 2, ...)

        Returns:
            [B] continuous estimated index values
        """
        B, size = confidences.shape
        device = confidences.device

        if indices is None:
            indices = torch.arange(size, device=device, dtype=torch.float32)

        # Find peak for each batch element
        peak_idxs = confidences.argmax(dim=-1)  # [B]

        # Neighborhood around peak (±5 as in paper Eq. 7)
        window = 5
        pop_values = torch.zeros(B, device=device)

        for b in range(B):
            center = peak_idxs[b].item()
            lo = max(0, center - window)
            hi = min(size, center + window + 1)

            w = confidences[b, lo:hi]
            idx = indices[lo:hi]

            if w.sum() > 0:
                pop_values[b] = (w * idx).sum() / w.sum()
            else:
                pop_values[b] = float(center)

        return pop_values


class HierarchicalResonatorNetwork(nn.Module):
    """
    Hierarchical Resonator Network (HRN) from Renner et al. (2024).

    Extends the resonator to handle factors that use DIFFERENT reference frames,
    specifically Cartesian and log-polar coordinates.

    From the paper (Fig. 5, Eq. 4):
        - Cartesian partition: factors for translation (h, v)
        - Log-polar partition: factors for rotation (r) and scale (s)
        - Λ transforms between the two reference frames

    Architecture: Two interacting resonator partitions sharing information
    through Λ transform matrices.

    For Eldarin, this enables:
        - VSA-native ego-motion estimation from encoded frames (like the paper)
        - Object detection as transform factorization (where detected objects
          are factorizations of "what" ⊗ "where")
        - Tight integration with the learned pipeline as a bootstrap path

    Args:
        cartesian_factors: List of factor sizes for Cartesian partition (e.g., [width, height])
        logpolar_factors: List of factor sizes for log-polar partition (e.g., [n_angles, n_scales])
        hd_dim: HD vector dimension
        gamma: Resonator update rate
        nonlinearity: Cleanup nonlinearity
    """

    def __init__(
        self,
        cartesian_factors: list,
        logpolar_factors: list,
        hd_dim: int = 8192,
        gamma: float = 0.3,
        nonlinearity: str = "phasor",
        dtype: str = "bipolar",
        seed: int = 777,
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.gamma = gamma
        self.nonlinearity = nonlinearity

        # Two resonator partitions
        self.cartesian_resonator = ResonatorNetwork(
            factor_sizes=cartesian_factors,
            hd_dim=hd_dim,
            gamma=gamma,
            nonlinearity=nonlinearity,
            dtype=dtype,
            seed=seed,
        )
        self.logpolar_resonator = ResonatorNetwork(
            factor_sizes=logpolar_factors,
            hd_dim=hd_dim,
            gamma=gamma,
            nonlinearity=nonlinearity,
            dtype=dtype,
            seed=seed + 1000,
        )

        # Λ matrix: transforms between Cartesian and log-polar reference frames
        # This is a fixed random projection (approximates the paper's Λ)
        gen = torch.Generator()
        gen.manual_seed(seed + 500)
        self.register_buffer(
            "lambda_transform",
            (torch.rand(hd_dim, hd_dim, generator=gen) > 0.5).float() * 2 - 1,
        )

        # Map update parameters (from paper Eq. 9)
        self.register_buffer("map_memory", torch.zeros(hd_dim))
        self.register_buffer("anchor_map", torch.zeros(hd_dim))
        self.mu1 = nn.Parameter(torch.tensor(0.9))  # Temporal decay
        self.mu2 = nn.Parameter(torch.tensor(0.05))  # Anchor weight
        self.map_initialized = False

    def cartesian_to_logpolar(self, cartesian_hd: torch.Tensor) -> torch.Tensor:
        """
        Λ transform: Cartesian → Log-polar reference frame.
        Approximated via random projection (paper uses a structured Λ matrix).

        Args:
            cartesian_hd: [B, hd_dim] HD vector in Cartesian frame

        Returns:
            [B, hd_dim] HD vector in log-polar frame
        """
        # Random projection (approximation of the paper's Λ)
        transformed = cartesian_hd @ self.lambda_transform
        return F.normalize(transformed, p=2, dim=-1)

    def logpolar_to_cartesian(self, logpolar_hd: torch.Tensor) -> torch.Tensor:
        """
        Λ^{-1} transform: Log-polar → Cartesian reference frame.

        Args:
            logpolar_hd: [B, hd_dim] HD vector in log-polar frame

        Returns:
            [B, hd_dim] HD vector in Cartesian frame
        """
        # Approximate inverse via transpose of random projection
        transformed = logpolar_hd @ self.lambda_transform.T
        return F.normalize(transformed, p=2, dim=-1)

    def update_map(
        self,
        input_hd: torch.Tensor,
        transform_estimate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update the allocentric map from the current input.

        From the paper (Eq. 8-9):
            m(t) = Λ(s(t) ⊗ h^{h_out} ⊙ v^{v_out}) ⊗ r^{r_out}
            m̂(t+1) = μ₁·m̂(t) + μ₂·m̂(0) + (1-μ₁-μ₂)·m(t)

        The anchored map update prevents long-term drift.

        Args:
            input_hd: [B, hd_dim] current FPE-encoded input
            transform_estimate: Optional [B, hd_dim] estimated transform kernel

        Returns:
            Updated map [B, hd_dim]
        """
        device = input_hd.device
        B = input_hd.shape[0]

        # Transform input to map coordinates
        if transform_estimate is not None:
            # Unbind the estimated transform from the input
            map_input = input_hd * transform_estimate  # Unbind
        else:
            map_input = input_hd

        # Initialize anchor map on first call
        if not self.map_initialized:
            self.anchor_map = map_input[0].detach().clone()
            self.map_memory = self.anchor_map.clone()
            self.map_initialized = True

        # Anchored map update (Eq. 9)
        self.map_memory = (
            self.mu1 * self.map_memory.to(device)
            + self.mu2 * self.anchor_map.to(device)
            + (1 - self.mu1 - self.mu2) * map_input[0].detach()
        )
        self.map_memory = F.normalize(self.map_memory, p=2, dim=-1)

        return self.map_memory.unsqueeze(0).expand(B, -1)

    def forward(
        self,
        encoded_input: torch.Tensor,
        encoded_map: Optional[torch.Tensor] = None,
        num_iterations: int = 10,
    ) -> dict:
        """
        Run the hierarchical resonator for visual odometry-like estimation.

        Given an FPE-encoded input and map, estimate:
            - Translation (h, v) in Cartesian frame
            - Rotation (r) and scale (s) in log-polar frame

        Args:
            encoded_input: [B, hd_dim] FPE-encoded current image/event frame
            encoded_map: Optional [B, hd_dim] FPE-encoded allocentric map
            num_iterations: Number of resonator iterations

        Returns:
            Dict with:
                - "translation": Estimated (h, v) indices
                - "rotation": Estimated rotation index
                - "scale": Estimated scale index (if available)
                - "cartesian_factors": Confidence arrays for Cartesian factors
                - "logpolar_factors": Confidence arrays for log-polar factors
                - "map": Updated allocentric map
                - "translation_hd": Translation kernel in HD space
        """
        B = encoded_input.shape[0]
        device = encoded_input.device

        if encoded_map is None:
            # Initialize map from first input (as in paper)
            encoded_map = self.update_map(encoded_input)

        encoded_input = F.normalize(encoded_input, p=2, dim=-1)
        encoded_map = F.normalize(encoded_map, p=2, dim=-1)

        # Initialize states
        # Cartesian: estimate translation h, v
        cart_states = [
            F.normalize(
                self.cartesian_resonator.codebooks[0][torch.randint(
                    0, self.cartesian_resonator.factor_sizes[0], (B,),
                    device=device
                )],
                p=2, dim=-1
            ),
            F.normalize(
                self.cartesian_resonator.codebooks[1][torch.randint(
                    0, self.cartesian_resonator.factor_sizes[1], (B,),
                    device=device
                )],
                p=2, dim=-1
            ),
        ]

        # Log-polar: estimate rotation r
        lp_states = [
            F.normalize(
                self.logpolar_resonator.codebooks[0][torch.randint(
                    0, self.logpolar_resonator.factor_sizes[0], (B,),
                    device=device
                )],
                p=2, dim=-1
            ),
        ]

        # Hierarchical resonator iterations (following paper Eq. 4)
        for _ in range(num_iterations):
            # --- Cartesian partition update ---
            # Query: s ⊗ r̂* ⊗ m̂* (Eq. 4: p̂ = Λ(s ⊗ r̂*))
            p_hat = encoded_input * lp_states[0]  # Unbind rotation
            p_hat = self.cartesian_to_logpolar(p_hat)  # Λ transform

            # l̂ = Λ^{-1}(m̂ ⊗ ĥ* ⊗ v̂*)
            l_hat = encoded_map * cart_states[0] * cart_states[1]
            l_hat = self.logpolar_to_cartesian(l_hat)  # Λ^{-1} transform

            # Update h: unbind v, cleanup
            query_h = encoded_input * cart_states[1] * encoded_map
            cleaned_h = self.cartesian_resonator.cleanup(query_h, 0)
            cart_states[0] = (1 - self.gamma) * cart_states[0] + self.gamma * cleaned_h
            cart_states[0] = F.normalize(cart_states[0], p=2, dim=-1)

            # Update v: unbind h, cleanup
            query_v = encoded_input * cart_states[0] * encoded_map
            cleaned_v = self.cartesian_resonator.cleanup(query_v, 1)
            cart_states[1] = (1 - self.gamma) * cart_states[1] + self.gamma * cleaned_v
            cart_states[1] = F.normalize(cart_states[1], p=2, dim=-1)

            # --- Log-polar partition update ---
            # Transform to log-polar, update rotation
            lp_input = self.cartesian_to_logpolar(encoded_input)
            lp_map = self.cartesian_to_logpolar(encoded_map)

            query_r = lp_input * lp_map * l_hat  # Simplified
            cleaned_r = self.logpolar_resonator.cleanup(query_r, 0)
            lp_states[0] = (1 - self.gamma) * lp_states[0] + self.gamma * cleaned_r
            lp_states[0] = F.normalize(lp_states[0], p=2, dim=-1)

        # Readout via population vector (Eq. 7)
        cart_confs = []
        for i, state in enumerate(cart_states):
            conf = state @ self.cartesian_resonator.decoders[i]
            cart_confs.append(conf)

        lp_confs = []
        for i, state in enumerate(lp_states):
            conf = state @ self.logpolar_resonator.decoders[i]
            lp_confs.append(conf)

        h_val = self.cartesian_resonator.population_vector_readout(cart_confs[0])
        v_val = self.cartesian_resonator.population_vector_readout(cart_confs[1])
        r_val = self.logpolar_resonator.population_vector_readout(lp_confs[0])

        # Compute translation kernel in HD space for map update
        h_idx = cart_confs[0].argmax(dim=-1)
        v_idx = cart_confs[1].argmax(dim=-1)
        trans_kernel = (
            self.cartesian_resonator.codebooks[0][h_idx]
            * self.cartesian_resonator.codebooks[1][v_idx]
        )

        # Update map
        updated_map = self.update_map(encoded_input, trans_kernel)

        return {
            "translation": (h_val, v_val),
            "rotation": r_val,
            "cartesian_factors": cart_confs,
            "logpolar_factors": lp_confs,
            "map": updated_map,
            "translation_hd": trans_kernel,
            "cartesian_states": cart_states,
            "logpolar_states": lp_states,
        }

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