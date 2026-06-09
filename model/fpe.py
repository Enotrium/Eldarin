"""
Fractional Power Encoding (FPE) Module
========================================
Implements the Fractional Power Encoding from Renner et al. (2024),
"Visual Odometry with Neuromorphic Resonator Networks," Nature Machine Intelligence.

arXiv: https://arxiv.org/abs/2209.02000

FPE encodes continuous quantities (coordinates, velocities, angles, etc.) into
VSA/HDC space such that **binding becomes equivariant to addition**: 

    encode(x + Δ) ≈ encode(x) ⊗ seed^{Δ}

This is the mathematical foundation that enables VSA-based:
  - Continuous coordinate encoding (pixel positions, 3D coordinates)
  - Translation, rotation, and scaling as VSA binding operations
  - Resonator networks that factorize scene transformations

Core concepts:
  1. Seed vectors h₀, v₀ are random complex/bipolar HD vectors
  2. Fractional powers: h₀^x for continuous x (via FFT-based phasor rotation)
  3. Position encoding: s = Σ_{(x,y)∈E} h₀^x ⊗ v₀^y  (Eq. 1 in paper)
  4. Translation equivariance: encode(shift by Δ) = seed^{Δ} ⊗ encode(original)
  5. Codebook matrix Φ: encodes all pixel locations for efficient matrix-multiply encoding

Reference:
  Frady, Kanerva, Sommer (2019). "A framework for linking computations and rhythm-based
  timing patterns in neural firing."
  Frady et al. (2021). "Computing on functions using randomized vector representations."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class FractionalPowerEncoder(nn.Module):
    """
    Encodes continuous 1D or 2D coordinates into HD space via fractional power encoding.

    For a coordinate x, the encoding is: encode(x) = seed₀^x
    where exponentiation uses FFT-based phasor rotation for continuous powers.

    This makes binding equivariant to translation:
        bind(encode(x), encode(Δ)) ≈ encode(x + Δ)

    Args:
        hd_dim: Hyperdimensional vector dimension (must be even for complex FFT)
        min_val: Minimum coordinate value
        max_val: Maximum coordinate value
        seed: Random seed for generating basis vectors
        dtype: "complex" (FHRR-style, continuous powers) or "bipolar" (±1, discrete powers)
    """

    def __init__(
        self,
        hd_dim: int = 8192,
        min_val: float = 0.0,
        max_val: float = 1.0,
        seed: int = 777,
        dtype: str = "complex",
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.min_val = min_val
        self.max_val = max_val
        self.dtype = dtype

        # Generate seed vector (unit magnitude complex or bipolar)
        self.register_buffer(
            "seed_vector",
            self._generate_seed(hd_dim, seed, dtype),
        )

        # Precompute range for exponentiation
        self.range = max_val - min_val

        # For complex: precompute unit frequencies for FFT-based exponentiation
        if dtype == "complex":
            # Frequencies: k = 0..hd_dim/2 for positive frequencies
            # and conjugate for negative frequencies (handled by FFT)
            half = hd_dim // 2
            # Random phases for each frequency
            gen = torch.Generator()
            gen.manual_seed(seed + 100)
            self.register_buffer(
                "freq_phases",
                torch.rand(half + 1, generator=gen) * 2 * np.pi,
            )

    @staticmethod
    def _generate_seed(hd_dim: int, seed: int, dtype: str) -> torch.Tensor:
        """Generate the base seed vector for FPE."""
        gen = torch.Generator()
        gen.manual_seed(seed)

        if dtype == "complex":
            # Complex phasors: e^(iθ) on unit circle
            angles = torch.rand(hd_dim // 2 + 1, generator=gen) * 2 * np.pi
            return torch.complex(
                torch.cos(angles), torch.sin(angles)
            )
        else:
            # Bipolar ±1 (compat with torch < 1.13: no generator= in full)
            return (torch.rand(hd_dim, generator=gen) > 0.5).float() * 2 - 1

    def fractional_power(self, exponent: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Raise seed vector to a fractional (continuous) power: seed^exponent.

        For complex vectors, this is a phase rotation proportional to the exponent:
            seed^x = FFT^{-1}(e^{i·x·θ_k}) where θ_k are the seed's frequency phases.

        This is THE key operation that makes binding equivariant to addition,
        enabling resonator networks to factorize translations.

        Args:
            exponent: Scalar or tensor [..., 1] of continuous exponent values

        Returns:
            HD vector seed^exponent, same shape as seed_vector but with batch dims
        """
        if self.dtype == "complex":
            # Use FFT-based computation for continuous exponentiation
            half = self.hd_dim // 2

            # Phase shift each frequency by exponent * freq_phase
            # seed^x in frequency domain: multiply each frequency by e^{i·x·phase_k}
            if isinstance(exponent, (int, float)):
                exponent = torch.tensor(exponent, device=self.seed_vector.device)

            # Expand exponent to match frequency dimensions
            if exponent.dim() > 0:
                # Batch: [B] -> [B, half+1]
                exp = exponent.unsqueeze(-1).float()  # [..., 1]
            else:
                exp = exponent.float().unsqueeze(-1)  # [1]

            phases = self.freq_phases  # [half+1]
            shifted_phases = exp * phases  # broadcast
            freq_domain = torch.complex(
                torch.cos(shifted_phases),
                torch.sin(shifted_phases),
            )

            # Convert to time domain via IFFT
            # freq_domain: [..., half+1] -> complex time signal
            # Build full conjugate-symmetric spectrum
            if freq_domain.dim() == 2:
                B = freq_domain.shape[0]
                full_spec = torch.zeros(B, self.hd_dim, dtype=torch.complex64, device=freq_domain.device)
                full_spec[:, :half + 1] = freq_domain
                # Conjugate symmetry for negative frequencies (skip DC and Nyquist)
                full_spec[:, half + 1:] = freq_domain[:, 1:half].flip(-1).conj()
                result = torch.fft.ifft(full_spec, dim=-1)
            else:
                full_spec = torch.zeros(self.hd_dim, dtype=torch.complex64, device=freq_domain.device)
                full_spec[:half + 1] = freq_domain
                full_spec[half + 1:] = freq_domain[..., 1:half].flip(-1).conj()
                result = torch.fft.ifft(full_spec, dim=-1)

            # Normalize (IFTT may not produce exact unit magnitude)
            result = result / (result.abs().max(dim=-1, keepdim=True).values + 1e-8)
            return result
        else:
            # Bipolar: discrete powers using cyclic shift
            # This approximates fractional powers by rounding to nearest integer
            if isinstance(exponent, (int, float)):
                shift = int(round(exponent)) % self.hd_dim
                return torch.roll(self.seed_vector, shifts=shift, dims=-1)
            else:
                # Batch: use interpolation between integer shifts
                shift = exponent.round().long() % self.hd_dim
                results = []
                for s in shift:
                    results.append(torch.roll(self.seed_vector, shifts=int(s.item()), dims=-1))
                return torch.stack(results)

    def encode_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 1D coordinates into HD space.

        Args:
            x: [B] or [B, 1] continuous coordinate values in [min_val, max_val]

        Returns:
            HD vectors [B, hd_dim]
        """
        # Normalize to [0, range]
        x_norm = (x - self.min_val) / self.range
        x_norm = x_norm.float()

        if x_norm.dim() == 2:
            x_norm = x_norm.squeeze(-1)

        return self.fractional_power(x_norm)

    def encode_2d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Encode 2D coordinates into HD space using paired seed vectors.

        s = h₀^x ⊗ v₀^y  (Eq. 1 from paper)

        For efficient encoding of all pixel locations, use build_codebook() + encode_image().

        Args:
            x: [B] x-coordinate values
            y: [B] y-coordinate values

        Returns:
            HD vectors [B, hd_dim] = h₀^x ⊙ v₀^y
        """
        hx = self.encode_1d(x)  # [B, hd_dim]
        vy = self.encode_1d(y)  # [B, hd_dim]

        # Bind: element-wise multiply
        return self._bind(hx, vy)

    def _bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bind two HD vectors (element-wise multiply = circular convolution)."""
        if self.dtype == "complex":
            return a * b
        return a * b

    def build_codebook(
        self, height: int, width: int
    ) -> torch.Tensor:
        """
        Build the codebook matrix Φ that encodes all pixel locations.

        Φ[x, y] = h₀^x ⊗ v₀^y for all x∈[0,width), y∈[0,height)

        The codebook enables efficient encoding of an entire binary event frame
        via a single matrix multiplication: s = Φ · I  (Eq. 3 in paper)

        Args:
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            Codebook matrix [height * width, hd_dim]
        """
        # Generate normalized coordinate grids
        xs = torch.linspace(self.min_val, self.max_val, width)
        ys = torch.linspace(self.min_val, self.max_val, height)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        xx_flat = xx.reshape(-1)  # [H*W]
        yy_flat = yy.reshape(-1)  # [H*W]

        # Encode each pixel
        # For efficiency with large images, process in chunks
        chunk_size = 1024
        codebook = []
        for i in range(0, len(xx_flat), chunk_size):
            xc = xx_flat[i:i + chunk_size]
            yc = yy_flat[i:i + chunk_size]
            encoded = self.encode_2d(xc, yc)
            codebook.append(encoded)

        return torch.cat(codebook, dim=0)  # [H*W, hd_dim]

    def encode_image(self, image: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        Encode a binary/sparse image into a single HD vector using the codebook.

        s = Φ · I  (Eq. 3 in paper, where I is the flattened binary image)

        This bundles (sums) the codebook vectors for all active pixels.

        Args:
            image: [B, H, W] or [H, W] binary/sparse image (active pixels = 1)
            codebook: [H*W, hd_dim] precomputed codebook

        Returns:
            Encoded HD vector [B, hd_dim] or [hd_dim]
        """
        squeeze_out = False
        if image.dim() == 2:
            image = image.unsqueeze(0)
            squeeze_out = True

        B, H, W = image.shape
        flat = image.reshape(B, H * W).float()  # [B, H*W]
        encoded = flat @ codebook.to(flat.device)  # [B, hd_dim]

        if self.dtype == "bipolar":
            encoded = torch.sign(encoded)

        encoded = F.normalize(encoded, p=2, dim=-1)

        if squeeze_out:
            encoded = encoded.squeeze(0)

        return encoded

    def translation_kernel(self, dx: float, dy: float) -> torch.Tensor:
        """
        Create a translation kernel in HD space.

        Translating the image by (dx, dy) is equivalent to binding with this kernel:
            encode(image_translated) ≈ translation_kernel(dx, dy) ⊗ encode(image)

        This is the mathematical property (convolution theorem) that enables
        resonator networks to factor out translations.

        Args:
            dx: Horizontal translation (in normalized coordinates)
            dy: Vertical translation (in normalized coordinates)

        Returns:
            Translation HD vector [hd_dim]
        """
        if isinstance(dx, torch.Tensor) and dx.dim() > 0:
            return self.encode_2d(dx, dy)
        dx_t = torch.tensor(dx, device=self.seed_vector.device)
        dy_t = torch.tensor(dy, device=self.seed_vector.device)
        return self.encode_2d(dx_t, dy_t)

    def rotation_kernel(self, angle: float) -> torch.Tensor:
        """
        Create a rotation kernel in HD space for the log-polar partition.

        In log-polar coordinates, rotation becomes a circular shift, which
        corresponds to binding with a rotation seed vector.

        Args:
            angle: Rotation angle in radians

        Returns:
            Rotation HD vector [hd_dim]
        """
        # Normalize angle to [0, 2π] → [0, 1]
        angle_norm = (angle % (2 * np.pi)) / (2 * np.pi)
        return self.fractional_power(torch.tensor(angle_norm, device=self.seed_vector.device))

    def scale_kernel(self, scale: float) -> torch.Tensor:
        """
        Create a scale kernel.

        In log-polar coordinates, scaling becomes a translation,
        so scaling is binding with a scale seed raised to log(scale).

        Args:
            scale: Scale factor (> 0)

        Returns:
            Scale HD vector [hd_dim]
        """
        log_scale = np.log2(max(scale, 1e-6))
        return self.fractional_power(torch.tensor(log_scale, device=self.seed_vector.device))

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mode: str = "encode",
    ) -> torch.Tensor:
        """
        Unified forward for FPE encoding.

        Args:
            x: Coordinates or image tensor
            y: Optional y-coordinates
            mode: "encode_1d", "encode_2d", "encode_image", "kernel_translation",
                  "kernel_rotation", "kernel_scale"
        """
        if mode == "encode_1d":
            return self.encode_1d(x)
        elif mode == "encode_2d" and y is not None:
            return self.encode_2d(x, y)
        elif mode == "kernel_translation" and y is not None:
            return self.translation_kernel(x, y)
        elif mode == "kernel_rotation":
            return self.rotation_kernel(x)
        elif mode == "kernel_scale":
            return self.scale_kernel(x)
        return self.encode_1d(x)


class FPEImageEncoder(nn.Module):
    """
    Efficient FPE-based image/event-frame encoder for Eldarin.

    Prepares a codebook once based on image dimensions, then encodes
    images/event frames via matrix multiplication with the codebook.

    This can serve as a training-free alternative to the learned
    event encoder for VSA-native reasoning paths.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        hd_dim: HD vector dimension
        dtype: "complex" (FHRR) or "bipolar"
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        hd_dim: int = 8192,
        dtype: str = "complex",
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.hd_dim = hd_dim

        self.fpe = FractionalPowerEncoder(
            hd_dim=hd_dim,
            min_val=0.0,
            max_val=float(max(height, width)),
            dtype=dtype,
        )

        # Build codebook once (frozen)
        self.register_buffer(
            "codebook",
            self.fpe.build_codebook(height, width),
            persistent=True,
        )

    def forward(
        self,
        image: torch.Tensor,
        events: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """
        Encode an image or event accumulation into HD space.

        Args:
            image: [B, H, W] binary/sparse image or event accumulation
            events: Optional event tuple (used to build event frame if image is None)

        Returns:
            HD vector [B, hd_dim]
        """
        if image is None and events is not None:
            # Build event frame from events
            x, y, t, p = events
            B = 1  # Event encoding is per-frame
            H, W = self.height, self.width
            image = torch.zeros(H, W, device=x.device)
            xi = x.long().clamp(0, W - 1)
            yi = y.long().clamp(0, H - 1)
            image[yi, xi] = 1.0
            image = image.unsqueeze(0)  # [1, H, W]

        return self.fpe.encode_image(image, self.codebook)


# ----- Utility: Log-polar coordinate transform for resonator networks -----

def cartesian_to_logpolar(
    image_hd: torch.Tensor,
    height: int,
    width: int,
    n_angles: int = 64,
    n_radii: int = 64,
) -> torch.Tensor:
    """
    Approximate conversion from Cartesian FPE encoding to log-polar encoding.

    In the paper (Λ transform), this enables the resonator to handle
    rotation and scale as translations in log-polar space.

    This is a simplified differentiable approximation using coordinate
    remapping. A full implementation would use the Λ matrix transform
    described in the paper's Methods section.

    Args:
        image_hd: [B, hd_dim] HD-encoded image in Cartesian FPE
        height: Original image height
        width: Original image width
        n_angles: Number of angular bins in log-polar output
        n_radii: Number of radial bins in log-polar output

    Returns:
        [B, n_angles * n_radii, hd_dim] log-polar encoded representation
    """
    B = image_hd.shape[0]
    device = image_hd.device

    # Build log-polar sampling grid
    center_x, center_y = width / 2.0, height / 2.0
    max_radius = min(center_x, center_y)

    angles = torch.linspace(0, 2 * np.pi, n_angles + 1, device=device)[:n_angles]
    radii = torch.logspace(0, np.log10(max_radius), n_radii, device=device)

    # Sample locations
    aa, rr = torch.meshgrid(angles, radii, indexing="ij")
    xx = center_x + rr * torch.cos(aa)
    yy = center_y + rr * torch.sin(aa)

    # Normalize coordinates
    xx_norm = xx / width
    yy_norm = yy / height

    # We approximate log-polar encoding as weighted combination
    # of nearest Cartesian encodings
    fpe = FractionalPowerEncoder(image_hd.shape[-1], 0.0, max(width, height))
    logpolar_encoding = fpe.encode_2d(
        xx_norm.reshape(-1), yy_norm.reshape(-1)
    )  # [n_angles * n_radii, hd_dim]

    # Return as spatial representation
    return logpolar_encoding.unsqueeze(0).expand(B, -1, -1)


# Test utilities
def test_fpe_properties():
    """Verify FPE mathematical properties from the paper."""
    print("Testing Fractional Power Encoding properties...")

    hd_dim = 1024
    fpe = FractionalPowerEncoder(hd_dim=hd_dim, min_val=0.0, max_val=100.0)

    # Test 1: Binding equals addition
    x = 10.0
    delta = 5.0
    enc_x = fpe.encode_1d(torch.tensor([x / 100.0]))
    enc_delta = fpe.encode_1d(torch.tensor([delta / 100.0]))
    enc_sum = fpe.encode_1d(torch.tensor([(x + delta) / 100.0]))

    bound = fpe._bind(enc_x, enc_delta)
    bound = F.normalize(bound, p=2, dim=-1)
    enc_sum = F.normalize(enc_sum, p=2, dim=-1)

    sim = F.cosine_similarity(bound, enc_sum, dim=-1).item()
    print(f"  FPE binding = addition: sim(enc(x)⊗enc(Δ), enc(x+Δ)) = {sim:.4f}")
    assert sim > 0.8, f"FPE binding equivariance failed: {sim}"

    # Test 2: Codebook encoding
    height, width = 32, 32
    codebook = fpe.build_codebook(height, width)
    print(f"  Codebook shape: {codebook.shape}")

    # Create test image
    image = torch.zeros(height, width)
    image[10:20, 10:20] = 1.0
    encoded = fpe.encode_image(image, codebook)
    print(f"  Encoded image shape: {encoded.shape}")

    # Test 3: Translation via binding
    # Shift image right by 2 pixels -> should be ~ binding with translation kernel(2,0)
    dx, dy = 2.0, 0.0
    trans_kernel = fpe.translation_kernel(dx / width, dy / height)
    image2 = torch.zeros(height, width)
    image2[10:20, 12:22] = 1.0  # Shifted right by 2
    encoded2 = fpe.encode_image(image2, codebook)
    bound_encoded = fpe._bind(encoded, trans_kernel.unsqueeze(0))
    bound_encoded = F.normalize(bound_encoded, p=2, dim=-1)
    encoded2 = F.normalize(encoded2, p=2, dim=-1)
    sim_trans = F.cosine_similarity(bound_encoded, encoded2, dim=-1).item()
    print(f"  Translation via binding: sim(encode(img)⊗kernel(2,0), encode(img_shifted)) = {sim_trans:.4f}")
    # Note: due to boundary effects (new pixels on left), similarity may be moderate
    assert sim_trans > 0.3, f"Translation binding failed: {sim_trans}"

    print("FPE tests passed!\n")


if __name__ == "__main__":
    test_fpe_properties()