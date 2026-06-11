"""
HDC-EVIO — Hyperdimensional Event Visual-Inertial Odometry
===========================================================
Cortex-A soft real-time estimator (10s–100s of Hz). Replaces GPU-heavy
VIO with hyperdimensional (VSA/HDC) operations that can run with NEON SIMD.

Pipeline (per the outline §4.2):
  1. Receive SensorPacket (event frame + IMU + optical flow + distance + baro)
  2. Spatial-Inertial Hypervector Encoding:
     • FPE-encode event frame → translation-equivariant HD vector
     • Bind IMU via FPE (Eq. 10 from Renner et al.)
     • Fuse optical flow + range + baro as additional HD bindings
  3. HDC-EVIO estimation:
     • Resonator network factorises translation + rotation from
       encoded input and allocentric map
     • Population vector readout → sub-pixel pose
     • Map anchoring (Eq. 9) prevents long-term drift
  4. Emit StateEstimate correction → MCU EKF at ~20–50 Hz

Design decision (§6.1): hypervector encoding currently runs on Cortex-A
for fast iteration. Move to FPGA only if CPU numbers don't meet budget.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque

from ..messages import (
    SensorPacket, EventFrameMessage, ImuMessage, OpticalFlowMessage,
    DistanceMessage, BarometerMessage, StateEstimate, now_us,
)

# Import existing VSA infrastructure from the model package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.fpe import FractionalPowerEncoder
from model.vsa_hdc import HierarchicalResonatorNetwork


@dataclass
class HDCEvioConfig:
    """
    Configuration for the HDC-EVIO pipeline.
    Tuned for Cortex-A with NEON SIMD (binary or phasor HDC).
    """
    # Model dimensions
    hd_dim: int = 8192
    image_height: int = 180
    image_width: int = 240

    # Resonator
    cartesian_factors: Tuple[int, int] = (64, 64)
    logpolar_factors: Tuple[int, ...] = (36,)
    resonator_gamma: float = 0.3
    resonator_nonlinearity: str = "phasor"
    resonator_iterations: int = 10
    hd_dtype: str = "bipolar"  # "bipolar" for NEON bit-packed, "complex" for phasor prototype
    seed: int = 777

    # Working memory
    map_forgetting: float = 0.90    # μ₁ in Eq. 9
    anchor_weight: float = 0.05     # μ₂ in Eq. 9
    update_delay: int = 100         # steps before live map updates

    # IMU fusion (Eq. 10)
    enable_imu_fusion: bool = True
    imu_weight: float = 0.3  # how much IMU prediction shifts the resonator prior

    # Output rate
    correction_rate_hz: float = 20.0  # how often to emit StateEstimate corrections

    # Covariance estimates for the correction message
    position_cov: float = 0.05    # m² — from paper benchmarks (0.078m ATE)
    velocity_cov: float = 0.1     # (m/s)²
    attitude_cov: float = 0.005   # rad² — 3.5° ≈ 0.061 rad → ~0.004 rad²

    # NEON SIMD hints
    use_bit_packed: bool = False  # when True, use binary bipolar vectors + XOR/popcount
    recommended_core: int = 0     # dedicated core number for EVIO thread


class SpatialInertialEncoder:
    """
    Encoder that fuses all sensor modalities into a single spatial-inertial
    hypervector representing current ego-motion context (§4.5.1 of outline).

    Encoding strategy:
      - Event frame → FPE (translation-equivariant)
      - IMU → FPE binding with seed vectors (Eq. 10)
      - Optical flow → bind as velocity context HD vector
      - Distance → bind as altitude/depth context
      - Barometer → bind as absolute altitude reference
      - Bundle all together → single spatial-inertial HV
    """

    def __init__(self, cfg: HDCEvioConfig):
        self.cfg = cfg
        self.hd_dim = cfg.hd_dim

        # Core FPE encoder (same as VisualOdometryVSA)
        self.fpe = FractionalPowerEncoder(
            hd_dim=cfg.hd_dim,
            min_val=0.0,
            max_val=float(max(cfg.image_height, cfg.image_width)),
            dtype=cfg.hd_dtype,
            seed=cfg.seed,
        )

        # Codebook cache
        self._codebook: Optional[torch.Tensor] = None

        # Context seed vectors (pre-generated, fixed)
        rng = np.random.RandomState(cfg.seed + 100)
        if cfg.hd_dtype == "bipolar":
            self.imu_context_seed = torch.from_numpy(
                (rng.rand(cfg.hd_dim) > 0.5).astype(np.float32) * 2 - 1
            )
            self.flow_context_seed = torch.from_numpy(
                (rng.rand(cfg.hd_dim) > 0.5).astype(np.float32) * 2 - 1
            )
            self.range_context_seed = torch.from_numpy(
                (rng.rand(cfg.hd_dim) > 0.5).astype(np.float32) * 2 - 1
            )
            self.baro_context_seed = torch.from_numpy(
                (rng.rand(cfg.hd_dim) > 0.5).astype(np.float32) * 2 - 1
            )
        else:
            angles = rng.rand(cfg.hd_dim) * 2 * np.pi
            self.imu_context_seed = torch.from_numpy(
                np.exp(1j * angles).astype(np.complex64)
            )
            self.flow_context_seed = torch.from_numpy(
                np.exp(1j * (angles + np.pi)).astype(np.complex64)
            )
            self.range_context_seed = torch.from_numpy(
                np.exp(1j * (angles + np.pi / 2)).astype(np.complex64)
            )
            self.baro_context_seed = torch.from_numpy(
                np.exp(1j * (angles - np.pi / 2)).astype(np.complex64)
            )

    def get_codebook(self) -> torch.Tensor:
        if self._codebook is None:
            self._codebook = self.fpe.build_codebook(
                self.cfg.image_height, self.cfg.image_width
            )
        return self._codebook

    def encode_event_frame(self, ef: EventFrameMessage) -> torch.Tensor:
        """Convert sparse event frame to FPE-encoded HD vector."""
        H, W = self.cfg.image_height, self.cfg.image_width
        device = self.imu_context_seed.device

        # Build dense binary image from sparse activations
        image = torch.zeros(H, W, device=device)
        if ef.xs:
            xi = torch.tensor(ef.xs, device=device, dtype=torch.long).clamp(0, W - 1)
            yi = torch.tensor(ef.ys, device=device, dtype=torch.long).clamp(0, H - 1)
            image[yi, xi] = torch.tensor(ef.activations, device=device, dtype=torch.float32)

        codebook = self.get_codebook()
        encoded = self.fpe.encode_image(image.unsqueeze(0), codebook)
        return F.normalize(encoded, p=2, dim=-1)

    def fuse_imu(self, encoded: torch.Tensor, imu: ImuMessage) -> torch.Tensor:
        """IMU fusion via FPE binding (Renner et al. Eq. 10)."""
        if not self.cfg.enable_imu_fusion:
            return encoded

        # Use gyro z (yaw rate) + accel magnitude as context binding
        yaw_rate = imu.gyro[2]
        accel_norm = np.linalg.norm(imu.accel)

        # FPE: seed^(IMU_reading) — approximate with circular shift / phase offset
        # For bipolar: XOR with context modulated by reading
        imu_shift = torch.roll(
            self.imu_context_seed.to(encoded.device),
            shifts=int(yaw_rate * 1000) % self.hd_dim,
        )
        if self.cfg.hd_dtype == "bipolar":
            # Bipolar binding = elementwise multiplication (XOR equivalent)
            encoded = encoded * imu_shift.unsqueeze(0)
        else:
            encoded = encoded * imu_shift.unsqueeze(0)

        encoded = F.normalize(encoded, p=2, dim=-1)
        return encoded

    def fuse_optical_flow(self, encoded: torch.Tensor, flow: OpticalFlowMessage) -> torch.Tensor:
        """Bind optical flow as velocity context."""
        # Modulate by flow magnitude
        mag = np.sqrt(flow.dx ** 2 + flow.dy ** 2)
        shift = int(mag * 500) % self.hd_dim
        ctx = torch.roll(self.flow_context_seed.to(encoded.device), shifts=shift)
        if self.cfg.hd_dtype == "bipolar":
            encoded = encoded * ctx.unsqueeze(0)
        else:
            encoded = encoded * ctx.unsqueeze(0)
        return F.normalize(encoded, p=2, dim=-1)

    def fuse_distance(self, encoded: torch.Tensor, dist: DistanceMessage) -> torch.Tensor:
        """Bind altitude/depth context."""
        alt_norm = max(dist.down_mm / 10_000.0, 0.01)  # normalise ~ 0.01–1.0
        shift = int(alt_norm * 800) % self.hd_dim
        ctx = torch.roll(self.range_context_seed.to(encoded.device), shifts=shift)
        if self.cfg.hd_dtype == "bipolar":
            encoded = encoded * ctx.unsqueeze(0)
        else:
            encoded = encoded * ctx.unsqueeze(0)
        return F.normalize(encoded, p=2, dim=-1)

    def fuse_barometer(self, encoded: torch.Tensor, baro: BarometerMessage) -> torch.Tensor:
        """Bind absolute altitude context."""
        alt = baro.pressure_hpa / 1013.25  # normalised pressure
        shift = int(alt * 500) % self.hd_dim
        ctx = torch.roll(self.baro_context_seed.to(encoded.device), shifts=shift)
        if self.cfg.hd_dtype == "bipolar":
            encoded = encoded * ctx.unsqueeze(0)
        else:
            encoded = encoded * ctx.unsqueeze(0)
        return F.normalize(encoded, p=2, dim=-1)

    def encode(self, packet: SensorPacket) -> torch.Tensor:
        """
        Full spatial-inertial encoding from a SensorPacket.
        Returns [1, hd_dim] hypervector.
        """
        encoded = None

        # Primary: event frame (mandatory)
        if packet.event_frame is not None:
            encoded = self.encode_event_frame(packet.event_frame)
        else:
            # Fallback: zero vector (will be ignored)
            encoded = torch.zeros(1, self.hd_dim, device=self.imu_context_seed.device)

        # Fuse each available sensor
        if packet.imu is not None:
            encoded = self.fuse_imu(encoded, packet.imu)
        if packet.optical_flow is not None:
            encoded = self.fuse_optical_flow(encoded, packet.optical_flow)
        if packet.distance is not None:
            encoded = self.fuse_distance(encoded, packet.distance)
        if packet.barometer is not None:
            encoded = self.fuse_barometer(encoded, packet.barometer)

        return encoded


class HDCEvioEstimator:
    """
    HDC-EVIO: Ego-motion estimation from spatial-inertial hypervectors.

    Implements §4.2 of the outline with the Renner et al. (2024) VSA-native
    VO pipeline as the core estimation engine.
    """

    def __init__(
        self,
        config: Optional[HDCEvioConfig] = None,
        device: str = "cpu",
    ):
        self.cfg = config or HDCEvioConfig()
        self.device = torch.device(device)

        # Spatial-inertial encoder
        self.encoder = SpatialInertialEncoder(self.cfg)

        # Hierarchical resonator (core VO engine)
        self.resonator = HierarchicalResonatorNetwork(
            cartesian_factors=list(self.cfg.cartesian_factors),
            logpolar_factors=list(self.cfg.logpolar_factors),
            hd_dim=self.cfg.hd_dim,
            gamma=self.cfg.resonator_gamma,
            nonlinearity=self.cfg.resonator_nonlinearity,
            dtype=self.cfg.hd_dtype,
            seed=self.cfg.seed,
        )
        # Move to device
        for cb in self.resonator.codebooks:
            cb.data = cb.data.to(self.device)

        # Allocentric map
        self.map_hd: Optional[torch.Tensor] = None
        self.anchor_map: Optional[torch.Tensor] = None
        self._step_count: int = 0
        self._map_initialized: bool = False

        # State
        self.current_pose = {"h": 0.0, "v": 0.0, "r": 0.0, "s": 1.0}
        self.position = np.zeros(3)  # x, y, z (world frame, metres)
        self.velocity = np.zeros(3)

        # Correction throttling
        self._last_correction_us: int = 0
        self._min_correction_interval_us: int = int(
            1e6 / self.cfg.correction_rate_hz
        )

        # Trajectory log
        self._trajectory: deque = deque(maxlen=10000)
        self._trajectory.append([0.0, 0.0, 0.0])

    def _initialize_map(self, encoded: torch.Tensor):
        """Set the initial allocentric map to the first encoded input."""
        self.map_hd = encoded.detach().clone()
        self.anchor_map = self.map_hd.clone()
        self._map_initialized = True

    def _update_map(self, encoded: torch.Tensor, translation_hd: Optional[torch.Tensor] = None):
        """Anchored map update (Renner et al. Eq. 9)."""
        self._step_count += 1

        if not self._map_initialized:
            self._initialize_map(encoded)
            return

        if self._step_count < self.cfg.update_delay:
            return

        # Transform encoded input by estimated translation kernel
        map_input = encoded
        if translation_hd is not None:
            if self.cfg.hd_dtype == "complex":
                map_input = encoded * translation_hd.conj()
            else:
                map_input = encoded * translation_hd
            map_input = F.normalize(map_input, p=2, dim=-1)

        # Eq. 9: m̂(t+1) = μ₁·m̂(t) + μ₂·m̂(0) + (1-μ₁-μ₂)·m(t)
        new_map = (
            self.cfg.map_forgetting * self.map_hd
            + self.cfg.anchor_weight * self.anchor_map
            + (1.0 - self.cfg.map_forgetting - self.cfg.anchor_weight) * map_input.detach()
        )
        self.map_hd = F.normalize(new_map, p=2, dim=-1)

    def step(self, packet: SensorPacket) -> Dict[str, Any]:
        """
        Process one sensor packet through the HDC-EVIO pipeline.

        Returns dict with:
          - pose: {h, v, r, s} from resonator
          - position: np.array([x, y, z])
          - velocity: np.array([vx, vy, vz])
          - correction: StateEstimate (if ready to emit)
          - map_initialized: bool
          - resonator_output: full detail dict
        """
        # Step 1: Spatial-inertial encoding
        encoded = self.encoder.encode(packet)
        encoded = F.normalize(encoded, p=2, dim=-1)

        # Step 2: Initialize or retrieve map
        if not self._map_initialized:
            self._initialize_map(encoded)
            return {
                "pose": self.current_pose,
                "position": self.position,
                "velocity": self.velocity,
                "correction": None,
                "map_initialized": False,
                "resonator_output": {},
            }

        B = encoded.shape[0]
        map_batch = self.map_hd.unsqueeze(0).expand(B, -1)
        map_batch = F.normalize(map_batch, p=2, dim=-1)

        # Step 3: Resonator factorisation
        resonator_out = self.resonator(
            encoded_input=encoded,
            encoded_map=map_batch,
            num_iterations=self.cfg.resonator_iterations,
        )

        # Step 4: Population vector readout → pose
        h_val = float(resonator_out["translation"][0])
        v_val = float(resonator_out["translation"][1])
        r_val = float(resonator_out.get("rotation", torch.tensor(0.0)))
        self.current_pose = {"h": h_val, "v": v_val, "r": r_val, "s": 1.0}

        # Convert pixel-space translation to world metres
        # (calibration: pixels → metres scale factor)
        scale = 0.01  # metres per pixel (platform-dependent)
        self.position[0] += h_val * scale * np.cos(self.position[2] if hasattr(self, '_yaw') else 0.0)
        self.position[1] += v_val * scale * np.sin(self.position[2] if hasattr(self, '_yaw') else 0.0)
        self.position[2] = 0.0  # altitude from range/baro in full system

        self._trajectory.append(self.position.copy())

        # Step 5: Map update
        translation_hd = resonator_out.get("translation_hd")
        self._update_map(encoded, translation_hd)

        # Step 6: Should we emit a correction?
        now = now_us()
        correction = None
        if now - self._last_correction_us >= self._min_correction_interval_us:
            correction = self._build_correction(now)
            self._last_correction_us = now

        return {
            "pose": self.current_pose,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "correction": correction,
            "map_initialized": True,
            "resonator_output": resonator_out,
        }

    def _build_correction(self, timestamp_us: int) -> StateEstimate:
        """Build a StateEstimate from current EVIO state."""
        return StateEstimate(
            timestamp_us=timestamp_us,
            x=float(self.position[0]),
            y=float(self.position[1]),
            z=float(self.position[2]),
            vx=float(self.velocity[0]),
            vy=float(self.velocity[1]),
            vz=float(self.velocity[2]),
            roll=0.0,
            pitch=0.0,
            yaw=float(self.current_pose["r"]),
            cov_xx=self.cfg.position_cov,
            cov_yy=self.cfg.position_cov,
            cov_zz=self.cfg.position_cov * 2.0,
            cov_vx=self.cfg.velocity_cov,
            cov_vy=self.cfg.velocity_cov,
            cov_vz=self.cfg.velocity_cov * 2.0,
            cov_rr=self.cfg.attitude_cov,
            cov_rp=self.cfg.attitude_cov,
            cov_ry=self.cfg.attitude_cov,
            valid=True,
            source="hdc_evio",
        )

    def get_map(self) -> Optional[torch.Tensor]:
        return self.map_hd

    def get_trajectory(self) -> np.ndarray:
        return np.array(list(self._trajectory))

    def reset(self):
        self.map_hd = None
        self.anchor_map = None
        self._map_initialized = False
        self._step_count = 0
        self.current_pose = {"h": 0.0, "v": 0.0, "r": 0.0, "s": 1.0}
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self._trajectory.clear()
        self._trajectory.append([0.0, 0.0, 0.0])
        self._last_correction_us = 0