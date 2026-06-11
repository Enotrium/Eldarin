"""
Inter-Domain Message Contracts
===============================
All message types that cross compute-domain boundaries (MCU ↔ Cortex-A ↔ FPGA).

Design rule: define interfaces early so teams can develop in parallel against
stubs.  Every message includes a timestamp (µs since epoch) for synchronization
across the three clock domains.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


# ── Timestamp helper ──────────────────────────────────────────────────────────

def now_us() -> int:
    """Monotonic timestamp in microseconds (use MCU / FPGA compatible source)."""
    return int(time.monotonic() * 1_000_000)


# ── Sensor messages (FPGA / MCU → Cortex-A) ──────────────────────────────────

@dataclass
class ImuMessage:
    """6-DoF IMU reading at ~1 kHz from MCU."""
    timestamp_us: int
    accel: Tuple[float, float, float]   # m/s²  (x, y, z)
    gyro:  Tuple[float, float, float]   # rad/s (roll, pitch, yaw)
    temperature: float = 25.0


@dataclass
class EventFrameMessage:
    """Sparse event accumulation from FPGA SNN feature tracker.
    Contains per-pixel activity and identified feature indices."""
    timestamp_us: int
    width: int
    height: int
    # Sparse CSR-style: active pixel coordinates + activation values
    xs: List[int] = field(default_factory=list)
    ys: List[int] = field(default_factory=list)
    activations: List[float] = field(default_factory=list)
    # SNN-tracked feature IDs for each active pixel
    feature_ids: List[int] = field(default_factory=list)


@dataclass
class OpticalFlowMessage:
    """Downward optical-flow reading from MCU."""
    timestamp_us: int
    dx: float  # pixels/frame (ground-relative velocity proxy)
    dy: float


@dataclass
class DistanceMessage:
    """Infrared laser rangefinder readings."""
    timestamp_us: int
    front_mm: float
    down_mm: float
    left_mm: float  = 0.0
    right_mm: float = 0.0
    back_mm: float  = 0.0


@dataclass
class BarometerMessage:
    """Barometer altitude reading."""
    timestamp_us: int
    pressure_hpa: float
    temperature_c: float = 15.0

# ── State messages (Cortex-A → MCU) ──────────────────────────────────────────

@dataclass
class StateEstimate:
    """Full 6-DoF state correction from HDC-EVIO → MCU EKF.
    Sent at ~20-50 Hz (soft real-time)."""
    timestamp_us: int
    # Position (world frame, metres)
    x: float; y: float; z: float
    # Velocity (world frame, m/s)
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0
    # Orientation (Euler, radians)
    roll: float = 0.0; pitch: float = 0.0; yaw: float = 0.0
    # Covariance diagonal (position uncertainty)
    cov_xx: float = 0.01; cov_yy: float = 0.01; cov_zz: float = 0.01
    cov_vx: float = 0.1;  cov_vy: float = 0.1;  cov_vz: float = 0.1
    cov_rr: float = 0.01; cov_rp: float = 0.01; cov_ry: float = 0.02
    # Flags
    valid: bool = True
    source: str = "hdc_evio"  # "ekf_only" | "hdc_evio" | "hdc_slam"


@dataclass
class ControlCommand:
    """Motor commands from LQR controller → Motor ESC."""
    timestamp_us: int
    # Normalised motor forces [0, 1] or PWM µs
    motor_1: float; motor_2: float
    motor_3: float; motor_4: float
    # Aux channels (servos, payload trigger, etc.)
    aux_channels: List[float] = field(default_factory=list)


@dataclass
class Waypoint:
    """3-DOF target from path planner → LQR controller."""
    timestamp_us: int
    x: float; y: float; z: float
    yaw: float = 0.0
    hold_time_s: float = 0.0       # loiter duration
    acceptance_radius_m: float = 1.0
    speed_m_s: float = 5.0         # max transit speed


# ── Domain boundary packets ───────────────────────────────────────────────────

@dataclass
class MpuMessage:
    """Aggregated MCU → Cortex-A packet (every 10 ms).
    Carries raw sensor data for HDC-EVIO and logging."""
    timestamp_us: int
    imu: ImuMessage
    optical_flow: Optional[OpticalFlowMessage] = None
    distances: Optional[DistanceMessage] = None
    barometer: Optional[BarometerMessage] = None
    # Current EKF-only state for comparison
    ekf_state: Optional[StateEstimate] = None


@dataclass
class SensorPacket:
    """Aggregated sensor bundle for HDC encoding pipelines."""
    # Spatial-inertial encoding inputs
    event_frame: Optional[EventFrameMessage] = None
    imu: Optional[ImuMessage] = None
    optical_flow: Optional[OpticalFlowMessage] = None
    distance: Optional[DistanceMessage] = None
    barometer: Optional[BarometerMessage] = None
    # Feature encoding inputs
    # (same event_frame, but with depth/altitude appended)


@dataclass
class DomainMessage:
    """Generic message envelope for inter-domain communication.
    Wraps any of the above types with routing metadata."""
    source: str        # "mcu" | "cortex_a" | "fpga"
    dest: str          # "mcu" | "cortex_a" | "fpga"
    priority: int = 0  # 0=normal, 1=high (flight-critical)
    payload: object = None