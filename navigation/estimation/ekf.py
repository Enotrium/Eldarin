"""
Extended Kalman Filter for MCU Inner Loop
==========================================
High-rate (kHz) sensor fusion for drone state estimation. Runs on a
high-performance MCU (the "inner loop") and accepts periodic corrections
from the Cortex-A's HDC-EVIO estimator.

State vector (12-D):
    x = [p_x, p_y, p_z, v_x, v_y, v_z, roll, pitch, yaw, b_a, b_g_x, b_g_y, b_g_z]  (13-D)

Sensors fused:
    - 6-DoF IMU (accel + gyro) at ~1 kHz          → gyro prediction for attitude
    - Downward optical flow at ~30–100 Hz          → v_x, v_y observation
    - IR laser rangefinders at ~30 Hz              → altitude observation
    - Barometer at ~10 Hz                          → slow absolute altitude
    - HDC-EVIO corrections at ~20 Hz               → full-state correction

Reference: Thrun, Burgard, Fox "Probabilistic Robotics" Ch. 3 (EKF)
           Beard & McLain "Small Unmanned Aircraft" Ch. 7 (drone EKF)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ..messages import ImuMessage, OpticalFlowMessage, DistanceMessage, BarometerMessage, StateEstimate


# ── State indices ────────────────────────────────────────────────────────────

PX, PY, PZ = 0, 1, 2      # position (world frame, m)
VX, VY, VZ = 3, 4, 5      # velocity (world frame, m/s)
ROLL, PITCH, YAW = 6, 7, 8  # Euler angles (rad)
BAX, BAY, BAZ = 9, 10, 11   # accel bias (m/s²)
BGX, BGY, BGZ = 12, 13, 14  # gyro bias (rad/s) — extended to 15-D for completeness
STATE_DIM = 15


@dataclass
class EKFConfig:
    """Configuration for the MCU Extended Kalman Filter."""
    # Initial uncertainty
    pos_sigma: float = 1.0         # m
    vel_sigma: float = 0.5         # m/s
    att_sigma: float = 0.1         # rad
    bias_a_sigma: float = 0.1      # m/s²
    bias_g_sigma: float = 0.01     # rad/s

    # Process noise (per √Δt)
    accel_noise: float = 0.5       # m/s²/√Hz
    gyro_noise: float = 0.01       # rad/s/√Hz
    bias_a_walk: float = 0.001     # m/s²/√Hz  bias random walk
    bias_g_walk: float = 0.0001    # rad/s/√Hz

    # Observation noise
    flow_noise: float = 0.1        # m/s (optical flow)
    range_noise: float = 0.05      # m   (IR rangefinder)
    baro_noise: float = 1.0        # m   (barometer)

    # Gravity
    g: float = 9.81

    # HDC-EVIO correction acceptance
    max_position_residual: float = 2.0     # m — reject if larger
    max_velocity_residual: float = 1.0     # m/s
    max_attitude_residual: float = 0.5     # rad


class ExtendedKalmanFilter:
    """
    15-state Extended Kalman Filter for drone navigation.

    Runs the prediction step at IMU rate (~1 kHz) and observation updates
    at sensor rates.  HDC-EVIO corrections are treated as full-state
    measurement updates at ~20 Hz.
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        self.cfg = config or EKFConfig()

        # State
        self.x = np.zeros(STATE_DIM)
        self._reset_attitude()

        # Covariance
        self.P = np.eye(STATE_DIM)
        self.P[PX, PX] = self.cfg.pos_sigma ** 2
        self.P[PY, PY] = self.cfg.pos_sigma ** 2
        self.P[PZ, PZ] = self.cfg.pos_sigma ** 2
        self.P[VX, VX] = self.cfg.vel_sigma ** 2
        self.P[VY, VY] = self.cfg.vel_sigma ** 2
        self.P[VZ, VZ] = self.cfg.vel_sigma ** 2
        self.P[ROLL, ROLL] = self.cfg.att_sigma ** 2
        self.P[PITCH, PITCH] = self.cfg.att_sigma ** 2
        self.P[YAW, YAW] = self.cfg.att_sigma ** 2
        self.P[BAX, BAX] = self.cfg.bias_a_sigma ** 2
        self.P[BAY, BAY] = self.cfg.bias_a_sigma ** 2
        self.P[BAZ, BAZ] = self.cfg.bias_a_sigma ** 2
        self.P[BGX, BGX] = self.cfg.bias_g_sigma ** 2
        self.P[BGY, BGY] = self.cfg.bias_g_sigma ** 2
        self.P[BGZ, BGZ] = self.cfg.bias_g_sigma ** 2

        self.I = np.eye(STATE_DIM)
        self._last_imu_us: int = 0
        self._n_predict = 0

    def _reset_attitude(self):
        """Start level with yaw=0 (world = body initially)."""
        self.x[ROLL] = 0.0
        self.x[PITCH] = 0.0
        self.x[YAW] = 0.0

    # ── Core EKF steps ───────────────────────────────────────────────────────

    def predict(self, imu: ImuMessage, dt: float) -> None:
        """
        IMU-driven prediction step.
        dt in seconds (IMU inter-sample interval, typ. 0.001 s).
        """
        accel = np.array(imu.accel) - self.x[BAX:BAZ + 1]
        gyro = np.array(imu.gyro) - self.x[BGX:BGZ + 1]

        roll, pitch, yaw = self.x[ROLL], self.x[PITCH], self.x[YAW]

        # ── Attitude kinematics (gyro integration) ──
        # Euler rate → world-frame angular velocity
        # (small-angle approximation for typical drone dt)
        cos_p = np.cos(pitch);  sin_p = np.sin(pitch)
        cos_r = np.cos(roll);   sin_r = np.sin(roll)

        # Body-rate → Euler-rate conversion matrix W^-1
        W_inv = np.array([
            [1, sin_r * sin_p / max(cos_p, 1e-6), cos_r * sin_p / max(cos_p, 1e-6)],
            [0, cos_r, -sin_r],
            [0, sin_r / max(cos_p, 1e-6), cos_r / max(cos_p, 1e-6)],
        ])
        euler_rate = W_inv @ gyro

        self.x[ROLL]  += euler_rate[0] * dt
        self.x[PITCH] += euler_rate[1] * dt
        self.x[YAW]   += euler_rate[2] * dt

        # ── Rotation matrix (body → world) ──
        R = self._rotation_matrix()

        # ── Velocity (accelerometer in world frame) ──
        accel_world = R @ accel
        accel_world[2] -= self.cfg.g  # remove gravity

        self.x[VX] += accel_world[0] * dt
        self.x[VY] += accel_world[1] * dt
        self.x[VZ] += accel_world[2] * dt

        # ── Position (trapezoidal velocity) ──
        self.x[PX] += self.x[VX] * dt + 0.5 * accel_world[0] * dt * dt
        self.x[PY] += self.x[VY] * dt + 0.5 * accel_world[1] * dt * dt
        self.x[PZ] += self.x[VZ] * dt + 0.5 * accel_world[2] * dt * dt

        # ── Covariance propagation ──
        # Compute Jacobian F = ∂f/∂x (simplified analytical)
        F = self._compute_jacobian(dt, R, accel, gyro)
        Q = self._process_noise_covariance(dt)

        self.P = F @ self.P @ F.T + Q
        self._enforce_symmetry()

        self._n_predict += 1
        self._last_imu_us = imu.timestamp_us

    def update_optical_flow(self, flow: OpticalFlowMessage, dt: float) -> None:
        """Optical flow velocity observation."""
        H = np.zeros((2, STATE_DIM))
        H[0, VX] = 1.0
        H[1, VY] = 1.0

        z = np.array([flow.dx, flow.dy])
        R = np.eye(2) * self.cfg.flow_noise ** 2

        self._kalman_update(H, z, R)

    def update_distance(self, dist: DistanceMessage) -> None:
        """IR rangefinder altitude observation (down sensor)."""
        H = np.zeros((1, STATE_DIM))
        H[0, PZ] = 1.0

        z = np.array([dist.down_mm / 1000.0])  # mm → m
        R = np.eye(1) * self.cfg.range_noise ** 2

        self._kalman_update(H, z, R)

    def update_barometer(self, baro: BarometerMessage) -> None:
        """
        Barometric altitude observation.
        Converts pressure to altitude using standard atmosphere model.
        """
        # Standard atmosphere: z = 44330 * (1 - (P/P0)^(1/5.255))
        P0 = 1013.25  # hPa at sea level
        alt_m = 44330.0 * (1.0 - (baro.pressure_hpa / P0) ** (1.0 / 5.255))

        H = np.zeros((1, STATE_DIM))
        H[0, PZ] = 1.0

        z = np.array([alt_m])
        R = np.eye(1) * self.cfg.baro_noise ** 2

        self._kalman_update(H, z, R)

    def apply_hdc_correction(self, corr: StateEstimate) -> bool:
        """
        Apply state correction from HDC-EVIO (Cortex-A).
        Returns True if correction was accepted (passed residual gates).
        """
        predicted = np.array([self.x[PX], self.x[PY], self.x[PZ],
                               self.x[VX], self.x[VY], self.x[VZ],
                               self.x[ROLL], self.x[PITCH], self.x[YAW]])
        observed = np.array([corr.x, corr.y, corr.z,
                             corr.vx, corr.vy, corr.vz,
                             corr.roll, corr.pitch, corr.yaw])

        residual = observed - predicted

        # ── Residual gates ──
        if abs(residual[PX]) > self.cfg.max_position_residual:
            return False
        if abs(residual[PY]) > self.cfg.max_position_residual:
            return False
        if abs(residual[PZ]) > self.cfg.max_position_residual:
            return False
        if np.linalg.norm(residual[VX:VZ + 1]) > self.cfg.max_velocity_residual:
            return False
        if np.max(np.abs(residual[ROLL:YAW + 1])) > self.cfg.max_attitude_residual:
            return False

        # ── Full-state measurement update (pos + vel + attitude) ──
        H = np.zeros((9, STATE_DIM))
        for i in range(9):
            H[i, i] = 1.0

        R = np.diag([
            corr.cov_xx, corr.cov_yy, corr.cov_zz,
            corr.cov_vx, corr.cov_vy, corr.cov_vz,
            corr.cov_rr, corr.cov_rp, corr.cov_ry,
        ])

        self._kalman_update(H, observed, R)
        return True

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_state(self) -> StateEstimate:
        """Produce a StateEstimate message from current EKF state."""
        return StateEstimate(
            timestamp_us=self._last_imu_us,
            x=self.x[PX], y=self.x[PY], z=self.x[PZ],
            vx=self.x[VX], vy=self.x[VY], vz=self.x[VZ],
            roll=self.x[ROLL], pitch=self.x[PITCH], yaw=self.x[YAW],
            cov_xx=self.P[PX, PX], cov_yy=self.P[PY, PY], cov_zz=self.P[PZ, PZ],
            cov_vx=self.P[VX, VX], cov_vy=self.P[VY, VY], cov_vz=self.P[VZ, VZ],
            cov_rr=self.P[ROLL, ROLL], cov_rp=self.P[PITCH, PITCH], cov_ry=self.P[YAW, YAW],
            source="ekf_only",
        )

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[PX:PZ + 1]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[VX:VZ + 1]

    @property
    def attitude(self) -> np.ndarray:
        return self.x[ROLL:YAW + 1]

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _rotation_matrix(self) -> np.ndarray:
        """Body-to-world rotation matrix from current Euler angles."""
        cr = np.cos(self.x[ROLL]);  sr = np.sin(self.x[ROLL])
        cp = np.cos(self.x[PITCH]); sp = np.sin(self.x[PITCH])
        cy = np.cos(self.x[YAW]);   sy = np.sin(self.x[YAW])

        R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

        return R_z @ R_y @ R_x

    def _compute_jacobian(
        self, dt: float, R: np.ndarray, accel: np.ndarray, gyro: np.ndarray
    ) -> np.ndarray:
        """State transition Jacobian F = I + A*dt (first-order)."""
        F = np.eye(STATE_DIM)

        # position ← velocity
        F[PX, VX] = dt; F[PY, VY] = dt; F[PZ, VZ] = dt

        # velocity ← attitude (gravity projection, accel bias)
        F[VX:VZ + 1, ROLL:YAW + 1] = self._dR_dtheta(accel) * dt
        F[VX:VZ + 1, BAX:BAZ + 1] = -R * dt

        # Euler-rate ← gyro bias
        # (simplified — full Jacobian has d(euler_rate)/d(attitude))
        F[ROLL:YAW + 1, BGX:BGZ + 1] = -np.eye(3) * dt

        return F

    def _dR_dtheta(self, accel: np.ndarray) -> np.ndarray:
        """Derivative of R*accel w.r.t. roll, pitch, yaw at current attitude."""
        # Numerical approximation — sufficient for drone dt
        dR = np.zeros((3, 3))
        eps = 1e-6
        for i in range(3):
            orig = self.x[ROLL + i]
            self.x[ROLL + i] += eps
            Rp = self._rotation_matrix()
            self.x[ROLL + i] -= 2 * eps
            Rm = self._rotation_matrix()
            self.x[ROLL + i] = orig
            dR[:, i] = ((Rp - Rm) / (2 * eps)) @ accel
        return dR

    def _process_noise_covariance(self, dt: float) -> np.ndarray:
        """Discrete-time process noise covariance Q."""
        Q = np.zeros((STATE_DIM, STATE_DIM))

        a2 = self.cfg.accel_noise ** 2 * dt
        g2 = self.cfg.gyro_noise ** 2 * dt
        ba2 = self.cfg.bias_a_walk ** 2 * dt
        bg2 = self.cfg.bias_g_walk ** 2 * dt

        # Position (integrated twice)
        Q[PX, PX] = a2 * dt * dt / 3; Q[PY, PY] = a2 * dt * dt / 3; Q[PZ, PZ] = a2 * dt * dt / 3
        # Velocity (integrated once)
        Q[VX, VX] = a2; Q[VY, VY] = a2; Q[VZ, VZ] = a2
        # Attitude
        Q[ROLL, ROLL] = g2; Q[PITCH, PITCH] = g2; Q[YAW, YAW] = g2
        # Bias walks
        Q[BAX, BAX] = ba2; Q[BAY, BAY] = ba2; Q[BAZ, BAZ] = ba2
        Q[BGX, BGX] = bg2; Q[BGY, BGY] = bg2; Q[BGZ, BGZ] = bg2

        return Q

    def _kalman_update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        """Standard Kalman measurement update."""
        y = z - H @ self.x  # innovation
        S = H @ self.P @ H.T + R  # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P
        self._enforce_symmetry()

    def _enforce_symmetry(self) -> None:
        """Ensure P remains symmetric (numerical stability)."""
        self.P = 0.5 * (self.P + self.P.T)

    def reset(self, pos: Optional[np.ndarray] = None) -> None:
        """Re-initialise filter (e.g. after arming)."""
        self.x = np.zeros(STATE_DIM)
        self._reset_attitude()
        if pos is not None:
            self.x[PX:PZ + 1] = pos
        self.P = np.eye(STATE_DIM) * 0.01


# ── Convenience wrapper ──────────────────────────────────────────────────────

class IMUOdometryEKF:
    """
    Lightweight IMU-odometry EKF for the inner loop.
    Accepts raw sensors at their native rates and provides a unified interface.
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        self.ekf = ExtendedKalmanFilter(config)
        self._last_imu_time: float = 0.0
        self._last_flow_time: float = 0.0
        self._last_range_time: float = 0.0
        self._last_baro_time: float = 0.0

    def feed_imu(self, imu: ImuMessage) -> None:
        now = imu.timestamp_us / 1e6
        if self._last_imu_time == 0:
            self._last_imu_time = now
            return  # need first dt
        dt = now - self._last_imu_time
        dt = max(min(dt, 0.01), 1e-6)  # clamp
        self.ekf.predict(imu, dt)
        self._last_imu_time = now

    def feed_optical_flow(self, flow: OpticalFlowMessage) -> None:
        now = flow.timestamp_us / 1e6
        dt = max(now - self._last_flow_time, 0.001)
        self.ekf.update_optical_flow(flow, dt)
        self._last_flow_time = now

    def feed_distance(self, dist: DistanceMessage) -> None:
        self.ekf.update_distance(dist)
        self._last_range_time = dist.timestamp_us / 1e6

    def feed_barometer(self, baro: BarometerMessage) -> None:
        self.ekf.update_barometer(baro)
        self._last_baro_time = baro.timestamp_us / 1e6

    def feed_hdc_correction(self, corr: StateEstimate) -> bool:
        return self.ekf.apply_hdc_correction(corr)

    def get_state(self) -> StateEstimate:
        return self.ekf.get_state()

    @property
    def position(self) -> np.ndarray:
        return self.ekf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.ekf.velocity

    @property
    def attitude(self) -> np.ndarray:
        return self.ekf.attitude