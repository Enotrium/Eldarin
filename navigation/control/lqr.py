"""
LQR Controller for MCU Inner Loop
==================================
Linear Quadratic Regulator for quadrotor attitude and position control.
Runs at hard real-time kHz rate on the MCU, taking the EKF state estimate
and control targets (waypoints from path planner) to compute actuator commands.

Reference: Beard & McLain "Small Unmanned Aircraft" Ch. 6, Ch. 10
           Stevens & Lewis "Aircraft Control and Simulation" Ch. 5

Quadrotor state (12-D):
    x = [p_n, p_e, p_d, u, v, w, φ, θ, ψ, p, q, r]
    p_n, p_e, p_d:  North, East, Down position (m)
    u, v, w:        Body-frame velocity (m/s)
    φ, θ, ψ:        Roll, pitch, yaw (rad)
    p, q, r:        Body-frame angular rates (rad/s)

Control inputs (4-D):
    u = [δ_t, δ_a, δ_e, δ_r]
    δ_t:  Thrust (N)
    δ_a:  Roll moment (N·m)
    δ_e:  Pitch moment (N·m)
    δ_r:  Yaw moment (N·m)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ..messages import StateEstimate, Waypoint, ControlCommand, now_us


@dataclass
class LQRConfig:
    """Configuration for the LQR controller."""
    # Quadrotor physical parameters
    mass_kg: float = 1.5          # vehicle mass
    Ixx: float = 0.03             # kg·m²
    Iyy: float = 0.03
    Izz: float = 0.05
    arm_length_m: float = 0.25    # motor-to-CG distance
    max_thrust_per_motor: float = 10.0  # N
    min_thrust_per_motor: float = 0.0
    hover_thrust: float = 0.25       # normalised hover thrust fraction

    # LQR weights
    # Position
    Q_xy: float = 1.0
    Q_z: float = 2.0
    # Velocity
    Q_vxy: float = 0.5
    Q_vz: float = 1.0
    # Attitude
    Q_roll: float = 10.0
    Q_pitch: float = 10.0
    Q_yaw: float = 5.0
    # Angular rates
    Q_p: float = 1.0
    Q_q: float = 1.0
    Q_r: float = 1.0

    # Control effort
    R_thrust: float = 0.1
    R_roll: float = 0.1
    R_pitch: float = 0.1
    R_yaw: float = 0.1

    # Limits
    max_roll_rad: float = np.deg2rad(30)
    max_pitch_rad: float = np.deg2rad(30)
    max_yaw_rate: float = np.deg2rad(90)

    # NED frame definition
    gravity: float = 9.81  # m/s²


class MotorMixing:
    """
    Converts control forces/moments (thrust, roll, pitch, yaw moments)
    into individual motor commands for common quadrotor frames.

    Supports:
      - Quad X (default)
      - Quad +
    """

    def __init__(
        self,
        arm_length: float = 0.25,
        torque_constant: float = 0.01,  # N·m per unit thrust (simplified)
        frame_type: str = "x",
    ):
        self.L = arm_length
        self.k_tau = torque_constant
        self.frame_type = frame_type

    def mix(self, thrust: float, tau_phi: float, tau_theta: float, tau_psi: float) -> np.ndarray:
        """
        Convert [T, τ_φ, τ_θ, τ_ψ] → [ω₁², ω₂², ω₃², ω₄²] (normalised motor speeds).

        Quad X mixer:
            ω₁² = T/4 - τ_φ/(4L) - τ_θ/(4L) - τ_ψ/(4kτ)
            ω₂² = T/4 - τ_φ/(4L) + τ_θ/(4L) + τ_ψ/(4kτ)
            ω₃² = T/4 + τ_φ/(4L) + τ_θ/(4L) - τ_ψ/(4kτ)
            ω₄² = T/4 + τ_φ/(4L) - τ_θ/(4L) + τ_ψ/(4kτ)
        """
        L = self.L
        k = max(self.k_tau, 0.001)

        if self.frame_type == "x":
            # Quad X configuration
            M = np.array([
                [0.25, -0.25 / L, -0.25 / L, -0.25 / k],
                [0.25, -0.25 / L,  0.25 / L,  0.25 / k],
                [0.25,  0.25 / L,  0.25 / L, -0.25 / k],
                [0.25,  0.25 / L, -0.25 / L,  0.25 / k],
            ])
        else:
            # Quad + configuration
            M = np.array([
                [0.25,  0.0,     -0.25 / L, -0.25 / k],
                [0.25,  0.25 / L,  0.0,       0.25 / k],
                [0.25,  0.0,      0.25 / L, -0.25 / k],
                [0.25, -0.25 / L, 0.0,       0.25 / k],
            ])

        controls = np.array([thrust, tau_phi, tau_theta, tau_psi])
        motor_squared = M @ controls
        # Clamp to valid range
        motor_squared = np.clip(motor_squared, 0.0, 1.0)
        return motor_squared


class LQRController:
    """
    LQR position + attitude controller for quadrotor.

    Two-stage cascaded architecture:
      1. Outer loop (position): position error → desired velocity → desired thrust + attitude
      2. Inner loop (attitude): attitude error → desired moments

    This is a P/PID approximation of LQR — full LQR requires computing the
    infinite-horizon gain matrix K for the linearised quadrotor model, which
    is typically pre-computed offline. The controller here uses the same
    state-space structure with tunable gains.
    """

    def __init__(self, config: Optional[LQRConfig] = None):
        self.cfg = config or LQRConfig()
        self.mixer = MotorMixing(arm_length=self.cfg.arm_length_m)
        self._last_command_time: int = 0

        # Pre-computed LQR gains (infinite-horizon for linearised model)
        # These would normally come from solving the CARE, but we use
        # hand-tuned equivalents for now
        self._K_pos = np.array([
            self.cfg.Q_xy,    self.cfg.Q_xy,    self.cfg.Q_z,     # position
            self.cfg.Q_vxy,   self.cfg.Q_vxy,   self.cfg.Q_vz,    # velocity
        ])
        self._K_att = np.array([
            self.cfg.Q_roll, self.cfg.Q_pitch, self.cfg.Q_yaw,
            self.cfg.Q_p, self.cfg.Q_q, self.cfg.Q_r,
        ])

    def compute_control(
        self,
        state: StateEstimate,
        waypoint: Waypoint,
    ) -> ControlCommand:
        """
        Compute motor commands from current state and target waypoint.

        Args:
            state: Current EKF state estimate (world-frame NED)
            waypoint: Desired position setpoint

        Returns:
            ControlCommand with motor values
        """
        # Gain scales
        Kp_xy = 0.8      # position → velocity proportional gain
        Kp_z = 1.2        # altitude gain (more aggressive for safety)
        Kd_xy = 0.4       # velocity gain
        Kd_z = 0.8

        Kp_roll = 6.0     # attitude P gains
        Kp_pitch = 6.0
        Kp_yaw = 3.0
        Kd_roll = 1.5     # rate D gains
        Kd_pitch = 1.5
        Kd_yaw = 1.0

        # ── Position error (world frame) ──
        ex = waypoint.x - state.x
        ey = waypoint.y - state.y
        ez = waypoint.z - state.z

        # ── Desired acceleration (outer loop) ──
        ax_des = Kp_xy * ex + Kd_xy * (0.0 - state.vx)
        ay_des = Kp_xy * ey + Kd_xy * (0.0 - state.vy)
        az_des = Kp_z * ez + Kd_z * (0.0 - state.vz) + self.cfg.gravity

        # ── Desired thrust (NED: down is positive) ──
        thrust_des = self.cfg.mass_kg * az_des
        thrust_norm = max(thrust_des / (4.0 * self.cfg.max_thrust_per_motor), 0.0)
        thrust_norm = min(thrust_norm, 1.0)

        # ── Desired attitude (from acceleration) ──
        # ax_des ≈ -g * sin(θ_des), ay_des ≈ g * sin(φ_des)
        roll_des = np.clip(
            np.arcsin(ay_des / self.cfg.gravity),
            -self.cfg.max_roll_rad, self.cfg.max_roll_rad
        )
        pitch_des = np.clip(
            -np.arcsin(ax_des / self.cfg.gravity),
            -self.cfg.max_pitch_rad, self.cfg.max_pitch_rad
        )
        yaw_des = waypoint.yaw

        # ── Attitude error ──
        e_roll = roll_des - state.roll
        e_pitch = pitch_des - state.pitch
        e_yaw = yaw_des - state.yaw
        # Wrap yaw to [-π, π]
        e_yaw = np.arctan2(np.sin(e_yaw), np.cos(e_yaw))

        # ── Body moments (inner loop) ──
        tau_phi = Kp_roll * e_roll   # roll moment
        tau_theta = Kp_pitch * e_pitch  # pitch moment
        tau_psi = Kp_yaw * e_yaw      # yaw moment

        # ── Saturate moments ──
        max_moment = 2.0  # N·m
        tau_phi = np.clip(tau_phi, -max_moment, max_moment)
        tau_theta = np.clip(tau_theta, -max_moment, max_moment)
        tau_psi = np.clip(tau_psi, -max_moment, max_moment)

        # ── Motor mixing ──
        motor_sq = self.mixer.mix(
            thrust=thrust_norm,
            tau_phi=tau_phi,
            tau_theta=tau_theta,
            tau_psi=tau_psi,
        )

        # Convert squared speeds to normalised PWM (0–1)
        motor_outputs = np.sqrt(np.clip(motor_sq, 0.0, 1.0))

        now = now_us()
        self._last_command_time = now

        return ControlCommand(
            timestamp_us=now,
            motor_1=float(motor_outputs[0]),
            motor_2=float(motor_outputs[1]),
            motor_3=float(motor_outputs[2]),
            motor_4=float(motor_outputs[3]),
        )

    def compute_attitude_only(
        self,
        state: StateEstimate,
        roll_des: float = 0.0,
        pitch_des: float = 0.0,
        yaw_des: float = 0.0,
        thrust_norm: float = 0.25,
    ) -> ControlCommand:
        """
        Direct attitude control (bypasses position loop).
        Useful for manual/stabilised flight modes.
        """
        # Attitude error
        e_roll = roll_des - state.roll
        e_pitch = pitch_des - state.pitch
        e_yaw = yaw_des - state.yaw
        e_yaw = np.arctan2(np.sin(e_yaw), np.cos(e_yaw))

        Kp = 6.0
        tau_phi = Kp * e_roll
        tau_theta = Kp * e_pitch
        tau_psi = 3.0 * e_yaw

        max_moment = 2.0
        tau_phi = np.clip(tau_phi, -max_moment, max_moment)
        tau_theta = np.clip(tau_theta, -max_moment, max_moment)
        tau_psi = np.clip(tau_psi, -max_moment, max_moment)

        motor_sq = self.mixer.mix(
            thrust=thrust_norm,
            tau_phi=tau_phi,
            tau_theta=tau_theta,
            tau_psi=tau_psi,
        )
        motor_outputs = np.sqrt(np.clip(motor_sq, 0.0, 1.0))

        return ControlCommand(
            timestamp_us=now_us(),
            motor_1=float(motor_outputs[0]),
            motor_2=float(motor_outputs[1]),
            motor_3=float(motor_outputs[2]),
            motor_4=float(motor_outputs[3]),
        )

    @staticmethod
    def compute_lqr_gains_offline(
        A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        """
        Solve the Continuous-time Algebraic Riccati Equation for infinite-horizon LQR.
        K = R⁻¹BᵀP where AᵀP + PA - PBR⁻¹BᵀP + Q = 0

        Uses iterative method (Kleinman) when scipy is not available.
        """
        try:
            from scipy.linalg import solve_continuous_are
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.solve(R, B.T @ P)
            return K
        except ImportError:
            # Kleinman's algorithm for CARE
            n = A.shape[0]
            m = B.shape[1]
            P = Q.copy()
            K = np.zeros((m, n))

            for _ in range(100):
                K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
                A_cl = A - B @ K
                P_next = Q + K.T @ R @ K + A_cl.T @ P @ A_cl
                if np.allclose(P, P_next, atol=1e-10):
                    break
                P = P_next

            return K