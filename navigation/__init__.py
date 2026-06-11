"""
Eldarin Navigation System
=========================
GPS-denied autonomous drone navigation stack using neuromorphic processing
(event cameras + spiking neural networks) and Hyperdimensional Computing (HDC).

System Architecture (from outline):
  ┌────────────────────────────────────────────────────────────────┐
  │  Inner Loop (MCU)     │ Neuromorphic Perception (FPGA)         │
  │  • EKF sensor fusion  │ • SNN feature tracking                 │
  │  • LQR motor control  │ • Event-stream processing              │
  │  • Motor ESC interface│ • (optional) hypervector encoding      │
  │  Hard real-time kHz   │ Event-driven, asynchronous, sparse     │
  ├───────────────────────┴────────────────────────────────────────┤
  │  Mission/State Layer (Cortex-A, ROS2, NEON SIMD)               │
  │  • HDC-EVIO — ego-motion estimation from HD vectors            │
  │  • HDC-SLAM — compact associative map of landmarks             │
  │  • Path Planning — payload-adaptive waypoint generation        │
  │  • State corrections → MCU EKF                                 │
  │  Soft real-time, 10s–100s of Hz                                │
  └────────────────────────────────────────────────────────────────┘

Design principle: wherever a conventional pipeline would use a GPU-intensive
algorithm, substitute a sparse, event-driven, or hyperdimensional equivalent.
"""

from .system import NavigationSystem
from .messages import (
    ImuMessage, EventFrameMessage, StateEstimate, ControlCommand,
    Waypoint, MpuMessage, SensorPacket, DomainMessage,
)

# Estimation (MCU inner loop + Cortex-A HDC-EVIO)
from .estimation.ekf import ExtendedKalmanFilter, IMUOdometryEKF
from .estimation.hdc_evio import HDCEvioEstimator

# SLAM
from .slam.hdc_slam import HDCSlamMapper, LandmarkMemory

# Control (MCU inner loop)
from .control.lqr import LQRController, MotorMixing
from .control.motor_esc import MotorESC, EscProtocol

# Planning (Cortex-A)
from .planning.path_planner import PathPlanner, MissionPlanner

__all__ = [
    # System
    "NavigationSystem",
    # Messages
    "ImuMessage", "EventFrameMessage", "StateEstimate", "ControlCommand",
    "Waypoint", "MpuMessage", "SensorPacket", "DomainMessage",
    # Estimation
    "ExtendedKalmanFilter", "IMUOdometryEKF", "HDCEvioEstimator",
    # SLAM
    "HDCSlamMapper", "LandmarkMemory",
    # Control
    "LQRController", "MotorMixing", "MotorESC", "EscProtocol",
    # Planning
    "PathPlanner", "MissionPlanner",
]