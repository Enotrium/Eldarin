"""
Navigation System Orchestrator
================================
Top-level orchestrator that wires together all three compute domains
(MCU, FPGA, Cortex-A) per the outline architecture.

     ┌─────────────────────────────────────────────────────────┐
     │                 NavigationSystem                        │
     │                                                         │
     │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
     │  │  MCU Domain  │  │ FPGA Domain  │  │ Cortex-A      │  │
     │  │  • EKF       │  │ • SNN        │  │ • HDC-EVIO    │  │
     │  │  • LQR       │  │ • Event Encode│  │ • HDC-SLAM    │  │
     │  │  • Motor ESC │  │              │  │ • Path Planner│  │
     │  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘  │
     │         │                 │                  │           │
     │         └─────────────────┴──────────────────┘           │
     │                    SensorPacket bus                       │
     │               StateEstimate ↔ ControlCommand              │
     └─────────────────────────────────────────────────────────┘

Safety rule (§2.1): The MCU inner loop keeps the vehicle stable using
only IMU + optical flow + distance sensors, even if FPGA or Cortex-A
degrade. Higher layers improve the state estimate and provide goals.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import threading

from .messages import (
    ImuMessage, EventFrameMessage, OpticalFlowMessage, DistanceMessage,
    BarometerMessage, StateEstimate, ControlCommand, Waypoint,
    MpuMessage, SensorPacket, DomainMessage, now_us,
)
from .estimation.ekf import IMUOdometryEKF, EKFConfig
from .estimation.hdc_evio import HDCEvioEstimator, HDCEvioConfig
from .slam.hdc_slam import HDCSlamMapper, LandmarkConfig
from .control.lqr import LQRController, LQRConfig
from .control.motor_esc import MotorESC, EscProtocol
from .planning.path_planner import MissionPlanner, PlannerConfig

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Top-level configuration for the full navigation system."""
    # Domain enable/disable
    mcu_enabled: bool = True
    fpga_enabled: bool = True      # SNN feature tracking (stubbed when False)
    cortex_a_enabled: bool = True  # HDC-EVIO, HDC-SLAM, planning

    # Loop rates
    mcu_inner_loop_hz: float = 1000.0       # EKF + LQR
    cortex_a_evio_hz: float = 20.0           # HDC-EVIO correction rate
    cortex_a_slam_hz: float = 10.0           # HDC-SLAM mapping rate
    cortex_a_planning_hz: float = 10.0       # Path planning rate

    # Safety
    mcu_can_stabilize_alone: bool = True     # §2.1 invariant
    cortex_a_timeout_ms: int = 500           # use EKF-only if Cortex-A silent
    fpga_timeout_ms: int = 500               # use optical-flow-only if FPGA silent

    # Domain configs (sub-configs)
    ekf: EKFConfig = None
    hdc_evio: HDCEvioConfig = None
    hdc_slam: LandmarkConfig = None
    lqr: LQRConfig = None
    planner: PlannerConfig = None

    # ESC
    esc_protocol: str = "dshot600"


class NavigationSystem:
    """
    Full navigation system orchestrator running the three-domain architecture.

    Thread model:
      - MCU thread (hard real-time): EKF predict + LQR compute + ESC send
      - Cortex-A thread (soft real-time): HDC-EVIO, HDC-SLAM, Path Planning
      - FPGA is external, feeding SensorPackets asynchronously

    Usage:
        nav = NavigationSystem(config)
        nav.start()
        # ... sensor data arrives ...
        nav.feed_sensors(sensor_packet)
        nav.step()  # run one full iteration
        nav.stop()
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.cfg = config or SystemConfig()

        # ── MCU inner loop ──
        self.ekf = IMUOdometryEKF(self.cfg.ekf)
        self.lqr = LQRController(self.cfg.lqr)
        self.esc = MotorESC(
            protocol=EscProtocol.DSHOT600,
            arm_timeout_ms=self.cfg.cortex_a_timeout_ms * 2,
        )

        # ── Cortex-A perception ──
        self.hdc_evio = HDCEvioEstimator(self.cfg.hdc_evio, device="cpu")
        self.hdc_slam = HDCSlamMapper(self.cfg.hdc_slam)

        # ── Cortex-A planning ──
        self.mission_planner = MissionPlanner(self.cfg.planner)

        # ── State ──
        self._running: bool = False
        self._armed: bool = False
        self._last_cortex_correction_us: int = 0
        self._last_fpga_frame_us: int = 0
        self._iteration_count: int = 0

        # Message logging (ring buffer)
        self._state_history: deque = deque(maxlen=10000)
        self._command_history: deque = deque(maxlen=10000)
        self._slam_landmarks: List = []

        # Latest state
        self._latest_state: Optional[StateEstimate] = None
        self._latest_command: Optional[ControlCommand] = None
        self._latest_waypoint: Optional[Waypoint] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self, arm: bool = False):
        """Initialise and start the navigation system."""
        logger.info("NavigationSystem starting...")
        self._running = True

        if arm:
            self.arm()

    def stop(self):
        """Stop the navigation system, disarm, and clean up."""
        logger.info("NavigationSystem stopping...")
        self.disarm()
        self._running = False

    def arm(self) -> bool:
        """Arm the vehicle."""
        if not self._armed:
            logger.info("Arming ESCs...")
            self._armed = self.esc.arm()
            if self._armed:
                logger.info("Vehicle armed.")
        return self._armed

    def disarm(self):
        """Disarm the vehicle immediately."""
        if self._armed:
            logger.info("Disarming ESCs...")
            self.esc.disarm(immediate=True)
            self._armed = False
            logger.info("Vehicle disarmed.")

    # ── Sensor input ─────────────────────────────────────────────────────────

    def feed_imu(self, imu: ImuMessage):
        """Feed a 1 kHz IMU reading to the EKF."""
        self.ekf.feed_imu(imu)

    def feed_optical_flow(self, flow: OpticalFlowMessage):
        """Feed optical flow to the EKF."""
        self.ekf.feed_optical_flow(flow)

    def feed_distance(self, dist: DistanceMessage):
        """Feed rangefinder to the EKF."""
        self.ekf.feed_distance(dist)

    def feed_barometer(self, baro: BarometerMessage):
        """Feed barometer to the EKF."""
        self.ekf.feed_barometer(baro)

    def feed_event_frame(self, ef: EventFrameMessage):
        """Receive an event frame from the FPGA SNN tracker."""
        self._last_fpga_frame_us = ef.timestamp_us

        # Build a SensorPacket for the Cortex-A perception pipeline
        state = self._latest_state
        packet = SensorPacket(
            event_frame=ef,
            imu=ImuMessage(
                timestamp_us=ef.timestamp_us,
                accel=(0.0, 0.0, 0.0),
                gyro=(0.0, 0.0, 0.0),
            ),
            optical_flow=None,
            distance=None,
            barometer=None,
        )
        self._process_event_packet(packet)

    def feed_mpu_packet(self, mpu: MpuMessage):
        """Receive aggregated MCU sensor packet."""
        if mpu.imu:
            self.feed_imu(mpu.imu)
        if mpu.optical_flow:
            self.feed_optical_flow(mpu.optical_flow)
        if mpu.distances:
            self.feed_distance(mpu.distances)
        if mpu.barometer:
            self.feed_barometer(mpu.barometer)

    def _process_event_packet(self, packet: SensorPacket):
        """Run the Cortex-A perception pipeline on a new event frame."""
        if not self.cfg.cortex_a_enabled:
            return

        try:
            # HDC-EVIO step
            evio_result = self.hdc_evio.step(packet)

            # Apply HDC-EVIO correction to EKF if available
            correction = evio_result.get("correction")
            if correction is not None:
                accepted = self.ekf.feed_hdc_correction(correction)
                if accepted:
                    self._last_cortex_correction_us = correction.timestamp_us

            # HDC-SLAM step (share position from EVIO)
            slam_result = self.hdc_slam.step(
                event_frame=packet.event_frame,
                distance=packet.distance,
                barometer=packet.barometer,
                position=evio_result.get("position"),
            )

            # Collect SLAM landmarks for planning
            self._slam_landmarks = self.hdc_slam.get_map().tolist()

        except Exception as e:
            logger.error(f"Cortex-A perception error: {e}")

    # ── Main loop iteration ──────────────────────────────────────────────────

    def step(self) -> Dict[str, Any]:
        """
        Run one full iteration of the navigation system.
        Should be called at the MCU inner-loop rate (~1 kHz).

        Returns a dict with the current system state for monitoring.
        """
        self._iteration_count += 1

        # ── 1. Get EKF state ──
        ekf_state = self.ekf.get_state()

        # Cortex-A timeout check
        now = now_us()
        if self.cfg.cortex_a_enabled:
            elapsed = now - self._last_cortex_correction_us
            if elapsed > self.cfg.cortex_a_timeout_ms * 1000:
                ekf_state.source = "ekf_only"  # flag that HDC-EVIO is stale
                if self._latest_state is not None and self._latest_state.source == "hdc_evio":
                    logger.warning("HDC-EVIO timeout — falling back to EKF-only")

        self._latest_state = ekf_state
        self._state_history.append(ekf_state)

        # ── 2. Path planning (Cortex-A, at reduced rate) ──
        wp: Optional[Waypoint] = None
        if self.cfg.cortex_a_enabled and self._iteration_count % max(
            int(self.cfg.mcu_inner_loop_hz / self.cfg.cortex_a_planning_hz), 1
        ) == 0:
            try:
                slam_map = (
                    self.hdc_slam.get_map()
                    if self.hdc_slam.memory.get_num_landmarks() > 0
                    else None
                )
                wp = self.mission_planner.step(ekf_state, slam_map)
                if wp is not None:
                    self._latest_waypoint = wp
            except Exception as e:
                logger.error(f"Path planning error: {e}")

        # ── 3. LQR control ──
        if wp is not None:
            cmd = self.lqr.compute_control(ekf_state, wp)
        elif self._latest_waypoint is not None:
            # Recompute against last known waypoint
            cmd = self.lqr.compute_control(ekf_state, self._latest_waypoint)
        else:
            # Stabilise in place (attitude hold, hover thrust)
            cmd = self.lqr.compute_attitude_only(
                ekf_state,
                roll_des=0.0,
                pitch_des=0.0,
                yaw_des=0.0,
                thrust_norm=self.cfg.lqr.hover_thrust if self.cfg.lqr else 0.25,
            )

        self._latest_command = cmd
        self._command_history.append(cmd)

        # ── 4. Send to ESCs (if armed) ──
        if self._armed:
            if not self.esc.in_failsafe:
                self.esc.send_command(cmd)
            self.esc.check_failsafe()

        return {
            "iteration": self._iteration_count,
            "state": ekf_state,
            "command": cmd,
            "waypoint": wp,
            "armed": self._armed,
            "mode": self.mission_planner.phase if self.cfg.cortex_a_enabled else "mcu_only",
            "esc_failsafe": self.esc.in_failsafe,
            "hdc_evio_initialized": self.hdc_evio._map_initialized if self.cfg.cortex_a_enabled else False,
            "slam_landmarks": self.hdc_slam.memory.get_num_landmarks() if self.cfg.cortex_a_enabled else 0,
        }

    def run_with_sensor_stream(
        self,
        sensor_generator,
        max_iterations: int = 0,
        arm_after: int = 100,
    ):
        """
        Run the navigation system against a sensor data stream.

        Args:
            sensor_generator: Iterator yielding SensorPacket or MpuMessage objects
            max_iterations: Stop after N iterations (0 = infinite)
            arm_after: Auto-arm after N iterations
        """
        self.start(arm=False)

        for i, sensor_data in enumerate(sensor_generator):
            if not self._running:
                break

            # Feed sensors
            if isinstance(sensor_data, MpuMessage):
                self.feed_mpu_packet(sensor_data)
            elif isinstance(sensor_data, SensorPacket):
                self._process_event_packet(sensor_data)
            elif isinstance(sensor_data, EventFrameMessage):
                self.feed_event_frame(sensor_data)

            # Auto-arm
            if not self._armed and i >= arm_after:
                self.arm()

            # Step
            result = self.step()

            if max_iterations > 0 and i >= max_iterations:
                break

        self.stop()

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def state(self) -> Optional[StateEstimate]:
        return self._latest_state

    @property
    def command(self) -> Optional[ControlCommand]:
        return self._latest_command

    @property
    def waypoint(self) -> Optional[Waypoint]:
        return self._latest_waypoint

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def running(self) -> bool:
        return self._running

    def get_telemetry(self) -> Dict[str, Any]:
        """Aggregated telemetry for ground-station / logging."""
        return {
            "state": self._latest_state,
            "command": self._latest_command,
            "waypoint": self._latest_waypoint,
            "armed": self._armed,
            "iteration": self._iteration_count,
            "esc_telemetry": self.esc.read_telemetry(),
        }


def create_navigation_system(
    config_path: Optional[str] = None,
    **overrides,
) -> NavigationSystem:
    """
    Factory function to create a NavigationSystem from config.

    Args:
        config_path: Path to YAML config (e.g., config/navigation.yaml)
        **overrides: Override config values
    """
    config = SystemConfig()

    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)

        # Map YAML to SystemConfig sub-configs
        if yaml_cfg:
            ekf_cfg = yaml_cfg.get("ekf", {})
            if ekf_cfg:
                config.ekf = EKFConfig(**ekf_cfg)

            evio_cfg = yaml_cfg.get("hdc_evio", {})
            if evio_cfg:
                config.hdc_evio = HDCEvioConfig(**evio_cfg)

            slam_cfg = yaml_cfg.get("hdc_slam", {})
            if slam_cfg:
                config.hdc_slam = LandmarkConfig(**slam_cfg)

            lqr_cfg = yaml_cfg.get("lqr", {})
            if lqr_cfg:
                config.lqr = LQRConfig(**lqr_cfg)

            planner_cfg = yaml_cfg.get("planner", {})
            if planner_cfg:
                config.planner = PlannerConfig(**planner_cfg)

            # Top-level overrides
            for key in ["mcu_enabled", "fpga_enabled", "cortex_a_enabled",
                         "mcu_inner_loop_hz", "cortex_a_evio_hz",
                         "cortex_a_slam_hz", "cortex_a_planning_hz",
                         "cortex_a_timeout_ms", "fpga_timeout_ms"]:
                if key in yaml_cfg:
                    setattr(config, key, yaml_cfg[key])

    # Apply CLI overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return NavigationSystem(config)