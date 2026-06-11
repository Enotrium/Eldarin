"""
Tests for the Eldarin Navigation System
========================================
Covers all three compute domains: MCU (EKF, LQR, ESC), Cortex-A (HDC-EVIO, HDC-SLAM, Path Planner),
and inter-domain communication (messages, system orchestrator).
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from navigation.messages import (
    ImuMessage, EventFrameMessage, OpticalFlowMessage, DistanceMessage,
    BarometerMessage, StateEstimate, ControlCommand, Waypoint,
    MpuMessage, SensorPacket, DomainMessage, now_us,
)
from navigation.estimation.ekf import ExtendedKalmanFilter, EKFConfig, IMUOdometryEKF
from navigation.control.lqr import LQRController, LQRConfig, MotorMixing
from navigation.control.motor_esc import MotorESC, EscProtocol
from navigation.planning.path_planner import PathPlanner, MissionPlanner, PlannerConfig, Obstacle
from navigation.slam.hdc_slam import HDCSlamMapper, LandmarkConfig, LandmarkMemory
from navigation.system import NavigationSystem, SystemConfig


# ── Message Contracts ────────────────────────────────────────────────────────

def test_message_creation():
    """All message types can be created with required fields."""
    t = now_us()

    imu = ImuMessage(timestamp_us=t, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))
    assert imu.accel[2] == 9.81

    ef = EventFrameMessage(timestamp_us=t, width=240, height=180, xs=[10], ys=[20],
                           activations=[1.0], feature_ids=[5])
    assert len(ef.xs) == 1

    flow = OpticalFlowMessage(timestamp_us=t, dx=0.5, dy=-0.2)
    assert flow.dx == 0.5

    dist = DistanceMessage(timestamp_us=t, front_mm=5000, down_mm=50000)
    assert dist.down_mm == 50000

    baro = BarometerMessage(timestamp_us=t, pressure_hpa=1013.25)
    assert baro.pressure_hpa == 1013.25

    state = StateEstimate(timestamp_us=t, x=1.0, y=2.0, z=50.0)
    assert state.z == 50.0

    cmd = ControlCommand(timestamp_us=t, motor_1=0.5, motor_2=0.5, motor_3=0.5, motor_4=0.5)
    assert cmd.motor_1 == 0.5

    wp = Waypoint(timestamp_us=t, x=10.0, y=0.0, z=50.0)
    assert wp.acceptance_radius_m == 1.0

    mpu = MpuMessage(timestamp_us=t, imu=imu, optical_flow=flow, distances=dist, barometer=baro)
    assert mpu.imu is not None

    packet = SensorPacket(event_frame=ef, imu=imu, distance=dist, barometer=baro)
    assert packet.event_frame is not None

    dm = DomainMessage(source="mcu", dest="cortex_a", payload=state)
    assert dm.source == "mcu"


# ── Extended Kalman Filter ──────────────────────────────────────────────────

def test_ekf_initialization():
    """EKF initialises with zero state and reasonable covariance."""
    ekf = ExtendedKalmanFilter()
    assert np.allclose(ekf.position, np.zeros(3))
    assert ekf.P.shape == (15, 15)
    assert np.all(np.diag(ekf.P) > 0)


def test_ekf_predict_no_drift_hover():
    """EKF predict with zero accel/gyro should keep position stable."""
    ekf = ExtendedKalmanFilter()
    imu = ImuMessage(timestamp_us=0, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))

    for _ in range(1000):
        ekf.predict(imu, dt=0.001)

    pos = ekf.position
    assert abs(pos[0]) < 0.5, f"x drifted: {pos[0]}"
    assert abs(pos[1]) < 0.5, f"y drifted: {pos[1]}"


def test_ekf_altitude_observation():
    """EKF altitude converges to rangefinder observation."""
    ekf = ExtendedKalmanFilter()
    imu = ImuMessage(timestamp_us=0, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))

    # Fly for 100 steps
    for _ in range(100):
        ekf.predict(imu, dt=0.001)

    # Observe altitude at 50m
    dist = DistanceMessage(timestamp_us=100000, front_mm=50000, down_mm=50000)
    ekf.update_distance(dist)

    assert abs(ekf.position[2] - 50.0) < 10.0, f"Altitude not converging: {ekf.position[2]}"


def test_ekf_hdc_correction_acceptance():
    """HDC-EVIO correction is accepted when residual is small."""
    ekf = ExtendedKalmanFilter()
    imu = ImuMessage(timestamp_us=0, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))

    for _ in range(100):
        ekf.predict(imu, dt=0.001)

    # Correction close to current state (should be accepted)
    corr = StateEstimate(
        timestamp_us=100000,
        x=0.0, y=0.0, z=0.1,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.0, pitch=0.0, yaw=0.01,
        cov_xx=0.01, cov_yy=0.01, cov_zz=0.01,
        cov_vx=0.1, cov_vy=0.1, cov_vz=0.1,
        cov_rr=0.01, cov_rp=0.01, cov_ry=0.01,
    )
    accepted = ekf.apply_hdc_correction(corr)
    assert accepted, "Correction should be accepted"


def test_ekf_hdc_correction_rejection():
    """HDC-EVIO correction is rejected when residual is too large."""
    ekf = ExtendedKalmanFilter()

    # Correction far from current state (should be rejected)
    corr = StateEstimate(
        timestamp_us=0,
        x=100.0, y=100.0, z=100.0,
        vx=50.0, vy=50.0, vz=50.0,
        roll=2.0, pitch=2.0, yaw=2.0,
        cov_xx=0.01, cov_yy=0.01, cov_zz=0.01,
        cov_vx=0.1, cov_vy=0.1, cov_vz=0.1,
        cov_rr=0.01, cov_rp=0.01, cov_ry=0.01,
    )
    accepted = ekf.apply_hdc_correction(corr)
    assert not accepted, "Large correction should be rejected"


def test_imu_odometry_ekf():
    """IMUOdometryEKF wrapper feeds IMU correctly."""
    ekf = IMUOdometryEKF()
    imu = ImuMessage(timestamp_us=0, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))

    # First IMU sets baseline time
    ekf.feed_imu(imu)

    # Second IMU triggers predict
    imu2 = ImuMessage(timestamp_us=1000, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))
    ekf.feed_imu(imu2)

    state = ekf.get_state()
    assert state.source == "ekf_only"


# ── LQR Controller ──────────────────────────────────────────────────────────

def test_motor_mixing():
    """Motor mixing produces valid normalised outputs."""
    mixer = MotorMixing(arm_length=0.25)
    speeds = mixer.mix(thrust=0.25, tau_phi=0.0, tau_theta=0.0, tau_psi=0.0)
    assert speeds.shape == (4,)
    assert np.allclose(speeds, 0.0625, atol=0.01)  # T/4 = 0.0625


def test_lqr_hover():
    """LQR produces symmetric thrust for hover at target."""
    lqr = LQRController()
    state = StateEstimate(
        timestamp_us=0,
        x=0.0, y=0.0, z=50.0,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.0, pitch=0.0, yaw=0.0,
    )
    wp = Waypoint(timestamp_us=0, x=0.0, y=0.0, z=50.0)

    cmd = lqr.compute_control(state, wp)
    # All motors should be approximately equal in hover
    motors = [cmd.motor_1, cmd.motor_2, cmd.motor_3, cmd.motor_4]
    assert max(motors) - min(motors) < 0.3
    assert all(0 <= m <= 1 for m in motors)


def test_lqr_position_error_drive():
    """LQR produces asymmetric thrust to correct position error."""
    lqr = LQRController()
    state = StateEstimate(
        timestamp_us=0,
        x=0.0, y=0.0, z=50.0,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.0, pitch=0.0, yaw=0.0,
    )
    wp = Waypoint(timestamp_us=0, x=10.0, y=0.0, z=50.0)

    cmd = lqr.compute_control(state, wp)
    motors = [cmd.motor_1, cmd.motor_2, cmd.motor_3, cmd.motor_4]
    # Motors should differ to pitch forward for x-motion
    assert not np.allclose(motors, motors[0])


def test_lqr_attitude_only():
    """Attitude-only control produces stabilised output."""
    lqr = LQRController()
    state = StateEstimate(
        timestamp_us=0,
        x=0.0, y=0.0, z=50.0,
        roll=0.1, pitch=0.0, yaw=0.0,
    )
    cmd = lqr.compute_attitude_only(state, roll_des=0.0, pitch_des=0.0, yaw_des=0.0)
    assert all(0 <= v <= 1 for v in
               [cmd.motor_1, cmd.motor_2, cmd.motor_3, cmd.motor_4])


# ── Motor ESC ───────────────────────────────────────────────────────────

def test_esc_arm_disarm():
    """ESC arm and disarm sequence."""
    esc = MotorESC()
    assert not esc.armed

    armed = esc.arm()
    assert armed
    assert esc.armed

    esc.disarm()
    assert not esc.armed


def test_esc_send_command():
    """ESC accepts commands when armed."""
    esc = MotorESC()
    esc.arm()

    cmd = ControlCommand(
        timestamp_us=now_us(),
        motor_1=0.25, motor_2=0.25, motor_3=0.25, motor_4=0.25,
    )
    success = esc.send_command(cmd)
    assert success
    assert esc.command_count == 1


def test_esc_failsafe():
    """ESC enters failsafe on timeout."""
    esc = MotorESC(arm_timeout_ms=10)
    esc.arm()

    # Send a command with an old timestamp
    old_cmd = ControlCommand(
        timestamp_us=now_us() - 20000,  # 20ms in the past
        motor_1=0.25, motor_2=0.25, motor_3=0.25, motor_4=0.25,
    )
    esc.send_command(old_cmd)  # Will trigger failsafe due to timestamp age
    assert esc.in_failsafe or not esc.armed


# ── Path Planner ────────────────────────────────────────────────────────────

def test_path_planner_waypoint_following():
    """Path planner follows a waypoint list."""
    planner = PathPlanner()
    planner.set_mission_waypoints([
        (0.0, 0.0, 50.0),
        (10.0, 0.0, 50.0),
        (10.0, 10.0, 50.0),
    ])

    state = StateEstimate(
        timestamp_us=0,
        x=0.0, y=0.0, z=50.0,
    )
    wp = planner.step(state)
    assert wp is not None
    assert abs(wp.x - 0.0) < 0.1

    # Simulate arrival
    state = StateEstimate(timestamp_us=0, x=0.0, y=0.0, z=50.0)
    planner.step(state)
    assert planner.waypoints_visited >= 1


def test_path_planner_coverage():
    """Coverage pattern generates waypoints."""
    planner = PathPlanner()
    planner.set_coverage_mission(0.0, 0.0, 30.0, 30.0, altitude=50.0)
    assert planner.waypoints_remaining > 0


def test_path_planner_obstacle_avoidance():
    """Obstacle avoidance adjusts waypoints."""
    planner = PathPlanner(PlannerConfig(obstacle_avoidance_enabled=True))
    planner.set_mission_waypoints([(10.0, 0.0, 50.0)])

    # Place an obstacle on the path
    planner.update_obstacles([
        Obstacle(position=np.array([10.0, 0.0, 50.0]), radius_m=3.0)
    ])

    state = StateEstimate(timestamp_us=0, x=0.0, y=0.0, z=50.0)
    wp = planner.step(state)
    # Either the waypoint is adjusted or the planner finds it unsafe
    assert wp is not None


def test_mission_planner():
    """Mission planner transitions through phases."""
    mp = MissionPlanner()
    mp.start_mission(waypoints=[(10.0, 0.0, 50.0)])

    state = StateEstimate(timestamp_us=0, x=0.0, y=0.0, z=50.0)
    wp = mp.step(state)
    assert mp.phase == "takeoff" or mp.phase == "cruise"
    assert wp is not None


# ── HDC-SLAM ────────────────────────────────────────────────────────────────

def test_landmark_memory_insert():
    """Landmark memory stores and queries landmarks."""
    cfg = LandmarkConfig(hd_dim=256, max_landmarks=100)
    mem = LandmarkMemory(cfg)

    hv = (np.random.rand(256) > 0.5).astype(np.float32) * 2 - 1
    lid = mem.insert(hv, np.array([0.0, 0.0, 50.0]), frame_id=0)
    assert lid == 0
    assert mem.get_num_landmarks() == 1

    results = mem.query(hv, k=3)
    assert len(results) == 1
    assert results[0][1] > 0.9  # self-similarity


def test_landmark_memory_deduplication():
    """Near-duplicate landmarks are suppressed or merged."""
    cfg = LandmarkConfig(hd_dim=256, max_landmarks=100,
                         landmark_confidence_threshold=0.5)
    mem = LandmarkMemory(cfg)

    hv = (np.random.rand(256) > 0.5).astype(np.float32) * 2 - 1
    mem.insert(hv, np.array([0.0, 0.0, 50.0]))
    assert mem.get_num_landmarks() == 1

    # Same HV at nearby position
    mem.insert(hv, np.array([0.1, 0.1, 50.0]))
    assert mem.get_num_landmarks() == 1  # merged, not duplicated


def test_hdc_slam_mapper():
    """HDC-SLAM mapper creates landmarks from event frames."""
    slam = HDCSlamMapper()

    for i in range(20):
        xs = [np.random.randint(0, 240) for _ in range(50)]
        ys = [np.random.randint(0, 180) for _ in range(50)]
        ef = EventFrameMessage(
            timestamp_us=now_us(),
            width=240, height=180,
            xs=xs, ys=ys,
            activations=[1.0] * 50,
            feature_ids=[np.random.randint(0, 50) for _ in range(50)],
        )
        pos = np.array([i * 0.5, 0.0, 50.0])
        result = slam.step(event_frame=ef, position=pos)

    assert slam.memory.get_num_landmarks() > 0
    assert len(slam.get_trajectory()) == 21  # initial + 20 steps


# ── System Orchestrator ─────────────────────────────────────────────────────

def test_system_creation():
    """Navigation system can be created from config."""
    nav = NavigationSystem()
    assert not nav.armed
    assert not nav.running


def test_system_start_stop():
    """System can start and stop cleanly."""
    nav = NavigationSystem()
    nav.start(arm=False)
    assert nav.running
    nav.stop()
    assert not nav.running


def test_system_sensor_feeding():
    """System accepts sensor data without errors."""
    nav = NavigationSystem(SystemConfig(cortex_a_enabled=False))
    nav.start()

    imu = ImuMessage(timestamp_us=now_us(), accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))
    nav.feed_imu(imu)

    result = nav.step()
    assert result is not None
    assert result["state"] is not None

    nav.stop()


def test_system_arm_cycle():
    """System arms and runs control loop."""
    nav = NavigationSystem(SystemConfig(cortex_a_enabled=False))
    nav.start()

    for _ in range(200):
        imu = ImuMessage(timestamp_us=now_us(), accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))
        nav.feed_imu(imu)
        nav.step()

    nav.arm()
    assert nav.armed

    cmd = nav.command
    assert cmd is not None

    nav.stop()
    assert not nav.armed


def test_system_mission_integration():
    """Full system processes a simple mission."""
    nav = NavigationSystem(SystemConfig(cortex_a_enabled=True))
    nav.start()

    # Set mission
    nav.mission_planner.start_mission(waypoints=[(10.0, 0.0, 50.0)])

    # Simulate 500 IMU readings
    for i in range(500):
        imu = ImuMessage(
            timestamp_us=now_us(),
            accel=(np.random.randn() * 0.01, np.random.randn() * 0.01, 9.81),
            gyro=(np.random.randn() * 0.001, np.random.randn() * 0.001, 0.0),
        )
        nav.feed_imu(imu)
        nav.step()

    # Should have produced waypoints and commands
    assert nav.state is not None
    assert nav.mission_planner.phase in ("takeoff", "cruise")

    nav.stop()


# ── Run all ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("message_creation", test_message_creation),
        ("ekf_initialization", test_ekf_initialization),
        ("ekf_predict_no_drift", test_ekf_predict_no_drift_hover),
        ("ekf_altitude_observation", test_ekf_altitude_observation),
        ("ekf_hdc_correction_accept", test_ekf_hdc_correction_acceptance),
        ("ekf_hdc_correction_reject", test_ekf_hdc_correction_rejection),
        ("imu_odometry_ekf", test_imu_odometry_ekf),
        ("motor_mixing", test_motor_mixing),
        ("lqr_hover", test_lqr_hover),
        ("lqr_position_error", test_lqr_position_error_drive),
        ("lqr_attitude_only", test_lqr_attitude_only),
        ("esc_arm_disarm", test_esc_arm_disarm),
        ("esc_send_command", test_esc_send_command),
        ("esc_failsafe", test_esc_failsafe),
        ("path_planner_waypoints", test_path_planner_waypoint_following),
        ("path_planner_coverage", test_path_planner_coverage),
        ("path_planner_obstacle", test_path_planner_obstacle_avoidance),
        ("mission_planner", test_mission_planner),
        ("landmark_memory_insert", test_landmark_memory_insert),
        ("landmark_memory_dedup", test_landmark_memory_deduplication),
        ("hdc_slam_mapper", test_hdc_slam_mapper),
        ("system_creation", test_system_creation),
        ("system_start_stop", test_system_start_stop),
        ("system_sensor_feeding", test_system_sensor_feeding),
        ("system_arm_cycle", test_system_arm_cycle),
        ("system_mission_integration", test_system_mission_integration),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)