#!/usr/bin/env python3
"""
Eldarin Navigation System — Main Entry Point
==============================================
GPS-denied autonomous drone navigation using neuromorphic processing
(event cameras + spiking neural networks) and Hyperdimensional Computing (HDC).

Three compute domains per the system architecture:
  1. MCU Inner Loop — EKF sensor fusion + LQR motor control (hard real-time kHz)
  2. FPGA Neuromorphic Perception — SNN feature tracking (event-driven, sparse)
  3. Cortex-A Mission Layer — HDC-EVIO + HDC-SLAM + Path Planning (soft real-time)

Usage:
    # Run with config
    python main_navigation.py --config config/navigation.yaml

    # Simulate with synthetic sensor data
    python main_navigation.py --simulate --duration 60

    # Run HDC-EVIO benchmark (Renner et al. 2024 VO pipeline)
    python main_navigation.py --benchmark_evio --input /path/to/events.npy

    # Test individual components
    python main_navigation.py --test_ekf
    python main_navigation.py --test_lqr
    python main_navigation.py --test_slam
"""

import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from navigation.system import NavigationSystem, create_navigation_system
from navigation.messages import (
    ImuMessage, EventFrameMessage, OpticalFlowMessage, DistanceMessage,
    BarometerMessage, Waypoint, now_us, MpuMessage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main_navigation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eldarin Navigation System — Autonomous Neuromorphic Drone Navigation"
    )

    # Config
    parser.add_argument("--config", type=str, default="config/navigation.yaml",
                        help="Path to navigation YAML configuration")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for logs and telemetry")

    # Simulation
    parser.add_argument("--simulate", action="store_true",
                        help="Run simulated navigation with synthetic sensor data")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Simulation duration in seconds")

    # Benchmarks
    parser.add_argument("--benchmark_evio", action="store_true",
                        help="Benchmark HDC-EVIO pipeline")
    parser.add_argument("--benchmark_slam", action="store_true",
                        help="Benchmark HDC-SLAM pipeline")
    parser.add_argument("--input", type=str, default=None,
                        help="Input event data file for benchmarks")

    # Component tests
    parser.add_argument("--test_ekf", action="store_true",
                        help="Run EKF self-test")
    parser.add_argument("--test_lqr", action="store_true",
                        help="Run LQR controller self-test")
    parser.add_argument("--test_slam", action="store_true",
                        help="Run HDC-SLAM self-test")
    parser.add_argument("--test_all", action="store_true",
                        help="Run all component tests")

    # Mission
    parser.add_argument("--mission", type=str, default=None,
                        help="Mission type: waypoint, coverage, loiter")
    parser.add_argument("--waypoints", type=str, default=None,
                        help="Comma-separated waypoints: x1,y1,z1;x2,y2,z2;...")
    parser.add_argument("--coverage_area", type=str, default=None,
                        help="Coverage area: x_min,y_min,x_max,y_max")

    return parser.parse_args()


# ── Synthetic sensor generators ──────────────────────────────────────────────

def synthetic_imu_stream(duration_s: float, rate_hz: float = 1000.0):
    """Generate synthetic IMU data (hover + slight drift + noise)."""
    dt = 1.0 / rate_hz
    n = int(duration_s * rate_hz)
    t0 = now_us()

    for i in range(n):
        t = i * dt
        yield ImuMessage(
            timestamp_us=t0 + int(t * 1e6),
            accel=(np.random.randn() * 0.01,
                   np.random.randn() * 0.01,
                   9.81 + np.random.randn() * 0.05),
            gyro=(np.random.randn() * 0.001,
                  np.random.randn() * 0.001,
                  np.random.randn() * 0.002),
        )


def synthetic_optical_flow_stream(duration_s: float, rate_hz: float = 50.0):
    """Generate synthetic optical flow (small random drift)."""
    dt = 1.0 / rate_hz
    n = int(duration_s * rate_hz)
    t0 = now_us()

    for i in range(n):
        t = i * dt
        yield OpticalFlowMessage(
            timestamp_us=t0 + int(t * 1e6),
            dx=np.random.randn() * 0.1,
            dy=np.random.randn() * 0.1,
        )


def synthetic_distance_stream(duration_s: float, rate_hz: float = 30.0):
    """Generate synthetic rangefinder data (ground ~50m down)."""
    dt = 1.0 / rate_hz
    n = int(duration_s * rate_hz)
    t0 = now_us()

    for i in range(n):
        t = i * dt
        yield DistanceMessage(
            timestamp_us=t0 + int(t * 1e6),
            front_mm=50000.0 + np.random.randn() * 100,
            down_mm=50000.0 + np.random.randn() * 50,
        )


def synthetic_barometer_stream(duration_s: float, rate_hz: float = 10.0):
    """Generate synthetic barometer data (around 1013 hPa)."""
    dt = 1.0 / rate_hz
    n = int(duration_s * rate_hz)
    t0 = now_us()

    for i in range(n):
        t = i * dt
        yield BarometerMessage(
            timestamp_us=t0 + int(t * 1e6),
            pressure_hpa=1013.25 + np.random.randn() * 0.1,
            temperature_c=15.0,
        )


def synthetic_event_frame_stream(duration_s: float, rate_hz: float = 30.0):
    """Generate synthetic event frames (sparse random activations)."""
    dt = 1.0 / rate_hz
    n = int(duration_s * rate_hz)
    t0 = now_us()
    rng = np.random.RandomState(42)

    for i in range(n):
        t = i * dt
        n_events = rng.randint(100, 500)
        xs = rng.randint(0, 240, n_events).tolist()
        ys = rng.randint(0, 180, n_events).tolist()
        activations = rng.rand(n_events).tolist()
        feature_ids = rng.randint(0, 100, n_events).tolist()

        yield EventFrameMessage(
            timestamp_us=t0 + int(t * 1e6),
            width=240,
            height=180,
            xs=xs,
            ys=ys,
            activations=activations,
            feature_ids=feature_ids,
        )


# ── Simulation runner ────────────────────────────────────────────────────────

def run_simulation(config_path: str, duration_s: float):
    """Run a complete simulated navigation flight."""
    logger.info(f"Starting simulation ({duration_s:.0f}s)...")

    nav = create_navigation_system(config_path)
    nav.start()

    # Set up a simple waypoint mission
    nav.mission_planner.start_mission(
        waypoints=[
            (0.0, 0.0, 50.0),   # takeoff
            (20.0, 0.0, 50.0),  # fly east
            (20.0, 20.0, 50.0), # fly north
            (0.0, 20.0, 50.0),  # fly west
            (0.0, 0.0, 50.0),   # return
        ]
    )

    # Run sensor streams
    imu_gen = synthetic_imu_stream(duration_s)
    flow_gen = synthetic_optical_flow_stream(duration_s)
    dist_gen = synthetic_distance_stream(duration_s)
    baro_gen = synthetic_barometer_stream(duration_s)
    event_gen = synthetic_event_frame_stream(duration_s)

    imu_iter = iter(imu_gen)
    flow_iter = iter(flow_gen)
    dist_iter = iter(dist_gen)
    baro_iter = iter(baro_gen)
    event_iter = iter(event_gen)

    t0 = time.monotonic()
    last_log = t0
    iteration = 0

    try:
        while time.monotonic() - t0 < duration_s:
            iteration += 1

            # Feed IMU at 1 kHz
            try:
                while True:
                    nav.feed_imu(next(imu_iter))
            except StopIteration:
                pass

            # Feed other sensors at their rates (approx)
            if iteration % 20 == 0:
                try:
                    nav.feed_optical_flow(next(flow_iter))
                except StopIteration:
                    pass
            if iteration % 33 == 0:
                try:
                    nav.feed_distance(next(dist_iter))
                except StopIteration:
                    pass
                try:
                    nav.feed_barometer(next(baro_iter))
                except StopIteration:
                    pass
            if iteration % 35 == 0:
                try:
                    nav.feed_event_frame(next(event_iter))
                except StopIteration:
                    pass

            # Auto-arm after 100 IMU readings
            if iteration == 100:
                nav.arm()

            # Run control step
            result = nav.step()

            # Log periodically
            now = time.monotonic()
            if now - last_log >= 1.0:
                state = result["state"]
                logger.info(
                    f"t={now - t0:.1f}s | pos=({state.x:.2f},{state.y:.2f},{state.z:.2f}) "
                    f"| vel=({state.vx:.2f},{state.vy:.2f},{state.vz:.2f}) "
                    f"| mode={result['mode']} | landmarks={result['slam_landmarks']}"
                )
                last_log = now

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    finally:
        nav.stop()
        logger.info(f"Simulation complete. {iteration} iterations.")


# ── Benchmarks ───────────────────────────────────────────────────────────────

def benchmark_evio(config_path: str, input_file: Optional[str] = None):
    """Benchmark the HDC-EVIO pipeline."""
    logger.info("Running HDC-EVIO benchmark...")
    nav = create_navigation_system(config_path)

    # Generate test event frames
    if input_file:
        # Load from file
        logger.info(f"Loading events from {input_file}")
        # TODO: load actual event data
        pass

    # Use synthetic data
    n_steps = 500
    t0 = time.monotonic()
    event_iter = synthetic_event_frame_stream(60.0, 30.0)

    for i, ef in enumerate(event_iter):
        if i >= n_steps:
            break
        nav.feed_event_frame(ef)
        nav.step()

    elapsed = time.monotonic() - t0
    fps = n_steps / elapsed
    logger.info(f"HDC-EVIO benchmark: {n_steps} frames in {elapsed:.2f}s = {fps:.1f} FPS")
    logger.info(f"Map initialized: {nav.hdc_evio._map_initialized}")
    logger.info(f"Position: {nav.hdc_evio.position}")


def benchmark_slam(config_path: str):
    """Benchmark the HDC-SLAM pipeline."""
    logger.info("Running HDC-SLAM benchmark...")
    nav = create_navigation_system(config_path)

    n_steps = 500
    t0 = time.monotonic()
    event_iter = synthetic_event_frame_stream(60.0, 30.0)
    dist_iter = synthetic_distance_stream(60.0, 30.0)

    positions = []
    for i, (ef, dist) in enumerate(zip(event_iter, dist_iter)):
        if i >= n_steps:
            break
        position = np.array([i * 0.1, np.sin(i * 0.05) * 10.0, 50.0])
        positions.append(position)
        result = nav.hdc_slam.step(
            event_frame=ef,
            distance=dist,
            position=position,
        )

    elapsed = time.monotonic() - t0
    fps = n_steps / elapsed
    logger.info(f"HDC-SLAM benchmark: {n_steps} frames in {elapsed:.2f}s = {fps:.1f} FPS")
    logger.info(f"Landmarks: {nav.hdc_slam.memory.get_num_landmarks()}")


# ── Component tests ──────────────────────────────────────────────────────────

def test_ekf():
    """Self-test the Extended Kalman Filter."""
    logger.info("Testing EKF...")
    from navigation.estimation.ekf import IMUOdometryEKF, EKFConfig

    ekf = IMUOdometryEKF()

    for i in range(1000):
        imu = ImuMessage(
            timestamp_us=now_us(),
            accel=(0.0, 0.0, 9.81 + (i - 500) * 0.001),
            gyro=(0.0, 0.0, 0.0),
        )
        ekf.feed_imu(imu)

    state = ekf.get_state()
    logger.info(f"EKF state after 1000 IMU: pos_z={state.z:.3f} (expect ~0)")
    assert abs(state.z) < 1.0, f"EKF drift too high: z={state.z:.3f}"
    logger.info("EKF test PASSED")


def test_lqr():
    """Self-test the LQR controller."""
    logger.info("Testing LQR controller...")
    from navigation.control.lqr import LQRController, LQRConfig
    from navigation.messages import StateEstimate

    lqr = LQRController()

    state = StateEstimate(
        timestamp_us=now_us(),
        x=0.0, y=0.0, z=50.0,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.0, pitch=0.0, yaw=0.0,
    )
    wp = Waypoint(
        timestamp_us=now_us(),
        x=10.0, y=0.0, z=50.0,
    )

    cmd = lqr.compute_control(state, wp)
    logger.info(f"LQR command: M1={cmd.motor_1:.3f} M2={cmd.motor_2:.3f} "
                f"M3={cmd.motor_3:.3f} M4={cmd.motor_4:.3f}")
    assert all(0 <= v <= 1 for v in
               [cmd.motor_1, cmd.motor_2, cmd.motor_3, cmd.motor_4])
    logger.info("LQR test PASSED")


def test_slam():
    """Self-test HDC-SLAM."""
    logger.info("Testing HDC-SLAM...")
    from navigation.slam.hdc_slam import HDCSlamMapper

    slam = HDCSlamMapper()

    for i in range(100):
        xs = np.random.randint(0, 240, 50).tolist()
        ys = np.random.randint(0, 180, 50).tolist()
        ef = EventFrameMessage(
            timestamp_us=now_us(),
            width=240, height=180,
            xs=xs, ys=ys,
            activations=[1.0] * 50,
            feature_ids=(np.random.randint(0, 50, 50)).tolist(),
        )
        pos = np.array([i * 0.1, 0.0, 50.0])
        result = slam.step(event_frame=ef, position=pos)

    n_lm = slam.memory.get_num_landmarks()
    logger.info(f"HDC-SLAM landmarks after 100 steps: {n_lm}")
    assert n_lm > 0, "No landmarks created"
    logger.info("HDC-SLAM test PASSED")


def run_all_tests():
    """Run all component self-tests."""
    logger.info("=" * 60)
    logger.info("Running all component tests")
    logger.info("=" * 60)
    test_ekf()
    test_lqr()
    test_slam()
    logger.info("=" * 60)
    logger.info("All tests PASSED")
    logger.info("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Component tests
    if args.test_all or args.test_ekf or args.test_lqr or args.test_slam:
        if args.test_all:
            run_all_tests()
        else:
            if args.test_ekf:
                test_ekf()
            if args.test_lqr:
                test_lqr()
            if args.test_slam:
                test_slam()
        return

    # Benchmarks
    if args.benchmark_evio:
        benchmark_evio(args.config, args.input)
        return
    if args.benchmark_slam:
        benchmark_slam(args.config)
        return

    # Simulation
    if args.simulate:
        run_simulation(args.config, args.duration)
        return

    # Default: show help
    logger.info(
        "No action specified. Use --simulate, --benchmark_evio, --benchmark_slam, "
        "or --test_all to run."
    )


if __name__ == "__main__":
    main()