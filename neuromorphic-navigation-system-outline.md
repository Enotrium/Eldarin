# Neuromorphic Navigation System — Project Outline

A working reference for an autonomous drone navigation stack that replaces GPU-heavy visual navigation with neuromorphic processing (event cameras + spiking neural networks) and Hyperdimensional Computing (HDC). Intended audience: all sub-teams (Embedded, Event, HDC, FPGA) and any AI assistants working on the project.

---

## 1. Project Goals

1. **Fully autonomous navigation** — no human in the loop during flight; the vehicle plans and executes its own trajectory.
2. **GPS-denied operation** — all localization derived from onboard sensors (event-based visual-inertial odometry + SLAM). No reliance on GNSS, no external infrastructure.
3. **Low power, low latency, low weight** — explicitly benchmarked against conventional stacks built on LiDAR + GPU (e.g., Jetson-class SoCs running dense VIO/SLAM). Neuromorphic and HDC techniques are the mechanism for hitting this target.
4. **Payload-adaptive missions** — the flight computer consumes payload sensor data and adjusts waypoints/mission goals in flight.

**Design principle:** wherever a conventional pipeline would use a GPU-intensive algorithm (dense optical flow, feature-based VIO, dense SLAM, learned perception), substitute a sparse, event-driven, or hyperdimensional equivalent.

---

## 2. System Architecture Overview

The system is partitioned into three compute domains plus the sensor suite. Each domain has a distinct latency/criticality profile:

| Domain | Hardware | Responsibilities | Loop character |
|---|---|---|---|
| **Inner loop** | 1+ high-performance MCUs | EKF sensor fusion, LQR motor control, motor ESC interface | Hard real-time, kHz-rate, flight-critical |
| **Neuromorphic perception** | FPGA implementing SNN runtime | Event-stream processing, SNN feature tracking, (possibly) hypervector encoding | Event-driven, asynchronous, sparse |
| **Mission/state layer** | Cortex-A flight computer (RPi-class), ROS2, NEON SIMD | HDC-EVIO, HDC-SLAM, path planning, payload integration, state corrections to MCU | Soft real-time, 10s–100s of Hz |

### 2.1 Failure-isolation rationale
The MCU inner loop can keep the vehicle stable using only IMU + optical flow + distance sensors, even if the FPGA or flight computer degrades or restarts. Higher layers improve the state estimate and provide goals; they are not required for basic stabilization. This layering should be preserved as a hard architectural rule.

---

## 3. Sensor Suite

| Sensor | Measures | Primary consumers |
|---|---|---|
| **Stereo event cameras** | Per-pixel brightness changes (sparse, µs-latency) | SNN feature tracking on FPGA |
| **6-DoF IMU** | Linear acceleration, angular velocity | MCU EKF; spatial-inertial hypervector encoding |
| **Downward optical flow camera** | Frame-to-frame shifts (ground-relative velocity) | MCU EKF; spatial-inertial hypervector encoding |
| **Infrared laser rangefinders** | Directional depth / distance | MCU EKF; both hypervector encoders |
| **Barometer** | Absolute altitude | Spatial-inertial and feature hypervector encoding (altitude context) |
| **Payload sensors** | Mission-specific | Path planning (mission goal adjustment) |

Notes:
- The two event cameras are combined into a stereo pair to recover depth from event correspondences.
- The IR lasers and barometer provide metric scale and altitude grounding that pure event-VIO lacks.

---

## 4. Data Flow (from system diagram)

### 4.1 Perception path (FPGA)
1. Event camera stream → **SNN Feature Tracking** on the neuromorphic FPGA. The SNN converts raw event spikes into sparse feature activations (corners, edges, tracked features).
2. SNN output fans out to two encoders:
   - **Spatial-Inertial Hypervector Encoding** — fuses SNN features with IMU, optical flow, IR depth, and barometer into a single hypervector representing current ego-motion context.
   - **Feature Hypervector Encoding** — encodes landmark/appearance information (with depth and altitude context) for mapping.

### 4.2 Estimation path (Cortex-A)
3. Spatial-inertial hypervectors → **HDC-EVIO** (event visual-inertial odometry implemented with hyperdimensional operations). Outputs position/motion estimates.
4. Feature hypervectors → **HDC-SLAM** (compact associative map of landmarks). Outputs the environment map and loop-closure/landmark recognition.
5. HDC-EVIO sends **accurate state corrections** back down to the MCU's Extended Kalman Filter.

### 4.3 Planning path (Cortex-A)
6. **Path Planning** consumes position/motion (HDC-EVIO), the environment map (HDC-SLAM), and **mission goals derived from payload sensor data**.
7. Path planning emits **control targets** (waypoints/trajectory setpoints) to the MCU.

### 4.4 Control path (MCU)
8. **Extended Kalman Filter** fuses IMU, optical flow, and IR distance at high rate, corrected periodically by HDC-EVIO. Outputs the state estimate.
9. **LQR Controller** takes the state estimate and control targets, computes actuator commands.
10. **Motor ESC** executes commands.

---

## 5. Key Neuromorphic Techniques

### 5.1 Spiking Neural Networks on FPGA
- Purpose: extract usable feature information from the asynchronous event stream without frame reconstruction or GPU inference.
- The FPGA implements a neuromorphic SNN runtime — neuron/synapse update logic, spike routing, and an interface for loading trained network weights.
- Output is sparse activations, which keeps downstream bandwidth and power low.

### 5.2 Hyperdimensional Computing / Vector Symbolic Architecture
- Purpose: replace computationally expensive sensor fusion (especially event-based VIO) and dense SLAM with operations on high-dimensional binary/bipolar vectors (bind, bundle, permute, similarity search).
- Two pipelines:
  - **HDC-EVIO** — ego-motion estimation from encoded spatial-inertial hypervectors.
  - **HDC-SLAM** — compact associative memory of landmark hypervectors for place recognition and mapping.
- Implementation notes:
  - **Torchhd** is the most mature library and good for prototyping, but a custom bit-packed implementation will likely be preferred for deployment.
  - **NEON SIMD on Cortex-A is essential**: 128-bit XOR, popcount, and related primitives are the inner-loop operations of binary HDC.
  - Flight computer should be **at least quad-core** so HDC-EVIO, HDC-SLAM, and path planning run concurrently on dedicated cores.

---

## 6. Open Design Decisions

1. **Where does hypervector encoding live?** (This is why the encoders sit in the overlap of the FPGA and Cortex-A domains in the diagram.)
   - FPGA: lower power and latency, natural pairing with SNN output; but hardware development is slow.
   - CPU: much faster to implement and iterate; higher power and latency.
   - **Action: benchmark HDC encoding latency/throughput on the Cortex-A (with NEON) first.** Move encoding (and possibly more) to the FPGA only if the CPU numbers don't meet budget.
2. **MCU count and partitioning** — one MCU running EKF + LQR, or split estimation and control across two.
3. **Torchhd vs. custom bit-packed HDC** — prototype in Torchhd, define a migration path to a custom implementation with fixed memory layout.
4. **SNN architecture and training method** — network topology, training approach (offline surrogate-gradient training vs. converted ANN), and how trained weights are deployed to the FPGA runtime.
5. **Hardware selection** — specific MCUs, FPGA part, Cortex-A board, and each sensor; confirm **domestic supply chain availability** for all selected parts.

---

## 7. Team Workstreams

### Embedded team
- Implement on the conventional stack first (IMU + optical flow + distance): EKF, LQR control, motor/ESC interface.
- Define the interface for receiving asynchronous state corrections and control targets from the flight computer (rates, message formats, timeout/fallback behavior).
- Integrate HDC-derived corrections later, once HDC-EVIO is producing estimates.

### Event team
- Research and design SNNs that process stereo event-camera data into trackable features.
- Deliverables: candidate architectures, training pipeline, accuracy/sparsity benchmarks, and a weight format consumable by the FPGA runtime.

### HDC team *(one of the two hardest parts)*
- Design the encoding schemes and models for both pipelines:
  - Spatial-inertial encoding → HDC-EVIO (ego-motion).
  - Feature encoding → HDC-SLAM (landmark memory, place recognition).
- Benchmark on Cortex-A with NEON; produce the latency data needed for Decision #1.
- Evaluate Torchhd vs. custom bit-packed implementation.

### FPGA team *(the other hardest part)*
- Build the neuromorphic SNN runtime on the FPGA: spike I/O from the event cameras, neuron update pipeline, weight loading, and output interface to the encoders.
- Keep a path open to also host hypervector encoding on-chip pending Decision #1.

### Cross-team / program
- Hardware selection for sensors and processors; confirm domestic supply chain.
- Define inter-domain interfaces early (MCU ↔ flight computer ↔ FPGA message contracts) so teams can develop in parallel against stubs.
- Research further applications and investor needs.

---

## 8. Suggested Milestones

1. **M1 — Stable flight on conventional stack:** Embedded team's EKF + LQR flying with IMU/optical-flow/distance only.
2. **M2 — Perception bring-up:** SNN runs on FPGA against recorded event data; HDC pipelines run offline on logged sensor data.
3. **M3 — CPU benchmark gate:** HDC latency/power numbers on Cortex-A drive the FPGA-vs-CPU encoding decision.
4. **M4 — Closed-loop integration:** HDC-EVIO corrections feeding the live EKF; HDC-SLAM map feeding path planning.
5. **M5 — Mission demo:** GPS-denied flight with payload-driven waypoint adjustment.

---

## 9. Risks & Open Questions

- **HDC-EVIO accuracy** is unproven relative to conventional VIO at this maturity level; need quantitative drift benchmarks against a baseline (e.g., recorded datasets with ground truth).
- **SNN-on-FPGA throughput** under high event rates (fast motion, high-texture scenes) — define a worst-case event-rate budget.
- **Timing/synchronization** across three compute domains with very different clock domains and latencies; timestamping strategy needs to be defined early.
- **Scale and metric grounding**: event-VIO needs IR/barometer fusion to stay metric — failure modes when those sensors degrade (e.g., over water, in dust) should be enumerated.
- **Supply chain**: domestic sourcing for event cameras and neuromorphic-capable FPGAs is a narrower market than commodity parts; confirm before committing the architecture to specific parts.

---

## 10. Glossary

- **Event camera** — sensor reporting per-pixel brightness *changes* asynchronously (µs latency, sparse output) instead of full frames.
- **SNN (Spiking Neural Network)** — neural network whose units communicate via discrete spikes; maps naturally onto event-camera output and neuromorphic hardware.
- **HDC / VSA (Hyperdimensional Computing / Vector Symbolic Architecture)** — computing with very high-dimensional vectors (e.g., 10,000-bit) using cheap operations (XOR bind, majority bundle, permutation, Hamming similarity).
- **EVIO** — Event-based Visual-Inertial Odometry: estimating ego-motion from event-camera + IMU data.
- **EKF** — Extended Kalman Filter, the MCU-side high-rate sensor fusion.
- **LQR** — Linear Quadratic Regulator, the optimal-control law for motor commands.
- **NEON** — ARM's SIMD instruction set, used for fast wide XOR/popcount in binary HDC.
