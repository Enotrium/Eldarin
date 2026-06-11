# Eldarin — Neuromorphic Autonomous Navigation System

**Eldarin** is a GPS-denied autonomous drone navigation stack that replaces GPU-heavy visual navigation with neuromorphic processing (event cameras + spiking neural networks) and Hyperdimensional Computing (HDC).

The system is partitioned into **three compute domains** with distinct latency/criticality profiles, as specified in the [System Architecture Outline](neuromorphic-navigation-system-outline.md):

| Domain | Hardware | Responsibilities | Loop Character |
|--------|----------|-----------------|----------------|
| **Inner Loop** | 1+ high-performance MCUs | EKF sensor fusion, LQR motor control, motor ESC interface | Hard real-time, kHz-rate, flight-critical |
| **Neuromorphic Perception** | FPGA implementing SNN runtime | Event-stream processing, SNN feature tracking | Event-driven, asynchronous, sparse |
| **Mission / State Layer** | Cortex-A flight computer (RPi-class), ROS2, NEON SIMD | HDC-EVIO, HDC-SLAM, path planning, payload integration | Soft real-time, 10s–100s of Hz |

**Design principle:** wherever a conventional pipeline would use a GPU-intensive algorithm (dense optical flow, feature-based VIO, dense SLAM, learned perception), substitute a sparse, event-driven, or hyperdimensional equivalent.

---

## System Architecture

```
 ┌───────────────────────────────────────────────────────────────────┐
 │                     SENSOR SUITE                                   │
 │  Stereo Event Cameras  │ 6-DoF IMU │ Optical Flow │ IR Lasers    │
 │  Barometer             │ Payload Sensors                          │
 └───────────────┬───────────────────────────────────┬───────────────┘
                 │                                   │
 ┌───────────────▼──────────┐  ┌─────────────────────▼───────────────┐
 │  FPGA — Neuromorphic     │  │  MCU — Inner Loop (hard real-time)  │
 │  • SNN Feature Tracking  │  │  • Extended Kalman Filter           │
 │  • Event-stream encode   │  │  • LQR Motor Controller            │
 │  • (opt) HV encoding     │  │  • Motor ESC interface              │
 │  Event-driven, sparse    │  │  kHz-rate, flight-critical          │
 └───────────────┬──────────┘  └─────────────────────┬───────────────┘
                 │                                   │
                 │    SensorPackets                   │  StateEstimate
                 ▼                                   ▼  corrections
 ┌───────────────────────────────────────────────────────────────────┐
 │  Cortex-A — Mission / State Layer (soft real-time, NEON SIMD)     │
 │                                                                   │
 │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
 │  │  HDC-EVIO    │  │  HDC-SLAM    │  │  Path Planning       │    │
 │  │  Ego-motion  │  │  Landmark    │  │  • Waypoint following│    │
 │  │  estimation  │  │  map + loop  │  │  • Coverage patterns │    │
 │  │  from HD     │  │  closure     │  │  • Payload-adaptive  │    │
 │  │  vectors     │  │              │  │  • Obstacle avoidance│    │
 │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘    │
 │         │                 │                      │                │
 │         └─────────────────┴──────────────────────┘                │
 │                    Control targets → MCU LQR                       │
 └───────────────────────────────────────────────────────────────────┘
```

### Failure-isolation rule (§2.1 of outline)
The MCU inner loop can keep the vehicle stable using **only IMU + optical flow + distance sensors**, even if the FPGA or flight computer degrades or restarts. Higher layers improve the state estimate and provide goals; they are not required for basic stabilization.

---

## Quick Start

### Navigation System

```bash
# Run the full navigation system simulation
python main_navigation.py --config config/navigation.yaml --simulate --duration 60

# Run component self-tests
python main_navigation.py --test_all

# Benchmark HDC-EVIO pipeline
python main_navigation.py --benchmark_evio

# Benchmark HDC-SLAM pipeline
python main_navigation.py --benchmark_slam
```

### Detection / Tracking (original paper pipeline)

```bash
# Training
python main.py --config config/train_visdrone.yaml --data_root /path/to/VisDrone

# Inference
python inference.py --checkpoint checkpoints/best_model.pth --input video.mp4

# Visual Odometry (training-free)
python inference.py --config config/inference.yaml --mode vo --input events.npy
```

---

## Installation

```bash
git clone https://github.com/Enotrium/Eldarin.git
cd Eldarin

# Create conda environment
conda create -n eldarin python=3.10
conda activate eldarin

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt

# Optional: SNN framework for FPGA deployment
pip install snntorch lava-numpy
```

---

## Repository Structure

```
Eldarin/
├── README.md                              # This file
├── neuromorphic-navigation-system-outline.md  # Architecture specification
├── requirements.txt
├── main.py                                # Detection/tracking training entry point
├── main_navigation.py                     # 🆕 Navigation system entry point
├── inference.py                           # Real-time detection inference (+ VO mode)
├── config/
│   ├── navigation.yaml                    # 🆕 Navigation system configuration
│   ├── base.yaml                          # Base model configuration
│   ├── train_visdrone.yaml / train_multimodal.yaml / train_event.yaml
│   ├── inference.yaml
│   └── fpga_export.yaml
├── navigation/                            # 🆕 Three-domain navigation stack
│   ├── __init__.py                        # Public API
│   ├── messages.py                        # Inter-domain message contracts
│   ├── system.py                          # System orchestrator (wires all domains)
│   ├── estimation/
│   │   ├── ekf.py                         # Extended Kalman Filter (MCU inner loop)
│   │   └── hdc_evio.py                    # HDC-EVIO — ego-motion estimation (Cortex-A)
│   ├── slam/
│   │   └── hdc_slam.py                    # HDC-SLAM — landmark mapping + loop closure
│   ├── control/
│   │   ├── lqr.py                         # LQR controller + motor mixing (MCU)
│   │   └── motor_esc.py                   # Motor ESC interface (MCU)
│   └── planning/
│       └── path_planner.py                # Path planner + mission controller (Cortex-A)
├── model/                                 # Perception / detection / VSA models
│   ├── __init__.py
│   ├── eldarin_model.py                   # Main Eldarin detection model
│   ├── vo.py                              # Visual Odometry (Renner et al. 2024)
│   ├── fpe.py                             # Fractional Power Encoding
│   ├── vsa_hdc.py                         # VSA/HDC + Resonator + Hierarchical Resonator
│   ├── digital_twin.py                    # Digital Twin + Swarm Consensus
│   ├── encoders/                          # Visual, event, audio, IMU encoders
│   ├── hierarchy.py / mixing.py           # Hierarchical fusion + Bayesian mixing
│   ├── heads.py                           # Detection + 4D tracking heads
│   ├── hdc_classifier.py                  # HDC Classifier + Item Memory
│   ├── sdm.py                             # Sparse Distributed Memory
│   └── snn_layers.py                      # SNN-compatible layer definitions
├── utils/                                 # Data loading, metrics, visualization
├── datasets/                              # VisDrone, UAVDT, UAV3D, FRED, synthetic
├── fpga/                                  # FPGA HLS kernels, SNN conversion, export
├── tests/                                 # Unit tests
├── figures/ / images/                     # Paper figures (HTML + PNG)
└── scripts/                               # Dataset prep, figure generation, ablation
```

---

## Key Neuromorphic Techniques

### Spiking Neural Networks on FPGA

Purpose: extract usable feature information from the asynchronous event stream without frame reconstruction or GPU inference. The FPGA implements a neuromorphic SNN runtime — neuron/synapse update logic, spike routing, and an interface for loading trained network weights. Output is sparse activations, keeping downstream bandwidth and power low.

### Hyperdimensional Computing / VSA

Purpose: replace computationally expensive sensor fusion (especially event-based VIO) and dense SLAM with operations on high-dimensional binary/bipolar vectors (bind, bundle, permute, similarity search).

Two pipelines:
- **HDC-EVIO** — ego-motion estimation from encoded spatial-inertial hypervectors (based on Renner et al. 2024, *Nature Machine Intelligence*)
- **HDC-SLAM** — compact associative memory of landmark hypervectors for place recognition and mapping

Implementation notes:
- **Torchhd** is mature for prototyping, but a custom bit-packed implementation is preferred for deployment
- **NEON SIMD on Cortex-A is essential**: 128-bit XOR, popcount, and related primitives are the inner-loop operations of binary HDC
- Flight computer should be **at least quad-core** so HDC-EVIO, HDC-SLAM, and path planning run concurrently on dedicated cores

---

## Sensor Suite

| Sensor | Measures | Primary Consumers |
|--------|----------|-------------------|
| **Stereo event cameras** | Per-pixel brightness changes (sparse, µs-latency) | SNN feature tracking on FPGA |
| **6-DoF IMU** | Linear acceleration, angular velocity | MCU EKF; spatial-inertial hypervector encoding |
| **Downward optical flow camera** | Frame-to-frame shifts (ground-relative velocity) | MCU EKF; spatial-inertial hypervector encoding |
| **Infrared laser rangefinders** | Directional depth / distance | MCU EKF; both hypervector encoders |
| **Barometer** | Absolute altitude | Spatial-inertial and feature hypervector encoding |
| **Payload sensors** | Mission-specific | Path planning (mission goal adjustment) |

---

## Data Flow

### Perception path (FPGA)
1. Event camera stream → **SNN Feature Tracking** on the FPGA
2. SNN output fans out to two encoders:
   - **Spatial-Inertial Hypervector Encoding** — fuses SNN features with IMU, optical flow, IR depth, barometer
   - **Feature Hypervector Encoding** — encodes landmark/appearance information for mapping

### Estimation path (Cortex-A)
3. Spatial-inertial hypervectors → **HDC-EVIO** (event visual-inertial odometry)
4. Feature hypervectors → **HDC-SLAM** (compact associative map of landmarks)
5. HDC-EVIO sends **accurate state corrections** back down to the MCU's EKF

### Planning path (Cortex-A)
6. **Path Planning** consumes position/motion (HDC-EVIO), the environment map (HDC-SLAM), and **mission goals derived from payload sensor data**
7. Path planning emits **control targets** (waypoints) to the MCU

### Control path (MCU)
8. **Extended Kalman Filter** fuses IMU, optical flow, and IR distance at high rate, corrected periodically by HDC-EVIO
9. **LQR Controller** takes the state estimate and control targets, computes actuator commands
10. **Motor ESC** executes commands

---

## Development Milestones

| Milestone | Description |
|-----------|-------------|
| **M1 — Stable flight** | Embedded team's EKF + LQR flying with IMU/optical-flow/distance only |
| **M2 — Perception bring-up** | SNN runs on FPGA against recorded event data; HDC pipelines run offline |
| **M3 — CPU benchmark gate** | HDC latency/power numbers on Cortex-A drive FPGA-vs-CPU encoding decision |
| **M4 — Closed-loop integration** | HDC-EVIO corrections feeding the live EKF; HDC-SLAM map feeding path planning |
| **M5 — Mission demo** | GPS-denied flight with payload-driven waypoint adjustment |

---

## Open Design Decisions

1. **Where does hypervector encoding live?** Benchmark HDC encoding latency/throughput on the Cortex-A (with NEON) first. Move encoding (and possibly more) to FPGA only if CPU numbers don't meet budget.
2. **MCU count and partitioning** — one MCU running EKF + LQR, or split estimation and control across two.
3. **Torchhd vs. custom bit-packed HDC** — prototype in Torchhd, define a migration path to a custom implementation.
4. **SNN architecture and training method** — network topology, training approach, and weight deployment to FPGA runtime.
5. **Hardware selection** — specific MCUs, FPGA part, Cortex-A board, and sensors; confirm domestic supply chain availability.

---

## Supported Datasets

| Dataset | Modalities | Task |
|---------|-----------|------|
| **VisDrone** | RGB | Detection + Tracking |
| **UAVDT** | RGB | Vehicle Detection/Tracking |
| **UAV3D** | RGB + 3D Boxes | 3D Detection/Tracking |
| **FRED** | RGB + Event | Drone Detection |
| **MVSEC** | Stereo + Event | Multi-vehicle |
| **Event Camera Dataset** | Events + IMU + Ground Truth | Visual Odometry |

---

## Key Equations (Renner et al. 2024 VO)

| Eq. | Name | Formula |
|-----|------|---------|
| 1–2 | FPE Encoding | `s = Σ_{(x,y)∈E} h₀ˣ ⊗ v₀ʸ` |
| 3 | Codebook | `s = Φ · I` |
| 4 | Hierarchical Resonator | `ĥ(t+1) = (1-γ)·ĥ(t) + γ·f(H H† (ŝ(t) ⊙ v̂* ⊙ m̂*))` |
| 5–7 | Population Vector | `h_out = Σ i·h_sim(i) / Σ h_sim(i)` |
| 8 | Camera→Map | `m(t) = Λ(s(t) ⊗ h^{h_out} ⊗ v^{v_out}) ⊗ r^{r_out}` |
| 9 | Anchored Map Update | `m̂(t+1) = μ₁·m̂(t) + μ₂·m̂(0) + (1-μ₁-μ₂)·m(t)` |
| 10 | IMU Fusion | `r̂(t) = r̂(t-1) ⊗ r_seed^{IMU(t)}` |

---

## Citations

If you use Eldarin in your research, please cite:

```bibtex
@article{renner2024visual,
  title={Visual odometry with neuromorphic resonator networks},
  author={Renner, Alpha and Supic, Lazar and Danielescu, Andreea and
          Indiveri, Giacomo and Frady, E Paxon and Sommer, Friedrich T
          and Sandamirskaya, Yulia},
  journal={Nature Machine Intelligence},
  year={2024},
  doi={10.1038/s42256-024-00848-0}
}
```

```bibtex
@article{yan2026digital,
  title={Digital twin-driven swarm of autonomous underwater vehicles for marine exploration},
  author={Yan, Jing and Zhang, Tianyi and Guan, Xinping and Yang, Xian and Chen, Cailian},
  journal={Communications Engineering}, volume={5}, number={1}, year={2026},
  publisher={Nature Publishing Group}, doi={10.1038/s44172-025-00571-7}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) file.

## Contributing

Areas of particular interest:
- EKF/LQR tuning for real hardware
- HDC-EVIO accuracy benchmarking against conventional VIO
- SNN-on-FPGA throughput characterization
- Multi-UAV swarm consensus testing
- Payload-adaptive mission planning extensions
- NEON SIMD optimisation of HDC operations

---

**Links**: [System Outline](neuromorphic-navigation-system-outline.md) | [Renner et al. VO](https://arxiv.org/abs/2209.02000) | [FPGA Event Encode](https://github.com/Enotrium/FPGA-Event-Based-encode) | [arthedain-1 VSA/HDC](https://github.com/Enotrium/arthedain-1) | [Yan et al. 2026](https://www.nature.com/articles/s44172-025-00571-7)