# Eldarin — Hierarchical Multimodal 4D Object Detection & Tracking for UAVs

**Eldarin** is a complete adaptation of the [VioPose](https://github.com/SeongJong-Yoo/VioPose) framework, re-targeted from violin performance 4D human pose estimation to **UAV-based multi-object detection, localization, and 4D tracking** (3D position + velocity/trajectory over time) in dynamic, real-world environments.

It preserves VioPose's core hierarchical audiovisual multimodal architecture while integrating:

- **Event-based / neuromorphic sensing** via [FPGA-Event-Based-encode](https://github.com/Enotrium/FPGA-Event-Based-encode) for high-temporal-resolution, low-latency event stream processing
- **Vector Symbolic Architectures (VSA) / Hyperdimensional Computing (HDC)** via the [arthedain-1](https://github.com/Enotrium/arthedain-1) VSA/HDC repository for robust hyperdimensional binding, bundling, and symbolic reasoning over sparse/noisy sensor data
- **Digital Twin & Swarm Consensus** from [Yan et al. (2026) *Nature Communications Engineering*](https://www.nature.com/articles/s44172-025-00571-7) for multi-UAV collaborative perception, communication-aware fusion, and predictive virtual world modeling
- **Spiking Neural Network (SNN) paths** for ultra-low-power FPGA deployment on resource-constrained UAV hardware

## Key Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Multimodal Fusion** | Cascading high-level to low-level features across visual (RGB/event), audio, and IMU modalities |
| **VSA/HDC Binding & Bundling** | Hyperdimensional representations for robust feature fusion, memory, and uncertainty handling |
| **Bayesian-style Cross-modal Mixing** | Causal cross-modal updates from the original VioPose mixing module, enhanced with HDC operations |
| **4D Tracking Head** | Joint object detection (bounding boxes, class probabilities) + 3D position + velocity/trajectory estimation |
| **Event Camera Pipeline** | FPGA-optimized event encoding with SNN-compatible sparse representations |
| **Real-time UAV Inference** | Optimized for onboard deployment with fp16/int8 quantization, TensorRT export, and SNN conversion |
| **Multi-dataset Support** | VisDrone, UAVDT, UAV3D, FRED (RGB+Event), and synthetic data pipelines |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT MODALITIES                          │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│   RGB    │  Event   │  Audio   │   IMU    │    GPS/Pose         │
│  Frames  │  Stream  │  Stream  │  Sensor  │    (optional)       │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────────┬──────────┘
     │          │          │          │                │
     ▼          ▼          ▼          ▼                ▼
┌─────────┐┌─────────┐┌─────────┐┌─────────┐  ┌──────────────┐
│ Visual  ││ Event   ││ Audio   ││  IMU    │  │  Pose/Aux    │
│ Encoder ││ Encoder ││ Encoder ││ Encoder │  │  Embedding   │
│(ResNet/ ││(FPGA-   ││(Wave2Vec││(MLP/    │  │              │
│ ViT)    ││ Event)  ││ /CNN)   ││ LSTM)   │  │              │
└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘  └──────┬───────┘
     │          │          │          │                │
     └──────────┴──────────┴──────────┴────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  HIERARCHY MODULE   │
              │  (Cascading High →  │
              │   Low Features)     │
              │  + VSA/HDC Binding  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   MIXING MODULE     │
              │  (Bayesian-style    │
              │   Cross-modal       │
              │   Updates + HDC)    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  DETECTION + 4D     │
              │   TRACKING HEAD     │
              │  (BBox, Class, 3D   │
              │   Position, Velocity│
              │   Trajectory)       │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  FPGA / SNN Export  │
              │  (Lava, snnTorch,   │
              │   TensorRT, HLS)    │
              └─────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Enotrium/Eldarin.git
cd Eldarin

# Create conda environment (recommended)
conda create -n eldarin python=3.10
conda activate eldarin

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt

# Optional: Install SNN framework for FPGA deployment
pip install snntorch lava-numpy  # or lava-dl for Intel Loihi

# Optional: Install event-camera tools
pip install tonic metavision-preview  # for event data processing
```

## Quick Start

### Inference (Real-time UAV)

```bash
python inference.py \
  --config config/inference.yaml \
  --checkpoint checkpoints/eldarin_v1.pth \
  --input /path/to/video.mp4 \
  --modality rgb+event \
  --output results/
```

### Training

```bash
# Single GPU training with VisDrone
python main.py \
  --config config/train_visdrone.yaml \
  --data_root /path/to/VisDrone \
  --epochs 100 \
  --batch_size 8

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --config config/train_multimodal.yaml \
  --distributed

# With event data (FRED dataset)
python main.py \
  --config config/train_event.yaml \
  --data_root /path/to/FRED \
  --modality rgb+event
```

### FPGA / SNN Export

```bash
# Convert to SNN for neuromorphic hardware
python fpga/convert_to_snn.py --checkpoint checkpoints/eldarin_v1.pth --output checkpoints/eldarin_snn.pth

# Export for FPGA HLS synthesis
python fpga/export_fpga.py --config config/fpga_export.yaml
```

## Supported Datasets

| Dataset | Modalities | Task | Link |
|---------|-----------|------|------|
| **VisDrone** | RGB | Detection + Tracking | [GitHub](https://github.com/VisDrone/VisDrone-Dataset) |
| **UAVDT** | RGB | Vehicle Detection/Tracking | [DatasetNinja](https://datasetninja.com/uavdt) |
| **UAV3D** | RGB + 3D Boxes | 3D Detection/Tracking | [Project Page](https://uav3d.github.io/) |
| **FRED** | RGB + Event | Drone Detection | [FRED](https://github.com/francesco-p/FRED) |
| **MVSEC** | Stereo + Event | Multi-vehicle | [MVSEC](https://daniilidis-group.github.io/mvsec/) |

## Metrics

Eldarin evaluates on standard UAV detection and tracking metrics:

- **Detection**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Tracking**: MOTA, MOTP, IDF1, HOTA, trajectory error (ATE, RPE)
- **4D-specific**: 3D IoU, velocity RMSE, occlusion robustness score

## Repository Structure

```
Eldarin/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                      # Training entry point
├── inference.py                 # Real-time inference
├── config/
│   ├── __init__.py
│   ├── base.yaml                # Base configuration
│   ├── train_visdrone.yaml      # VisDrone training config
│   ├── train_multimodal.yaml    # Multi-modal training
│   ├── train_event.yaml         # Event-based training
│   ├── inference.yaml           # Inference configuration
│   └── fpga_export.yaml         # FPGA export settings
├── model/
│   ├── __init__.py
│   ├── eldarin_model.py         # Main Eldarin model
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── visual_encoder.py    # RGB frame encoder
│   │   ├── event_encoder.py     # Event stream encoder (FPGA-compatible)
│   │   ├── audio_encoder.py     # Audio encoder
│   │   └── imu_encoder.py       # IMU/auxiliary encoder
│   ├── hierarchy.py             # Hierarchy module (cascading fusion)
│   ├── mixing.py                # Bayesian-style mixing module
│   ├── vsa_hdc.py               # VSA/HDC operations (binding, bundling)
│   ├── heads.py                 # Detection + 4D tracking heads
│   └── snn_layers.py            # SNN-compatible layer definitions
├── utils/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   ├── event_utils.py           # Event data processing (FPGA encode)
│   ├── metrics.py               # Detection + tracking metrics
│   ├── visualization.py         # Visualization tools
│   ├── loss.py                  # Loss functions
│   └── trainer.py               # Training loop utilities
├── datasets/
│   ├── __init__.py
│   ├── visdrone.py              # VisDrone dataset loader
│   ├── uavdt.py                 # UAVDT dataset loader
│   ├── uav3d.py                 # UAV3D dataset loader
│   ├── fred.py                  # FRED event dataset loader
│   └── synthetic.py             # Synthetic data generator
├── fpga/
│   ├── __init__.py
│   ├── convert_to_snn.py        # ANN → SNN conversion
│   ├── export_fpga.py           # FPGA HLS export
│   ├── event_encode.py          # FPGA event encoding (from Enotrium)
│   ├── hls_kernels/             # HLS C++ kernel templates
│   │   └── vsa_kernel.cpp
│   └── snn_sim.py               # SNN simulation harness
├── scripts/
│   ├── download_datasets.sh     # Dataset download helper
│   ├── prepare_visdrone.py      # VisDrone preprocessing
│   └── run_ablation.py          # Ablation study runner
└── checkpoints/                 # Model weights directory
```

## Key Adaptations from VioPose

### 1. Domain Shift: Violin Pose → UAV 4D Tracking

The original VioPose estimates 3D violin performance poses over time using 2D keypoints + audio. Eldarin replaces:

- **Input**: 2D keypoints → RGB frames + event streams + optional audio/IMU
- **Output**: 3D joint positions → Bounding boxes, class probabilities, 3D positions, velocities, trajectories
- **Head**: Pose regression MLP → YOLO-style detection head + Kalman-inspired 4D tracking in hyperdimensional space

### 2. Event-based Encoding (FPGA-Event-Based-encode Integration)

Leverages the efficient FPGA event encoding from [Enotrium/FPGA-Event-Based-encode](https://github.com/Enotrium/FPGA-Event-Based-encode):

- Sparse event-to-frame conversion optimized for FPGA streaming
- SNN-compatible spike representations
- Low-latency feature extraction suitable for real-time UAV processing

### 3. VSA/HDC Integration (arthedain-1)

Incorporates [arthedain-1](https://github.com/Enotrium/arthedain-1) VSA/HDC primitives:

- **Binding (⊗)**: Associates features across modalities (e.g., visual feature ⊗ event feature)
- **Bundling (⊕)**: Superimposes multiple feature bindings for compact representation
- **Permutation (ρ)**: Encodes temporal/sequential relationships for trajectory modeling
- **Similarity**: Cosine/hamming distance for robust matching under noise

These replace/supplement attention mechanisms with hyperdimensional operations that are:
- More robust to noise and sparsity
- Naturally compatible with binary/spike-based computation
- Hardware-efficient (bitwise operations on FPGAs)

### 4. Hierarchy Module Enhancement

The cascading high→low feature flow is augmented with VSA binding:
- High-level semantics (object class, scene context) bind with low-level features (edges, motion)
- Creates hyperdimensional "role-filler" representations
- Enables robust feature reconstruction under occlusion

### 5. Mixing Module with Bayesian-HDC Updates

The Bayesian-style cross-modal updates now operate in hyperdimensional space:
- Prior: HDC bundle of previous modalities
- Likelihood: HDC encoding of new modality
- Posterior: Weighted bundle with uncertainty gating
- Natural handling of missing modalities (sparse sensor data)

### 6. SNN Conversion Paths

For FPGA deployment:
- ANN layers → IF/LIF neuron equivalents
- Rate-based → temporal spike-based conversion
- Compatible with snnTorch and Lava frameworks
- HLS C++ kernel templates for direct FPGA synthesis

## Citations

If you use Eldarin in your research, please cite:

### Core Framework
```bibtex
@article{yoo2024viopose,
  title={VioPose: Hierarchical Audiovisual Multimodal Network for 4D Human Pose Estimation in Violin Performances},
  author={Yoo, SeongJong and others},
  journal={arXiv preprint arXiv:2411.13607},
  year={2024}
}
```

### Event-based Encoding
```bibtex
@software{enotrium_fpga_event_encode,
  title={FPGA-Event-Based-encode: Efficient FPGA Event Data Processing},
  author={Enotrium},
  url={https://github.com/Enotrium/FPGA-Event-Based-encode}
}
```

### VSA/HDC Framework
```bibtex
@software{enotrium_arthedain,
  title={arthedain-1: Vector Symbolic Architecture / Hyperdimensional Computing},
  author={Enotrium},
  url={https://github.com/Enotrium/arthedain-1}
}
```

### Digital Twin & Swarm Consensus
```bibtex
@article{yan2026digital,
  title={Digital twin-driven swarm of autonomous underwater vehicles for marine exploration},
  author={Yan, Jing and Zhang, Tianyi and Guan, Xinping and Yang, Xian and Chen, Cailian},
  journal={Communications Engineering},
  volume={5},
  number={1},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s44172-025-00571-7}
}
```

### Datasets
```bibtex
@inproceedings{zhu2021visdrone,
  title={VisDrone-DET2021: The Vision Meets Drone Object Detection Challenge Results},
  author={Zhu, Pengfei and others},
  booktitle={ICCV Workshops},
  year={2021}
}
@article{du2018uavdt,
  title={The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking},
  author={Du, Dawei and others},
  journal={ECCV},
  year={2018}
}
```

## License

MIT License. See [LICENSE](LICENSE) file.

## Contributing

Contributions welcome! Areas of particular interest:
- Additional dataset loaders
- SNN accuracy optimization
- FPGA deployment testing
- Multi-UAV collaborative tracking extensions

---

**Links**: [VioPose](https://github.com/SeongJong-Yoo/VioPose) | [VioPose Paper](https://arxiv.org/pdf/2411.13607) | [VioPose Project](https://sj-yoo.info/viopose/) | [FPGA Event Encode](https://github.com/Enotrium/FPGA-Event-Based-encode) | [arthedain-1 VSA/HDC](https://github.com/Enotrium/arthedain-1)