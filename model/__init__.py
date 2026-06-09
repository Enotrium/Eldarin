# Eldarin Model Package
#  (https://github.com/Enotrium/Eldarin)
# VSA/HDC integration from arthedain-1 (https://github.com/Enotrium/arthedain-1)
# Event encoding from FPGA-Event-Based-encode (https://github.com/Enotrium/FPGA-Event-Based-encode)
#
# Kanerva (2009) extensions:
#   model/sdm.py          — Sparse Distributed Memory (§2.4, §4)
#   model/hdc_classifier.py — One-shot HDC classifier (§3.1.2) + Item Memory (§2.5)
#   model/vsa_hdc.py      — SparseBinaryVSA + CDT binding (§2.6.2)

from .eldarin_model import Eldarin
from .vsa_hdc import (
    VSAHDC, bind, bundle, permute, cosine_similarity,
    SparseBinaryVSA, sparse_bind, sparse_bundle,
)
from .hierarchy import HierarchyModule
from .mixing import MixingModule
from .heads import DetectionHead, TrackingHead
from .digital_twin import DigitalTwinState, SwarmConsensus, CommunicationAwareMixing
from .fpe import FractionalPowerEncoder, FPEImageEncoder
from .sdm import SparseDistributedMemory, SDMAutoencoder
from .hdc_classifier import HDCClassifier, ItemMemory, hdc_fit, hdc_predict

# Visual Odometry with Neuromorphic Resonator Networks
# Renner et al. (2024), Nature Machine Intelligence — arXiv:2209.02000
from .vo import WorkingMemory, VisualOdometryVSA, create_vo_pipeline

__all__ = [
    "Eldarin",
    "VSAHDC", "bind", "bundle", "permute", "cosine_similarity",
    "SparseBinaryVSA", "sparse_bind", "sparse_bundle",
    "HierarchyModule",
    "MixingModule",
    "DetectionHead", "TrackingHead",
    "DigitalTwinState", "SwarmConsensus", "CommunicationAwareMixing",
    "FractionalPowerEncoder", "FPEImageEncoder",
    "SparseDistributedMemory", "SDMAutoencoder",
    "HDCClassifier", "ItemMemory", "hdc_fit", "hdc_predict",
    "WorkingMemory", "VisualOdometryVSA", "create_vo_pipeline",
]
