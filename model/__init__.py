# Eldarin Model Package
#  (https://github.com/Enotrium/Eldarin)
# VSA/HDC integration from arthedain-1 (https://github.com/Enotrium/arthedain-1)
# Event encoding from FPGA-Event-Based-encode (https://github.com/Enotrium/FPGA-Event-Based-encode)

from .eldarin_model import Eldarin
from .vsa_hdc import VSAHDC, bind, bundle, permute, cosine_similarity
from .hierarchy import HierarchyModule
from .mixing import MixingModule
from .heads import DetectionHead, TrackingHead
from .digital_twin import DigitalTwinState, SwarmConsensus, CommunicationAwareMixing
from .fpe import FractionalPowerEncoder, FPEImageEncoder

__all__ = [
    "Eldarin",
    "VSAHDC", "bind", "bundle", "permute", "cosine_similarity",
    "HierarchyModule",
    "MixingModule",
    "DetectionHead", "TrackingHead",
    "DigitalTwinState", "SwarmConsensus", "CommunicationAwareMixing",
    "FractionalPowerEncoder", "FPEImageEncoder",
]
