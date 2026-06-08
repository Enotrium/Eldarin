# Eldarin Dataset Loaders
# Support for VisDrone, UAVDT, UAV3D, FRED, and synthetic data
#
# VisDrone: https://github.com/VisDrone/VisDrone-Dataset
# UAVDT: https://datasetninja.com/uavdt
# UAV3D: https://uav3d.github.io/
# FRED: https://github.com/francesco-p/FRED

from .visdrone import VisDroneDataset
from .uavdt import UAVDTDataset
from .uav3d import UAV3DDataset
from .fred import FREDDataset
from .synthetic import SyntheticDataset

__all__ = [
    "VisDroneDataset", "UAVDTDataset", "UAV3DDataset",
    "FREDDataset", "SyntheticDataset",
]