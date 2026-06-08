# Eldarin FPGA Deployment Package
# SNN conversion, FPGA export, and HLS synthesis support
#
# References:
#   FPGA-Event-Based-encode: https://github.com/Enotrium/FPGA-Event-Based-encode
#   arthedain-1 VSA/HDC: https://github.com/Enotrium/arthedain-1

from .convert_to_snn import convert_to_snn, calibrate_snn
from .export_fpga import export_to_onnx, export_to_tensorrt, generate_hls_config
from .event_encode import FPGAEventEncoder, StreamEventProcessor

__all__ = [
    "convert_to_snn", "calibrate_snn",
    "export_to_onnx", "export_to_tensorrt", "generate_hls_config",
    "FPGAEventEncoder", "StreamEventProcessor",
]