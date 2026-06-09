"""
Eldarin — Complete Hierarchical Multimodal 4D Detection & Tracking Model
=========================================================================

with integrations from:
  - FPGA-Event-Based-encode (https://github.com/Enotrium/FPGA-Event-Based-encode)
  - arthedain-1 VSA/HDC (https://github.com/Enotrium/arthedain-1)
  - Digital Twin & Swarm Consensus from Yan et al. (2026), Nature CommsEng
    "Digital twin-driven swarm of autonomous underwater vehicles for marine exploration"
    https://www.nature.com/articles/s44172-025-00571-7

Architecture:
  1. Single-modality encoders (visual, event, audio, IMU)
  2. Hierarchy module (cascading high→low features with VSA binding)
  3. Mixing module (Bayesian-style cross-modal fusion in HD space)
  4. Detection head (YOLO-style object detection)
  5. Tracking head (HD Kalman filter for 4D tracking)
  6. Digital Twin (HD virtual replica with predictive forward model)
  7. Swarm Consensus (multi-UAV collaborative fusion)

The model preserves Eldarin's core hierarchical multimodal approach
while adapting to UAV object detection and 4D tracking with
event-based sensing, hyperdimensional computing, and swarm coordination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union

from .encoders.visual_encoder import VisualEncoder
from .encoders.event_encoder import EventEncoder
from .encoders.audio_encoder import AudioEncoder
from .encoders.imu_encoder import IMUEncoder, PoseEncoder
from .hierarchy import HierarchyModule
from .mixing import MixingModule
from .heads import DetectionHead, TrackingHead
from .vsa_hdc import VSAHDC
from .digital_twin import DigitalTwinState, SwarmConsensus, CommunicationAwareMixing


class Eldarin(nn.Module):
    """
    Eldarin: Hierarchical Multimodal 4D Object Detection & Tracking for UAVs.

    This is the complete model combining:
      - Multi-modal encoders (visual, event, audio, IMU)
      - Hierarchical feature fusion with VSA/HDC binding
      - Bayesian-style cross-modal mixing
      - Detection + 4D tracking heads

    Args:
        config: Configuration dict with model, data, and training settings
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config.get("model", config)

        # Modality settings
        self.modalities = self._parse_modalities(model_cfg)

        # --- Encoders ---
        self.visual_encoder = None
        self.event_encoder = None
        self.audio_encoder = None
        self.imu_encoder = None
        self.pose_encoder = None

        visual_dim = model_cfg.get("visual_dim", 1024)
        event_dim = model_cfg.get("event_dim", 512)
        audio_dim = model_cfg.get("audio_dim", 512)
        imu_dim = model_cfg.get("imu_dim", 128)

        if "visual" in self.modalities or "rgb" in self.modalities:
            self.visual_encoder = VisualEncoder(
                backbone=model_cfg.get("visual_backbone", "resnet18"),
                pretrained=model_cfg.get("pretrained", True),
                out_dim=visual_dim,
                fpn_channels=model_cfg.get("fpn_channels", 256),
            )

        if "event" in self.modalities:
            event_cfg = model_cfg.get("event", {})
            self.event_encoder = EventEncoder(
                height=event_cfg.get("height", 480),
                width=event_cfg.get("width", 640),
                representation=event_cfg.get("representation", "voxel_grid"),
                num_bins=event_cfg.get("num_bins", 10),
                out_dim=event_dim,
            )

        if "audio" in self.modalities:
            audio_cfg = model_cfg.get("audio", {})
            self.audio_encoder = AudioEncoder(
                sample_rate=audio_cfg.get("sample_rate", 22050),
                out_dim=audio_dim,
            )

        if "imu" in self.modalities:
            self.imu_encoder = IMUEncoder(
                out_dim=imu_dim,
            )

        if "pose" in self.modalities:
            self.pose_encoder = PoseEncoder(
                out_dim=model_cfg.get("pose_dim", 128),
            )

        # --- Hierarchy Module ---
        hierarchy_cfg = model_cfg.get("hierarchy", {})
        self.hierarchy = HierarchyModule(
            level_dims=hierarchy_cfg.get("level_dims", [2048, 1024, 512, 256]),
            use_vsa_binding=hierarchy_cfg.get("use_vsa_binding", True),
            hd_dim=model_cfg.get("hd_dim", 8192),
            dropout=hierarchy_cfg.get("dropout", 0.1),
        )

        # --- Mixing Module ---
        mixing_cfg = model_cfg.get("mixing", {})
        # Build feature dims dict from active modalities
        feature_dims = {}
        if self.visual_encoder:
            feature_dims["visual"] = visual_dim
        if self.event_encoder:
            feature_dims["event"] = event_dim
        if self.audio_encoder:
            feature_dims["audio"] = audio_dim
        if self.imu_encoder:
            feature_dims["imu"] = imu_dim
        if self.pose_encoder:
            feature_dims["pose"] = model_cfg.get("pose_dim", 128)

        self.mixing = MixingModule(
            feature_dims=feature_dims if feature_dims else {"visual": visual_dim},
            hd_dim=model_cfg.get("hd_dim", 8192),
            num_iterations=mixing_cfg.get("num_iterations", 3),
            use_uncertainty_gating=mixing_cfg.get("use_uncertainty_gating", True),
            temporal_window=mixing_cfg.get("temporal_window", 16),
            prior_weight=mixing_cfg.get("prior_weight", 0.7),
        )

        # --- Detection Head ---
        heads_cfg = model_cfg.get("heads", {})
        det_cfg = heads_cfg.get("detection", {})
        self.detection_head = DetectionHead(
            in_channels=hierarchy_cfg.get("level_dims", [2048, 1024, 512, 256])[-1],
            num_classes=heads_cfg.get("num_classes", 10),
            num_anchors=det_cfg.get("anchors", 3),
            use_3d=heads_cfg.get("use_3d", True),
        )

        # --- Tracking Head ---
        track_cfg = heads_cfg.get("tracking", {})
        self.tracking_head = TrackingHead(
            state_dim=track_cfg.get("state_dim", 8),
            hd_dim=model_cfg.get("hd_dim", 8192),
            feature_dim=feature_dims.get("visual", visual_dim),
            max_age=track_cfg.get("max_age", 30),
            min_hits=track_cfg.get("min_hits", 3),
        )

        # --- Digital Twin (Yan et al. 2026, Nature CommsEng) ---
        twin_cfg = model_cfg.get("digital_twin", {})
        self.digital_twin = DigitalTwinState(
            hd_dim=model_cfg.get("hd_dim", 8192),
            num_object_slots=twin_cfg.get("num_object_slots", 64),
            state_dim=track_cfg.get("state_dim", 8),
            context_dim=model_cfg.get("visual_dim", 1024),
        ) if twin_cfg.get("enabled", True) else None

        # --- Swarm Consensus (Yan et al. 2026, Nature CommsEng) ---
        swarm_cfg = model_cfg.get("swarm", {})
        self.swarm = SwarmConsensus(
            num_agents=swarm_cfg.get("num_agents", 4),
            hd_dim=model_cfg.get("hd_dim", 8192),
            consensus_rounds=swarm_cfg.get("consensus_rounds", 3),
        ) if swarm_cfg.get("enabled", False) else None

        # Communication-aware feature mixing
        self.comm_aware_mixing = CommunicationAwareMixing(
            hd_dim=model_cfg.get("hd_dim", 8192),
            feature_dim=feature_dims.get("visual", visual_dim),
        ) if swarm_cfg.get("enabled", False) else None

        # --- VSA-native reasoning path (from Renner et al. 2024, arXiv:2209.02000) ---
        vsa_native_cfg = model_cfg.get("vsa_native", {})
        if vsa_native_cfg.get("enabled", True):
            from .vsa_hdc import ResonatorNetwork, HierarchicalResonatorNetwork
            from .fpe import FractionalPowerEncoder

            self.fpe_encoder = FractionalPowerEncoder(
                hd_dim=model_cfg.get("hd_dim", 8192),
                min_val=0.0,
                max_val=float(max(
                    vsa_native_cfg.get("image_height", 480),
                    vsa_native_cfg.get("image_width", 640),
                )),
                dtype=model_cfg.get("hd_dtype", "bipolar"),
                seed=777,
            )

            self.resonator = HierarchicalResonatorNetwork(
                cartesian_factors=vsa_native_cfg.get("cartesian_factors", [64, 64]),
                logpolar_factors=vsa_native_cfg.get("logpolar_factors", [36]),
                hd_dim=model_cfg.get("hd_dim", 8192),
                gamma=vsa_native_cfg.get("resonator_gamma", 0.3),
                nonlinearity=vsa_native_cfg.get("resonator_nonlinearity", "phasor"),
                dtype=model_cfg.get("hd_dtype", "bipolar"),
                seed=777,
            )

            # VSA-native IMU fusion (Eq. 10 from paper)
            self.vsa_imu_enabled = vsa_native_cfg.get("imu_fusion", True)
            self.vsa_map_integration = vsa_native_cfg.get("map_integration", True)
        else:
            self.fpe_encoder = None
            self.resonator = None
            self.vsa_imu_enabled = False
            self.vsa_map_integration = False

        # SNN mode flag
        self.snn_mode = model_cfg.get("snn", {}).get("enabled", False)

        self._init_weights()

    def _parse_modalities(self, config: dict) -> List[str]:
        """Parse active modalities from config."""
        data_cfg = config.get("data", config)
        modalities = data_cfg.get("modalities", ["visual"])
        if isinstance(modalities, str):
            modalities = [m.strip() for m in modalities.split("+")]
        return modalities

    def _init_weights(self):
        """Initialize any un-initialized weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight') and not hasattr(m, '_initialized'):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and not hasattr(m, '_initialized'):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def encode_modalities(
        self,
        frames: Optional[torch.Tensor] = None,
        events: Optional[Tuple[torch.Tensor, ...]] = None,
        event_duration_us: Optional[float] = None,
        audio: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all available modalities to feature representations.

        Args:
            frames: RGB frames [B, 3, H, W]
            events: Event tuple (x, y, t, p)
            event_duration_us: Event window duration
            audio: Audio waveform [B, T]
            imu_data: IMU time series [B, T, 9]
            pose: Pose data [B, 7]

        Returns:
            Dict with "global" features and "multiscale" feature lists
        """
        global_features = {}
        multiscale_features = {}

        if self.visual_encoder is not None and frames is not None:
            vis_out = self.visual_encoder(frames)
            global_features["visual"] = vis_out["global"]
            multiscale_features["visual"] = vis_out["multiscale"]

        if self.event_encoder is not None and events is not None:
            evt_out = self.event_encoder(events, event_duration_us)
            global_features["event"] = evt_out["global"]
            multiscale_features["event"] = evt_out["multiscale"]

        if self.audio_encoder is not None and audio is not None:
            aud_out = self.audio_encoder(audio)
            global_features["audio"] = aud_out["global"]

        if self.imu_encoder is not None and imu_data is not None:
            imu_out = self.imu_encoder(imu_data)
            global_features["imu"] = imu_out["global"]

        if self.pose_encoder is not None and pose is not None:
            global_features["pose"] = self.pose_encoder(pose)

        return {
            "global": global_features,
            "multiscale": multiscale_features,
        }

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        events: Optional[Tuple[torch.Tensor, ...]] = None,
        event_duration_us: Optional[float] = None,
        audio: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Full forward pass through Eldarin.

        Args:
            frames: RGB frames [B, 3, H, W]
            events: Event tuple (x, y, t, p)
            event_duration_us: Event window duration
            audio: Audio waveform [B, T]
            imu_data: IMU time series [B, T, 9]
            pose: Pose data [B, 7]
            return_all: Return all intermediate outputs

        Returns:
            Dict with detection and tracking outputs
        """
        B = frames.shape[0] if frames is not None else 1

        # Step 1: Multi-modal encoding
        encoded = self.encode_modalities(
            frames=frames,
            events=events,
            event_duration_us=event_duration_us,
            audio=audio,
            imu_data=imu_data,
            pose=pose,
        )
        global_features = encoded["global"]
        multiscale_features = encoded["multiscale"]

        # Step 2: Hierarchy module (cascading fusion)
        # Use visual multiscale features as primary, supplemented by others
        if "visual" in multiscale_features:
            primary_multiscale = multiscale_features["visual"]
        elif "event" in multiscale_features:
            primary_multiscale = multiscale_features["event"]
        else:
            # Fallback: create dummy multiscale from global features
            dummy = list(global_features.values())[0]
            primary_multiscale = [
                dummy.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
                for _ in range(self.hierarchy.num_levels)
            ]

        # Adjust to match hierarchy level count
        if len(primary_multiscale) < self.hierarchy.num_levels:
            # Pad by repeating finest level
            primary_multiscale = list(primary_multiscale) + [primary_multiscale[-1]] * (
                self.hierarchy.num_levels - len(primary_multiscale)
            )
        primary_multiscale = primary_multiscale[:self.hierarchy.num_levels]

        # Average global features as global context
        if global_features:
            global_stack = torch.stack(list(global_features.values()), dim=1)
            global_context = global_stack.mean(dim=1)  # [B, D]
        else:
            global_context = primary_multiscale[0].mean(dim=[-2, -1])

        hier_out = self.hierarchy(
            multiscale_features=primary_multiscale,
            global_features=global_context,
        )
        fused_features = hier_out["fused_features"]  # [B, C, H, W]

        # Step 3: Mixing module (Bayesian cross-modal fusion)
        # Project fused spatial features to global descriptor for mixing
        pooled = F.adaptive_avg_pool2d(fused_features, 1).squeeze(-1).squeeze(-1)

        # Build modality features dict for mixing
        mix_input = {}
        if "visual" in global_features:
            mix_input["visual"] = global_features["visual"]
            # Add visual bias from fused features
            mix_input["visual_fused"] = pooled

        for mod_name, feat in global_features.items():
            if mod_name not in mix_input:
                mix_input[mod_name] = feat

        mix_out = self.mixing(mix_input)

        # Step 4: Detection head
        det_out = self.detection_head(fused_features)

        # Step 5: Tracking
        # Extract per-detection features from fused representation
        # (Simplified: use RoI pooling-like approach)
        det_features = pooled  # [B, D] — use global context for tracking

        track_out = None
        if not self.training:
            # At inference, perform tracking
            # For training, tracking is handled in the loss
            track_out = self.tracking_head(
                detections=det_out["bbox"],  # Simplified — need NMS first
                detection_features=det_features,
                features_for_reid=det_features,
            )

        result = {
            "detection": det_out,
            "tracking": track_out,
            "fused_features": fused_features,
            "hd_representation": mix_out.get("hd_representation"),
            "modality_weights": mix_out.get("modality_weights"),
        }

        if return_all:
            result.update({
                "encoded": encoded,
                "hierarchy": hier_out,
                "mixing": mix_out,
            })

        return result

    def forward_sequence(
        self,
        frame_sequence: List[Dict[str, torch.Tensor]],
        tracks: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a temporal sequence of frames for 4D tracking.

        Args:
            frame_sequence: List of per-frame modality dicts
            tracks: Optional existing tracks

        Returns:
            List of per-frame outputs with tracking
        """
        outputs = []
        for t, frame_data in enumerate(frame_sequence):
            out = self.forward(
                frames=frame_data.get("frames"),
                events=frame_data.get("events"),
                event_duration_us=frame_data.get("event_duration_us"),
                audio=frame_data.get("audio"),
                imu_data=frame_data.get("imu_data"),
                pose=frame_data.get("pose"),
            )

            # Update tracking
            if out["detection"] is not None and self.tracking_head is not None:
                det_out = out["detection"]
                # NMS and post-processing would happen here in full implementation
                track_out = self.tracking_head(
                    detections=det_out["bbox"],
                    detection_features=out["fused_features"].mean(dim=[-2, -1]),
                    features_for_reid=out["fused_features"].mean(dim=[-2, -1]),
                    tracks=tracks,
                    frame_id=t,
                )
                out["tracking"] = track_out
                tracks = track_out.get("tracks", tracks)

            outputs.append(out)

        return outputs

    def to_snn(self) -> "Eldarin":
        """
        Convert model to SNN mode for neuromorphic inference.
        Replaces activation functions with IF/LIF neurons.
        """
        from .snn_layers import SNNConversionHelper
        SNNConversionHelper.convert_relu_to_if(self)
        self.snn_mode = True
        return self

    def reset_snn_state(self):
        """Reset all SNN neuron membrane states."""
        for module in self.modules():
            if hasattr(module, 'reset_state'):
                module.reset_state()

    def get_vsa_representation(
        self,
        frames: Optional[torch.Tensor] = None,
        events: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """
        Extract pure VSA/HDC representation from input modalities.
        Useful for HD memory, retrieval, and symbolic reasoning.

        Returns:
            Hyperdimensional vector [B, hd_dim]
        """
        out = self.forward(frames=frames, events=events)
        return out.get("hd_representation")

    def vsa_native_odometry(
        self,
        image: torch.Tensor,
        imu_reading: Optional[torch.Tensor] = None,
        map_hd: Optional[torch.Tensor] = None,
        num_resonator_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Training-free VSA-native visual odometry path.
        Implements the approach from Renner et al. (2024), arXiv:2209.02000.

        This bypasses all learned components and uses pure VSA/HDC operations:
          1. FPE-encode the input image/event-frame
          2. Optionally fuse IMU via FPE binding (Eq. 10)
          3. Run hierarchical resonator to estimate translation + rotation
          4. Update the allocentric map with anchoring (Eq. 8-9)
          5. Read out via population vector (Eq. 7)

        Args:
            image: [B, H, W] binary/sparse image or event accumulation
            imu_reading: Optional [B, 3] IMU (angular velocities or accelerations)
            map_hd: Optional [B, hd_dim] previous allocentric map
            num_resonator_iterations: Resonator convergence iterations

        Returns:
            Dict with:
                - "translation": Estimated (h, v) in index space
                - "rotation": Estimated rotation index
                - "map": Updated allocentric map
                - "cartesian_confidences": Factor confidence arrays
                - "translation_hd": HD translation kernel
        """
        if self.fpe_encoder is None or self.resonator is None:
            raise RuntimeError(
                "VSA-native path not enabled. Set vsa_native.enabled=true in config."
            )

        device = image.device
        B = image.shape[0]

        # Step 1: FPE-encode the input image (Eq. 1-3)
        # Build codebook if needed, or use direct FPE encoding
        H, W = image.shape[-2], image.shape[-1]
        codebook = self.fpe_encoder.build_codebook(H, W)
        encoded_input = self.fpe_encoder.encode_image(image, codebook)  # [B, hd_dim]

        # Step 2: IMU fusion via FPE binding (Eq. 10)
        # r(t) = r(t-1) * seed^{IMU_reading}
        if self.vsa_imu_enabled and imu_reading is not None:
            # Use IMU to predict state before resonator iteration
            imu_kernel = self.fpe_encoder.translation_kernel(
                imu_reading[:, 0].mean(),  # Aggregate dx
                imu_reading[:, 1].mean(),  # Aggregate dy
            )
            # Pre-rotate encoded input (simplified: bind IMU delta to input)
            encoded_input = encoded_input * imu_kernel.unsqueeze(0)
            encoded_input = F.normalize(encoded_input, p=2, dim=-1)

        # Step 3: Run hierarchical resonator
        resonator_out = self.resonator(
            encoded_input=encoded_input,
            encoded_map=map_hd,
            num_iterations=num_resonator_iterations,
        )

        return resonator_out

    def vsa_native_detect(
        self,
        image: torch.Tensor,
        template_hd: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        VSA-native object detection as transform factorization.
        Uses the resonator to detect objects as "what" x "where" factorizations.

        This is a novel extension of the Renner et al. approach:
          - Template HD vectors encode "what" (object identities)
          - Resonator network factorizes "where" (positions) from encoded scene
          - Detection = which template + which position best compose the scene

        Args:
            image: [B, H, W] input image
            template_hd: Optional [num_templates, hd_dim] pre-computed object templates

        Returns:
            Dict with detected objects: positions (population vector) + template matches
        """
        if self.fpe_encoder is None:
            raise RuntimeError("VSA-native path not enabled.")

        device = image.device
        B = image.shape[0]
        H, W = image.shape[-2], image.shape[-1]

        codebook = self.fpe_encoder.build_codebook(H, W)
        encoded = self.fpe_encoder.encode_image(image, codebook)  # [B, hd_dim]

        if template_hd is not None:
            # Resonator factorization: encoded = template ⊗ position
            # Use a simple 2-factor resonator
            from .vsa_hdc import ResonatorNetwork
            simple_resonator = ResonatorNetwork(
                factor_sizes=[template_hd.shape[0], H * W],
                hd_dim=encoded.shape[-1],
                gamma=0.3,
                nonlinearity="phasor",
            ).to(device)

            # Set the template codebook
            simple_resonator.codebooks[0] = torch.nn.Parameter(template_hd, requires_grad=False)
            simple_resonator.decoders[0] = torch.nn.Parameter(template_hd.T, requires_grad=False)

            result = simple_resonator(encoded, num_iterations=15)

            # Population vector readout for positions
            position_conf = result["factors"][1]  # [B, H*W]
            pos_idx = simple_resonator.population_vector_readout(position_conf)

            # Template matching
            template_conf = result["factors"][0]  # [B, num_templates]
            template_idx = template_conf.argmax(dim=-1)

            return {
                "positions": pos_idx,
                "template_indices": template_idx,
                "template_confidences": template_conf,
                "position_confidences": position_conf,
                "resonator_result": result,
            }

        return {"encoded": encoded}


def create_eldarin(config_path: str = None, config_dict: dict = None, **kwargs) -> Eldarin:
    """
    Factory function to create Eldarin model from config.
    Supports both YAML config path and dict config.

    Args:
        config_path: Path to YAML config file
        config_dict: Configuration dictionary
        **kwargs: Override config values

    Returns:
        Eldarin model instance
    """
    if config_dict is None and config_path is not None:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    # Apply overrides
    config_dict.update(kwargs)

    return Eldarin(config_dict)