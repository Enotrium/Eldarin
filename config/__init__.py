"""Eldarin configuration package with validation."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)

# Config schema version for forward/backward compatibility
CONFIG_VERSION = "2.0"


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_root: str = "data"
    val_root: str = ""
    modalities: str = "rgb"
    batch_size: int = 16
    img_size: int = 640
    num_workers: int = 4
    max_samples: int = -1
    dataset: str = "visdrone"
    use_mosaic: bool = True
    use_hsv: bool = True
    use_flip: bool = True
    num_classes: int = 10
    event_window_us: int = 50000
    event_polarity: bool = True
    audio_sample_rate: int = 16000
    audio_duration: float = 2.0
    imu_rate: int = 200
    augment: bool = True

    def validate(self) -> list:
        errors = []
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.img_size < 32:
            errors.append("img_size must be >= 32")
        if self.num_workers < 0:
            errors.append("num_workers must be >= 0")
        if self.num_classes < 1:
            errors.append("num_classes must be >= 1")
        if self.event_window_us <= 0:
            errors.append("event_window_us must be > 0")
        return errors


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "resnet50"
    pretrained: bool = True
    feature_dim: int = 256
    hidden_dim: int = 512
    hd_dim: int = 8192
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    use_fpe: bool = True
    use_hd_classifier: bool = True
    use_sdm: bool = False
    encoders: Dict[str, Any] = field(default_factory=dict)
    hierarchy_depth: int = 3
    mixing_strategy: str = "bayesian"
    tracking: bool = True
    vsa_dtype: str = "bipolar"
    vsa_binding: str = "circular"

    def validate(self) -> list:
        errors = []
        if self.hd_dim < 64:
            errors.append("hd_dim must be >= 64")
        if self.feature_dim < 1:
            errors.append("feature_dim must be >= 1")
        if self.hidden_dim < 1:
            errors.append("hidden_dim must be >= 1")
        if self.num_heads < 1:
            errors.append("num_heads must be >= 1")
        if self.num_layers < 1:
            errors.append("num_layers must be >= 1")
        if self.dropout < 0 or self.dropout > 1:
            errors.append("dropout must be in [0, 1]")
        if self.hierarchy_depth < 1:
            errors.append("hierarchy_depth must be >= 1")
        if self.mixing_strategy not in ("bayesian", "attention", "vsa", "concat"):
            errors.append(f"Unknown mixing_strategy: {self.mixing_strategy}")
        if self.vsa_dtype not in ("bipolar", "binary", "complex"):
            errors.append(f"Unknown vsa_dtype: {self.vsa_dtype}")
        if self.vsa_binding not in ("circular", "xor"):
            errors.append(f"Unknown vsa_binding: {self.vsa_binding}")
        return errors


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 5
    gradient_clip: float = 10.0
    amp: bool = True
    save_interval: int = 5
    eval_interval: int = 1
    loss: Dict[str, float] = field(default_factory=lambda: {
        "box": 7.5,
        "cls": 0.5,
        "obj": 1.0,
        "track": 2.0,
        "vsa_consistency": 0.1,
        "temporal_smooth": 0.5,
    })

    def validate(self) -> list:
        errors = []
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.lr <= 0:
            errors.append("lr must be > 0")
        if self.weight_decay < 0:
            errors.append("weight_decay must be >= 0")
        if self.momentum < 0 or self.momentum > 1:
            errors.append("momentum must be in [0, 1]")
        if self.warmup_epochs < 0:
            errors.append("warmup_epochs must be >= 0")
        if self.gradient_clip <= 0:
            errors.append("gradient_clip must be > 0")
        return errors


@dataclass
class EldarinConfig:
    """Complete Eldarin configuration with validation."""
    version: str = CONFIG_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str) -> "EldarinConfig":
        """Load and validate configuration from YAML file."""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}

        cfg = cls(_raw=raw)

        # Check version
        file_version = raw.get("version", "1.0")
        if file_version != CONFIG_VERSION:
            logger.warning(
                f"Config version {file_version} differs from code version {CONFIG_VERSION}. "
                f"Some keys may be missing or have changed."
            )

        # Parse data config
        data_raw = raw.get("data", {})
        cfg.data = DataConfig(
            data_root=data_raw.get("data_root", "data"),
            val_root=data_raw.get("val_root", ""),
            modalities=data_raw.get("modalities", "rgb"),
            batch_size=data_raw.get("batch_size", 16),
            img_size=data_raw.get("img_size", 640),
            num_workers=data_raw.get("num_workers", 4),
            max_samples=data_raw.get("max_samples", -1),
            dataset=data_raw.get("dataset", "visdrone"),
            use_mosaic=data_raw.get("use_mosaic", True),
            use_hsv=data_raw.get("use_hsv", True),
            use_flip=data_raw.get("use_flip", True),
            num_classes=data_raw.get("num_classes", 10),
            event_window_us=data_raw.get("event_window_us", 50000),
            augment=data_raw.get("augment", True),
        )

        # Parse model config
        model_raw = raw.get("model", {})
        cfg.model = ModelConfig(
            backbone=model_raw.get("backbone", "resnet50"),
            pretrained=model_raw.get("pretrained", True),
            feature_dim=model_raw.get("feature_dim", 256),
            hidden_dim=model_raw.get("hidden_dim", 512),
            hd_dim=model_raw.get("hd_dim", 8192),
            num_heads=model_raw.get("num_heads", 8),
            num_layers=model_raw.get("num_layers", 6),
            dropout=model_raw.get("dropout", 0.1),
            use_fpe=model_raw.get("use_fpe", True),
            use_hd_classifier=model_raw.get("use_hd_classifier", True),
            use_sdm=model_raw.get("use_sdm", False),
            encoders=model_raw.get("encoders", {}),
            hierarchy_depth=model_raw.get("hierarchy_depth", 3),
            mixing_strategy=model_raw.get("mixing_strategy", "bayesian"),
            tracking=model_raw.get("tracking", True),
            vsa_dtype=model_raw.get("vsa_dtype", "bipolar"),
            vsa_binding=model_raw.get("vsa_binding", "circular"),
        )

        # Parse training config
        train_raw = raw.get("training", {})
        cfg.training = TrainingConfig(
            epochs=train_raw.get("epochs", 100),
            lr=train_raw.get("lr", 0.001),
            weight_decay=train_raw.get("weight_decay", 0.0005),
            momentum=train_raw.get("momentum", 0.937),
            warmup_epochs=train_raw.get("warmup_epochs", 5),
            gradient_clip=train_raw.get("gradient_clip", 10.0),
            amp=train_raw.get("amp", True),
            save_interval=train_raw.get("save_interval", 5),
            eval_interval=train_raw.get("eval_interval", 1),
            loss=train_raw.get("loss", {}),
        )

        return cfg

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]] = None) -> "EldarinConfig":
        """Load and validate configuration from a dictionary."""
        if config_dict is None:
            config_dict = {}
        cfg = cls(_raw=config_dict)

        data_raw = config_dict.get("data", {})
        cfg.data = DataConfig(
            data_root=data_raw.get("data_root", "data"),
            val_root=data_raw.get("val_root", ""),
            modalities=data_raw.get("modalities", "rgb"),
            batch_size=data_raw.get("batch_size", 16),
            img_size=data_raw.get("img_size", 640),
            num_workers=data_raw.get("num_workers", 4),
            max_samples=data_raw.get("max_samples", -1),
            dataset=data_raw.get("dataset", "visdrone"),
            num_classes=data_raw.get("num_classes", 10),
            event_window_us=data_raw.get("event_window_us", 50000),
        )

        model_raw = config_dict.get("model", {})
        cfg.model = ModelConfig(
            backbone=model_raw.get("backbone", "resnet50"),
            pretrained=model_raw.get("pretrained", True),
            feature_dim=model_raw.get("feature_dim", 256),
            hidden_dim=model_raw.get("hidden_dim", 512),
            hd_dim=model_raw.get("hd_dim", 8192),
            num_heads=model_raw.get("num_heads", 8),
            num_layers=model_raw.get("num_layers", 6),
            dropout=model_raw.get("dropout", 0.1),
            use_fpe=model_raw.get("use_fpe", True),
            hierarchy_depth=model_raw.get("hierarchy_depth", 3),
            mixing_strategy=model_raw.get("mixing_strategy", "bayesian"),
            tracking=model_raw.get("tracking", True),
            vsa_dtype=model_raw.get("vsa_dtype", "bipolar"),
            vsa_binding=model_raw.get("vsa_binding", "circular"),
        )

        train_raw = config_dict.get("training", {})
        cfg.training = TrainingConfig(
            epochs=train_raw.get("epochs", 100),
            lr=train_raw.get("lr", 0.001),
            weight_decay=train_raw.get("weight_decay", 0.0005),
            momentum=train_raw.get("momentum", 0.937),
            warmup_epochs=train_raw.get("warmup_epochs", 5),
            gradient_clip=train_raw.get("gradient_clip", 10.0),
            amp=train_raw.get("amp", True),
            save_interval=train_raw.get("save_interval", 5),
            eval_interval=train_raw.get("eval_interval", 1),
            loss=train_raw.get("loss", {}),
        )

        return cfg

    def validate(self) -> list:
        """Validate all config sections. Returns list of error strings (empty = valid)."""
        errors = []
        for section_name, section in [
            ("data", self.data),
            ("model", self.model),
            ("training", self.training),
        ]:
            section_errors = section.validate()
            for e in section_errors:
                errors.append(f"[{section_name}] {e}")
        return errors

    def validate_or_raise(self):
        """Validate and raise ValueError on first error."""
        errors = self.validate()
        if errors:
            raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary (compatible with existing code)."""
        data_dict = asdict(self.data)
        model_dict = asdict(self.model)
        train_dict = asdict(self.training)

        return {
            "version": self.version,
            "data": data_dict,
            "model": model_dict,
            "training": train_dict,
        }


def load_config(path: str, validate: bool = True) -> EldarinConfig:
    """
    Load configuration from a YAML file with validation.

    Args:
        path: Path to YAML config file
        validate: If True, validates all config sections and raises on error

    Returns:
        Validated EldarinConfig instance

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = EldarinConfig.from_yaml(path)

    if validate:
        errors = cfg.validate()
        if errors:
            raise ValueError(
                f"Config validation failed for {path}:\n  " + "\n  ".join(errors)
            )

    logger.info(f"Loaded config from {path} (version {cfg.version}, {cfg.model.hd_dim}D HD)")
    return cfg


def load_config_dict(path: str) -> dict:
    """Load config as raw dict (for backwards compatibility with existing main.py)."""
    cfg = load_config(path, validate=True)
    return cfg.to_dict()