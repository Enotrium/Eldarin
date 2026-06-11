"""
Tests for config loading and validation.

Covers:
  - Config schema validation
  - Config completeness checks
  - Missing key detection
  - Version compatibility warnings
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigValidation:
    """Test config dataclass validation."""

    def test_valid_default_config(self):
        """Default config validates without errors."""
        from config import DataConfig, ModelConfig, TrainingConfig

        data_cfg = DataConfig()
        model_cfg = ModelConfig()
        train_cfg = TrainingConfig()

        assert len(data_cfg.validate()) == 0
        assert len(model_cfg.validate()) == 0
        assert len(train_cfg.validate()) == 0

    def test_data_config_invalid_batch_size(self):
        """DataConfig rejects invalid batch_size."""
        from config import DataConfig

        cfg = DataConfig(batch_size=0)
        errors = cfg.validate()
        assert len(errors) > 0
        assert any("batch_size" in e for e in errors)

    def test_model_config_invalid_dropout(self):
        """ModelConfig rejects out-of-range dropout."""
        from config import ModelConfig

        cfg = ModelConfig(dropout=1.5)
        errors = cfg.validate()
        assert len(errors) > 0
        assert any("dropout" in e for e in errors)

    def test_model_config_invalid_hd_dim(self):
        """ModelConfig rejects too-small hd_dim."""
        from config import ModelConfig

        cfg = ModelConfig(hd_dim=32)
        errors = cfg.validate()
        assert len(errors) > 0
        assert any("hd_dim" in e for e in errors)

    def test_model_config_unknown_mixing_strategy(self):
        """ModelConfig rejects unknown mixing strategy."""
        from config import ModelConfig

        cfg = ModelConfig(mixing_strategy="unknown_method")
        errors = cfg.validate()
        assert len(errors) > 0
        assert any("mixing_strategy" in e for e in errors)

    def test_training_config_invalid_lr(self):
        """TrainingConfig rejects zero learning rate."""
        from config import TrainingConfig

        cfg = TrainingConfig(lr=0.0)
        errors = cfg.validate()
        assert len(errors) > 0
        assert any("lr" in e for e in errors)

    def test_training_config_invalid_momentum(self):
        """TrainingConfig rejects out-of-range momentum."""
        from config import TrainingConfig

        cfg1 = TrainingConfig(momentum=-0.1)
        cfg2 = TrainingConfig(momentum=1.5)

        assert len(cfg1.validate()) > 0
        assert len(cfg2.validate()) > 0

    def test_training_config_invalid_gradient_clip(self):
        """TrainingConfig rejects non-positive gradient clip."""
        from config import TrainingConfig

        cfg = TrainingConfig(gradient_clip=0.0)
        errors = cfg.validate()
        assert len(errors) > 0
        assert any("gradient_clip" in e for e in errors)


class TestEldarinConfig:
    """Test full EldarinConfig validation."""

    def test_from_dict_creates_valid_config(self):
        """from_dict creates a validatable config."""
        from config import EldarinConfig

        cfg = EldarinConfig.from_dict({
            "data": {"batch_size": 32, "num_classes": 10},
            "model": {"feature_dim": 256, "hd_dim": 4096},
            "training": {"epochs": 50, "lr": 0.001},
        })

        errors = cfg.validate()
        assert len(errors) == 0, f"Config should be valid, got errors: {errors}"

    def test_to_dict_roundtrips(self):
        """to_dict produces a dictionary compatible with existing code."""
        from config import EldarinConfig

        cfg = EldarinConfig.from_dict({
            "data": {"batch_size": 16},
            "model": {"hd_dim": 1024},
            "training": {"epochs": 10},
        })

        d = cfg.to_dict()
        assert "data" in d
        assert "model" in d
        assert "training" in d
        assert d["data"]["batch_size"] == 16
        assert d["model"]["hd_dim"] == 1024
        assert d["training"]["epochs"] == 10

    def test_validate_or_raise_passes(self):
        """validate_or_raise doesn't raise on valid config."""
        from config import EldarinConfig

        cfg = EldarinConfig.from_dict({})
        # Should not raise
        cfg.validate_or_raise()

    def test_validate_or_raise_fails(self):
        """validate_or_raise raises on invalid config."""
        from config import EldarinConfig

        cfg = EldarinConfig.from_dict({
            "model": {"hd_dim": 0, "dropout": 2.0},
        })

        with pytest.raises(ValueError):
            cfg.validate_or_raise()

    def test_from_yaml_loads_config(self):
        """from_yaml loads a valid YAML config."""
        from config import EldarinConfig
        import yaml

        yaml_content = {
            "version": "2.0",
            "data": {
                "data_root": "./test_data",
                "batch_size": 8,
                "img_size": 320,
                "num_classes": 5,
            },
            "model": {
                "backbone": "resnet18",
                "hd_dim": 2048,
                "feature_dim": 128,
                "hierarchy_depth": 2,
                "mixing_strategy": "attention",
            },
            "training": {
                "epochs": 20,
                "lr": 0.0005,
                "gradient_clip": 5.0,
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            cfg = EldarinConfig.from_yaml(temp_path)
            errors = cfg.validate()
            assert len(errors) == 0, f"Loaded config should be valid: {errors}"
            assert cfg.data.batch_size == 8
            assert cfg.model.hd_dim == 2048
            assert cfg.training.epochs == 20
        finally:
            os.unlink(temp_path)

    def test_version_mismatch_warns(self):
        """Version mismatch should not crash, just warn."""
        from config import EldarinConfig

        cfg = EldarinConfig.from_dict({"version": "1.0"})
        # Should still create valid config from defaults
        assert cfg.data.num_classes == 10  # default


class TestLoadConfig:
    """Test the load_config helper function."""

    def test_load_config_valid_file(self):
        """load_config loads and validates a YAML file."""
        from config import load_config
        import yaml

        yaml_content = {
            "version": "2.0",
            "data": {"batch_size": 16, "num_classes": 10},
            "model": {"hd_dim": 4096, "feature_dim": 256},
            "training": {"epochs": 50},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            cfg = load_config(temp_path, validate=True)
            assert cfg.model.hd_dim == 4096
        finally:
            os.unlink(temp_path)

    def test_load_config_missing_file(self):
        """load_config raises on missing file."""
        from config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_invalid_fails_validation(self):
        """load_config raises validation error on invalid config."""
        from config import load_config
        import yaml

        yaml_content = {
            "model": {"hd_dim": -100},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                load_config(temp_path, validate=True)
        finally:
            os.unlink(temp_path)

    def test_load_config_dict(self):
        """load_config_dict returns raw dict compatible with existing main.py."""
        from config import load_config_dict
        import yaml

        yaml_content = {
            "version": "2.0",
            "data": {"batch_size": 32},
            "model": {"hd_dim": 2048},
            "training": {"epochs": 30},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            cfg_dict = load_config_dict(temp_path)
            assert isinstance(cfg_dict, dict)
            assert cfg_dict["data"]["batch_size"] == 32
            assert cfg_dict["model"]["hd_dim"] == 2048
        finally:
            os.unlink(temp_path)