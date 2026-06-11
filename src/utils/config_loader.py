"""
Configuration loader for the Food Safety ML Pipeline.
Handles loading and validating YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_data_path: str = "data/raw/food_safety_data.csv"
    processed_data_path: str = "data/processed/"
    test_split_size: float = 0.2
    val_split_size: float = 0.1
    random_state: int = 42
    target_column: str = "safety_status"


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    numeric_features: list = field(default_factory=list)
    categorical_features: list = field(default_factory=list)
    scaling_method: str = "standard"
    handle_missing: str = "median"


@dataclass
class ModelConfig:
    """Model configuration."""
    type: str = "xgboost"
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42
    use_gpu: bool = False


@dataclass
class HyperparameterTuningConfig:
    """Hyperparameter tuning configuration."""
    enabled: bool = True
    n_trials: int = 50
    timeout_seconds: int = 3600
    metric: str = "f1"


@dataclass
class TrainingConfig:
    """Training configuration."""
    early_stopping_rounds: int = 10
    cross_validation_folds: int = 5
    class_weight_balance: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"
    ])
    threshold: float = 0.5


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_path: str = "models/best_model.joblib"
    scaler_path: str = "models/scaler.joblib"
    encoder_path: str = "models/encoder.joblib"
    prediction_threshold: float = 0.5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs/"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class PathsConfig:
    """Path configuration."""
    base_dir: str = "."
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    notebooks_dir: str = "notebooks"


class Config:
    """Main configuration class that loads and manages all pipeline settings."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.
                        If None, uses default config location.
        """
        if config_path is None:
            # Try common locations
            possible_paths = [
                Path("config/config.yaml"),
                Path("../config/config.yaml"),
                Path(__file__).parent.parent.parent / "config" / "config.yaml",
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
            else:
                raise FileNotFoundError(
                    "Configuration file not found. Please provide a valid config_path."
                )

        self.config_path = Path(config_path)
        self._raw_config = self._load_yaml()
        self._parse_config()

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _parse_config(self):
        """Parse raw config into structured dataclasses."""
        cfg = self._raw_config

        self.data = DataConfig(**cfg.get("data", {}))
        self.preprocessing = PreprocessingConfig(**cfg.get("preprocessing", {}))
        self.model = ModelConfig(**cfg.get("model", {}))
        self.hyperparameter_tuning = HyperparameterTuningConfig(
            **cfg.get("hyperparameter_tuning", {})
        )
        self.training = TrainingConfig(**cfg.get("training", {}))
        self.evaluation = EvaluationConfig(**cfg.get("evaluation", {}))
        self.inference = InferenceConfig(**cfg.get("inference", {}))
        self.logging = LoggingConfig(**cfg.get("logging", {}))
        self.paths = PathsConfig(**cfg.get("paths", {}))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Dot-separated key (e.g., "data.test_split_size")
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._raw_config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def save(self, output_path: str):
        """Save current configuration to a YAML file."""
        with open(output_path, "w") as f:
            yaml.dump(self._raw_config, f, default_flow_style=False)

    def __repr__(self) -> str:
        return f"Config({self.config_path})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config object with all pipeline settings
    """
    return Config(config_path)
