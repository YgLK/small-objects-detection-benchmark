"""
Configuration Loader for Benchmark System

This module provides utilities for loading and managing benchmark configurations
from YAML files, environment variables, and command line arguments.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Dataset configuration
    dataset_path: str = "datasets/SkyFusion_yolo"
    dataset_split: str = "test"
    max_samples: int | None = None

    # Model configuration
    model_paths: list[str] = field(
        default_factory=lambda: [
            "models/yolov8m-baseline.pt",
            "models/yolov8m-baseline-aug.pt",
            "models/yolov8m-mosaic.pt",
            "models/yolov8m-mosaic-aug.pt",
        ]
    )

    # Output configuration
    output_dir: str = "materials/benchmarks"
    create_reports: bool = True
    create_visualizations: bool = True

    # Performance configuration
    batch_size: int = 1
    device: str = "auto"  # auto, cpu, cuda
    warmup_runs: int = 3

    # Incremental benchmarking
    use_database: bool = True
    force_rebenchmark: bool = False

    # Reporting configuration
    report_formats: list[str] = field(default_factory=lambda: ["latex", "markdown"])
    include_plots: bool = True
    plot_dpi: int = 300

    # Logging configuration
    log_level: str = "INFO"
    verbose: bool = False


class ConfigLoader:
    """
    Configuration loader with support for YAML files, environment variables,
    and command line arguments.
    """

    def __init__(self, config_dir: str = "src/odc/benchmark/configs"):
        """
        Initialize the configuration loader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)

    def load_config(self, config_file: str | None = None, overrides: dict[str, Any] | None = None) -> BenchmarkConfig:
        """
        Load configuration from file and apply overrides.

        Args:
            config_file: Path to YAML configuration file
            overrides: Dictionary of configuration overrides

        Returns:
            BenchmarkConfig object
        """
        # Start with default configuration
        config_dict = self._get_default_config()

        # Load from file if specified
        if config_file:
            file_config = self._load_config_file(config_file)
            config_dict = self._deep_update(config_dict, file_config)

        # Apply environment variable overrides
        env_config = self._load_env_config()
        config_dict = self._deep_update(config_dict, env_config)

        # Apply command line overrides
        if overrides:
            config_dict = self._deep_update(config_dict, overrides)

        # Create and validate configuration
        return self._create_config(config_dict)

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration as dictionary."""
        default_config = BenchmarkConfig()
        return {
            "dataset_path": default_config.dataset_path,
            "dataset_split": default_config.dataset_split,
            "max_samples": default_config.max_samples,
            "model_paths": default_config.model_paths,
            "output_dir": default_config.output_dir,
            "create_reports": default_config.create_reports,
            "create_visualizations": default_config.create_visualizations,
            "batch_size": default_config.batch_size,
            "device": default_config.device,
            "warmup_runs": default_config.warmup_runs,
            "use_database": default_config.use_database,
            "force_rebenchmark": default_config.force_rebenchmark,
            "report_formats": default_config.report_formats,
            "include_plots": default_config.include_plots,
            "plot_dpi": default_config.plot_dpi,
            "log_level": default_config.log_level,
            "verbose": default_config.verbose,
        }

    def _load_config_file(self, config_file: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_file)

        # Try relative to config directory if not absolute
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config or {}
        except yaml.YAMLError as e:
            logger.error(f"Error loading configuration file {config_path}: {e}")
            return {}

    def _load_env_config(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Define environment variable mappings
        env_mappings = {
            "ODC_DATASET_PATH": "dataset_path",
            "ODC_DATASET_SPLIT": "dataset_split",
            "ODC_MAX_SAMPLES": ("max_samples", int),
            "ODC_OUTPUT_DIR": "output_dir",
            "ODC_BATCH_SIZE": ("batch_size", int),
            "ODC_DEVICE": "device",
            "ODC_WARMUP_RUNS": ("warmup_runs", int),
            "ODC_USE_DATABASE": ("use_database", lambda x: x.lower() in ["true", "1", "yes"]),
            "ODC_FORCE_REBENCHMARK": ("force_rebenchmark", lambda x: x.lower() in ["true", "1", "yes"]),
            "ODC_INCLUDE_PLOTS": ("include_plots", lambda x: x.lower() in ["true", "1", "yes"]),
            "ODC_PLOT_DPI": ("plot_dpi", int),
            "ODC_LOG_LEVEL": "log_level",
            "ODC_VERBOSE": ("verbose", lambda x: x.lower() in ["true", "1", "yes"]),
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_key, tuple):
                    key, converter = config_key
                    try:
                        env_config[key] = converter(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid value for {env_var}: {value} ({e})")
                else:
                    env_config[config_key] = value

        if env_config:
            logger.debug(f"Loaded environment configuration: {list(env_config.keys())}")

        return env_config

    def _deep_update(self, base_dict: dict[str, Any], update_dict: dict[str, Any]) -> dict[str, Any]:
        """Deep update of dictionary."""
        result = base_dict.copy()

        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value

        return result

    def _create_config(self, config_dict: dict[str, Any]) -> BenchmarkConfig:
        """Create BenchmarkConfig from dictionary with validation."""
        try:
            # Handle model_paths as string (convert to list)
            if isinstance(config_dict.get("model_paths"), str):
                config_dict["model_paths"] = [config_dict["model_paths"]]

            # Handle report_formats as string (convert to list)
            if isinstance(config_dict.get("report_formats"), str):
                config_dict["report_formats"] = config_dict["report_formats"].split(",")

            # Create configuration object
            config = BenchmarkConfig(**config_dict)

            # Validate configuration
            self._validate_config(config)

            return config

        except TypeError as e:
            logger.error(f"Invalid configuration parameters: {e}")
            raise ValueError(f"Configuration validation failed: {e}")

    def _validate_config(self, config: BenchmarkConfig):
        """Validate configuration parameters."""
        # Validate dataset path
        if not Path(config.dataset_path).exists():
            logger.warning(f"Dataset path does not exist: {config.dataset_path}")

        # Validate model paths
        missing_models = []
        for model_path in config.model_paths:
            if not Path(model_path).exists():
                missing_models.append(model_path)

        if missing_models:
            logger.warning(f"Model files not found: {missing_models}")

        # Validate output directory
        try:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot create output directory {config.output_dir}: {e}")
            raise

        # Validate numeric parameters
        if config.max_samples is not None and config.max_samples <= 0:
            raise ValueError("max_samples must be positive")

        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if config.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")

        if config.plot_dpi <= 0:
            raise ValueError("plot_dpi must be positive")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

        # Validate report formats
        valid_formats = ["latex", "markdown", "csv"]
        invalid_formats = [fmt for fmt in config.report_formats if fmt not in valid_formats]
        if invalid_formats:
            raise ValueError(f"Invalid report formats: {invalid_formats}. Valid: {valid_formats}")

    def save_config(self, config: BenchmarkConfig, output_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            "dataset_path": config.dataset_path,
            "dataset_split": config.dataset_split,
            "max_samples": config.max_samples,
            "model_paths": config.model_paths,
            "output_dir": config.output_dir,
            "create_reports": config.create_reports,
            "create_visualizations": config.create_visualizations,
            "batch_size": config.batch_size,
            "device": config.device,
            "warmup_runs": config.warmup_runs,
            "use_database": config.use_database,
            "force_rebenchmark": config.force_rebenchmark,
            "report_formats": config.report_formats,
            "include_plots": config.include_plots,
            "plot_dpi": config.plot_dpi,
            "log_level": config.log_level,
            "verbose": config.verbose,
        }

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {output_path}")

    def create_sample_configs(self):
        """Create sample configuration files."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Quick test configuration
        quick_config = {
            "max_samples": 20,
            "model_paths": ["models/yolov8m-baseline.pt"],
            "create_reports": False,
            "create_visualizations": False,
            "use_database": False,
        }

        with open(self.config_dir / "quick_test.yaml", "w") as f:
            yaml.dump(quick_config, f, default_flow_style=False, indent=2)

        # Full benchmark configuration
        full_config = {
            "max_samples": None,  # Use all samples
            "create_reports": True,
            "create_visualizations": True,
            "include_plots": True,
            "report_formats": ["latex", "markdown"],
            "use_database": True,
        }

        with open(self.config_dir / "full_benchmark.yaml", "w") as f:
            yaml.dump(full_config, f, default_flow_style=False, indent=2)

        logger.info(f"Sample configurations created in {self.config_dir}")


def load_benchmark_config(config_file: str | None = None, **overrides) -> BenchmarkConfig:
    """
    Convenience function to load benchmark configuration.

    Args:
        config_file: Path to configuration file
        **overrides: Configuration overrides as keyword arguments

    Returns:
        BenchmarkConfig object
    """
    loader = ConfigLoader()
    return loader.load_config(config_file, overrides)
