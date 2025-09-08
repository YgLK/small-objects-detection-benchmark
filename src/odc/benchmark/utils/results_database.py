"""
Results Database for Incremental Benchmarking

This module provides persistent storage for benchmark results with model fingerprinting
to enable incremental benchmarking and avoid re-running unchanged models.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


logger = logging.getLogger(__name__)


@dataclass
class ModelFingerprint:
    """Model fingerprint for change detection."""

    file_size: int
    modified_time: str
    file_path: str
    config_hash: str | None = None

    def __eq__(self, other) -> bool:
        """Compare fingerprints for equality."""
        if not isinstance(other, ModelFingerprint):
            return False
        return (
            self.file_size == other.file_size
            and self.modified_time == other.modified_time
            and self.file_path == other.file_path
            and self.config_hash == other.config_hash
        )


@dataclass
class BenchmarkRun:
    """Information about a benchmark run."""

    timestamp: str
    models_benchmarked: list[str]
    sample_count: int
    results_path: str
    duration_seconds: float | None = None
    notes: str | None = None


class ResultsDatabase:
    """
    Persistent database for benchmark results with incremental benchmarking support.

    Features:
    - Model fingerprinting for change detection
    - Persistent JSON storage
    - Incremental benchmarking logic
    - Historical tracking of benchmark runs
    """

    def __init__(self, database_dir: str = "materials/benchmarks/database"):
        """
        Initialize the results database.

        Args:
            database_dir: Directory to store database files
        """
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.database_dir / "benchmark_results.json"
        self.fingerprints_file = self.database_dir / "model_fingerprints.json"
        self.metadata_file = self.database_dir / "dataset_metadata.json"

        self._data = self._load_database()

    def _load_database(self) -> dict[str, Any]:
        """Load the database from disk or create new if doesn't exist."""
        if self.results_file.exists():
            try:
                with open(self.results_file) as f:
                    data = json.load(f)
                logger.info(f"Loaded existing database with {len(data.get('models', {}))} models")
                return data
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error loading database: {e}. Creating new database.")

        # Create new database structure
        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0",
            },
            "models": {},
            "benchmark_runs": [],
        }

    def _save_database(self):
        """Save the database to disk."""
        self._data["metadata"]["last_updated"] = datetime.now().isoformat()

        with open(self.results_file, "w") as f:
            json.dump(self._data, f, indent=2)

        logger.debug(f"Database saved to {self.results_file}")

    def get_model_fingerprint(self, model_path: str) -> ModelFingerprint:
        """
        Generate fingerprint for a model file.

        Args:
            model_path: Path to the model file

        Returns:
            ModelFingerprint object
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        stat = model_path.stat()

        # Generate config hash if possible (for future extensibility)
        config_hash = None
        try:
            # This could be extended to hash model configuration
            # For now, use file content hash for small files or just metadata
            if stat.st_size < 1024 * 1024:  # 1MB threshold
                with open(model_path, "rb") as f:
                    content = f.read()
                config_hash = hashlib.md5(content).hexdigest()[:16]
        except Exception as e:
            logger.debug(f"Could not generate config hash for {model_path}: {e}")

        return ModelFingerprint(
            file_size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            file_path=str(model_path),
            config_hash=config_hash,
        )

    def get_stored_fingerprint(self, model_path: str) -> ModelFingerprint | None:
        """
        Get stored fingerprint for a model.

        Args:
            model_path: Path to the model file

        Returns:
            Stored ModelFingerprint or None if not found
        """
        model_key = str(Path(model_path).name)
        model_data = self._data["models"].get(model_key)

        if not model_data or "fingerprint" not in model_data:
            return None

        fp_data = model_data["fingerprint"]
        return ModelFingerprint(**fp_data)

    def should_benchmark_model(self, model_path: str) -> bool:
        """
        Determine if a model should be benchmarked based on fingerprint comparison.

        Args:
            model_path: Path to the model file

        Returns:
            True if model should be benchmarked, False otherwise
        """
        try:
            current_fingerprint = self.get_model_fingerprint(model_path)
            stored_fingerprint = self.get_stored_fingerprint(model_path)

            if stored_fingerprint is None:
                logger.info(f"Model {model_path} not found in database - will benchmark")
                return True

            if current_fingerprint != stored_fingerprint:
                logger.info(f"Model {model_path} has changed - will benchmark")
                return True

            logger.info(f"Model {model_path} unchanged - skipping benchmark")
            return False

        except Exception as e:
            logger.warning(f"Error checking model fingerprint for {model_path}: {e}")
            return True  # Benchmark on error to be safe

    def get_models_to_benchmark(self, model_paths: list[str]) -> list[str]:
        """
        Filter list of model paths to only those that need benchmarking.

        Args:
            model_paths: List of model file paths

        Returns:
            List of model paths that need benchmarking
        """
        return [path for path in model_paths if self.should_benchmark_model(path)]

    def store_model_results(self, model_path: str, results: dict[str, Any]):
        """
        Store benchmark results for a model.

        Args:
            model_path: Path to the model file
            results: Benchmark results dictionary
        """
        model_key = str(Path(model_path).name)
        fingerprint = self.get_model_fingerprint(model_path)

        # Convert results to JSON-serializable format
        if hasattr(results, "dict"):
            # If it's a Pydantic model, convert to dict
            serializable_results = results.dict()
        elif isinstance(results, dict):
            # If it's already a dict, ensure it's JSON serializable
            serializable_results = self._make_json_serializable(results)
        else:
            # Convert other types to dict
            serializable_results = dict(results)

        self._data["models"][model_key] = {
            "fingerprint": asdict(fingerprint),
            "results": serializable_results,
            "last_benchmarked": datetime.now().isoformat(),
        }

        self._save_database()
        logger.info(f"Stored results for model {model_key}")

    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if hasattr(obj, "dict"):
            return obj.dict()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

    def get_model_results(self, model_path: str) -> dict[str, Any] | None:
        """
        Get stored results for a model.

        Args:
            model_path: Path to the model file

        Returns:
            Stored results dictionary or None if not found
        """
        model_key = str(Path(model_path).name)
        model_data = self._data["models"].get(model_key)

        if model_data and "results" in model_data:
            return model_data["results"]

        return None

    def get_all_model_results(self) -> dict[str, dict[str, Any]]:
        """
        Get all stored model results.

        Returns:
            Dictionary mapping model names to their results
        """
        results = {}
        for model_key, model_data in self._data["models"].items():
            if "results" in model_data:
                results[model_key] = model_data["results"]

        return results

    def record_benchmark_run(self, run_info: BenchmarkRun):
        """
        Record information about a benchmark run.

        Args:
            run_info: BenchmarkRun object with run information
        """
        self._data["benchmark_runs"].append(asdict(run_info))
        self._save_database()
        logger.info(f"Recorded benchmark run: {run_info.timestamp}")

    def get_benchmark_history(self) -> list[BenchmarkRun]:
        """
        Get history of benchmark runs.

        Returns:
            List of BenchmarkRun objects
        """
        return [BenchmarkRun(**run_data) for run_data in self._data["benchmark_runs"]]

    def get_database_stats(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        return {
            "total_models": len(self._data["models"]),
            "total_runs": len(self._data["benchmark_runs"]),
            "created": self._data["metadata"]["created"],
            "last_updated": self._data["metadata"]["last_updated"],
            "database_size_mb": self.results_file.stat().st_size / (1024 * 1024) if self.results_file.exists() else 0,
        }

    def export_to_dataframe(self):
        """
        Export results to pandas DataFrame for analysis.

        Returns:
            pandas.DataFrame with all model results
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame export")

        rows = []
        for model_name, model_data in self._data["models"].items():
            if "results" not in model_data:
                continue

            results = model_data["results"]
            row = {"model_name": model_name}

            # Flatten nested dictionaries
            def flatten_dict(d, prefix=""):
                for key, value in d.items():
                    if isinstance(value, dict):
                        flatten_dict(value, f"{prefix}{key}_")
                    else:
                        row[f"{prefix}{key}"] = value

            flatten_dict(results)
            rows.append(row)

        return pd.DataFrame(rows)

    def cleanup_old_runs(self, keep_last_n: int = 10):
        """
        Clean up old benchmark run records.

        Args:
            keep_last_n: Number of recent runs to keep
        """
        if len(self._data["benchmark_runs"]) > keep_last_n:
            # Sort by timestamp and keep only the most recent
            runs = sorted(self._data["benchmark_runs"], key=lambda x: x["timestamp"], reverse=True)
            self._data["benchmark_runs"] = runs[:keep_last_n]
            self._save_database()
            logger.info(f"Cleaned up old benchmark runs, kept {keep_last_n} most recent")
