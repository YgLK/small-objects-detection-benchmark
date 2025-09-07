"""Enhanced results database with Pydantic validation and multiple storage formats.

This module provides an improved results database that:
- Uses Pydantic models for data validation and type safety
- Supports multiple storage formats (JSON, Parquet, CSV)
- Provides better performance and data integrity
- Enables easy querying and analysis
"""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from pydantic import ValidationError

from ..models.data_models import (
    BenchmarkResults,
    DatasetInfo,
    DetectionMetrics,
    DetectionStatistics,
    ModelMetadata,
    ModelResult,
    PerformanceMetrics,
)


class EnhancedResultsDatabase:
    """Enhanced persistent storage with Pydantic validation and multiple formats."""

    def __init__(self, database_path: str, storage_format: str = "parquet"):
        """Initialize the enhanced results database.

        Args:
            database_path: Path to the database directory
            storage_format: Storage format ('json', 'parquet', 'both')
        """
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)

        self.storage_format = storage_format

        # File paths for different formats
        self.json_file = self.database_path / "benchmark_results.json"
        self.parquet_file = self.database_path / "benchmark_results.parquet"
        self.metadata_file = self.database_path / "database_metadata.json"

        # Load existing data
        self.results_data = self._load_results()
        self.metadata = self._load_metadata()

    def _load_results(self) -> dict[str, Any]:
        """Load existing results from database."""
        # Try to load from Parquet first (better performance)
        if self.parquet_file.exists() and self.storage_format in ["parquet", "both"]:
            try:
                return self._load_from_parquet()
            except Exception as e:
                print(f"Warning: Failed to load from Parquet: {e}")

        # Fallback to JSON
        if self.json_file.exists():
            try:
                with open(self.json_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Failed to load from JSON: {e}")

        # Return empty structure
        return {"database_version": "2.0", "created_at": datetime.now().isoformat(), "models": {}, "benchmark_runs": []}

    def _load_from_parquet(self) -> dict[str, Any]:
        """Load results from Parquet format."""
        df = pd.read_parquet(self.parquet_file)

        # Convert DataFrame back to nested structure
        results_data = {
            "database_version": "2.0",
            "created_at": datetime.now().isoformat(),
            "models": {},
            "benchmark_runs": [],
        }

        # Group by model and reconstruct nested structure
        for model_name in df["model_name"].unique():
            model_df = df[df["model_name"] == model_name].iloc[0]

            # Reconstruct model data from flattened DataFrame
            model_data = {
                "model_path": model_df.get("model_path", ""),
                "model_config": json.loads(model_df.get("model_config", "{}")),
                "model_hash": model_df.get("model_hash", ""),
                "last_evaluated": model_df.get("last_evaluated", ""),
                "results": {
                    "model_name": model_name,
                    "model_metadata": {
                        "name": model_name,
                        "version": model_df.get("version", ""),
                        "parameters": model_df.get("parameters", 0),
                        "model_size_mb": model_df.get("model_size_mb", 0),
                        "gflops": model_df.get("gflops", None),
                        "framework": model_df.get("framework", "pytorch"),
                    },
                    "detection_metrics": {
                        "mAP@0.5": model_df.get("map_50", 0),
                        "mAP@0.75": model_df.get("map_75", 0),
                        "mAP@[0.5:0.05:0.95]": model_df.get("map_coco", 0),
                        # Add class metrics if available
                    },
                    "performance_metrics": {
                        "inference_time_ms": model_df.get("inference_time_ms", 0),
                        "fps": model_df.get("fps", 0),
                        "memory_usage_mb": model_df.get("memory_usage_mb", 0),
                        "parameters": model_df.get("parameters", 0),
                        "model_size_mb": model_df.get("model_size_mb", 0),
                        "gflops": model_df.get("gflops", None),
                    },
                    "statistics": {
                        "total_detections": model_df.get("total_detections", 0),
                        "avg_detections_per_image": model_df.get("avg_detections_per_image", 0),
                        "class_detection_counts": {},
                    },
                },
            }

            results_data["models"][model_name] = model_data

        return results_data

    def _load_metadata(self) -> dict[str, Any]:
        """Load database metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        return {
            "last_updated": None,
            "total_models": 0,
            "total_benchmark_runs": 0,
            "dataset_info": {},
            "storage_format": self.storage_format,
            "database_version": "2.0",
        }

    def _save_results(self):
        """Save results to database file(s)."""
        if self.storage_format in ["json", "both"]:
            self._save_to_json()

        if self.storage_format in ["parquet", "both"]:
            self._save_to_parquet()

    def _save_to_json(self):
        """Save results to JSON format."""
        with open(self.json_file, "w") as f:
            json.dump(self.results_data, f, indent=2, default=str)

    def _save_to_parquet(self):
        """Save results to Parquet format for better performance."""
        # Flatten the nested structure for DataFrame storage
        flattened_data = []

        for model_name, model_data in self.results_data["models"].items():
            if "results" not in model_data:
                continue

            result = model_data["results"]

            # Flatten all metrics into a single row
            row = {
                "model_name": model_name,
                "model_path": model_data.get("model_path", ""),
                "model_config": json.dumps(model_data.get("model_config", {})),
                "model_hash": model_data.get("model_hash", ""),
                "last_evaluated": model_data.get("last_evaluated", ""),
                # Model metadata
                "version": result.get("model_metadata", {}).get("version", ""),
                "parameters": result.get("model_metadata", {}).get("parameters", 0),
                "model_size_mb": result.get("model_metadata", {}).get("model_size_mb", 0),
                "gflops": result.get("model_metadata", {}).get("gflops", None),
                "framework": result.get("model_metadata", {}).get("framework", "pytorch"),
                # Detection metrics
                "map_50": result.get("detection_metrics", {}).get("mAP@0.5", 0),
                "map_75": result.get("detection_metrics", {}).get("mAP@0.75", 0),
                "map_coco": result.get("detection_metrics", {}).get("mAP@[0.5:0.05:0.95]", 0),
                # Performance metrics
                "inference_time_ms": result.get("performance_metrics", {}).get("inference_time_ms", 0),
                "fps": result.get("performance_metrics", {}).get("fps", 0),
                "memory_usage_mb": result.get("performance_metrics", {}).get("memory_usage_mb", 0),
                # Statistics
                "total_detections": result.get("statistics", {}).get("total_detections", 0),
                "avg_detections_per_image": result.get("statistics", {}).get("avg_detections_per_image", 0),
                # Timestamp
                "evaluation_timestamp": result.get("evaluation_timestamp", datetime.now().isoformat()),
            }

            # Add class-specific metrics as separate columns
            class_metrics = result.get("detection_metrics", {})
            for key, value in class_metrics.items():
                if key.startswith("AP@0.5_"):
                    class_name = key.replace("AP@0.5_", "")
                    row[f"ap_50_{class_name}"] = value

            flattened_data.append(row)

        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_parquet(self.parquet_file, index=False)

    def _save_metadata(self):
        """Save metadata to database file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _calculate_model_hash(self, model_path: str, model_config: dict[str, Any]) -> str:
        """Calculate hash for model and configuration to detect changes."""
        # Get model file modification time and size
        model_stat = os.stat(model_path)
        model_info = f"{model_path}_{model_stat.st_mtime}_{model_stat.st_size}"

        # Include configuration
        config_str = json.dumps(model_config, sort_keys=True)

        # Calculate hash
        combined = f"{model_info}_{config_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def store_model_results(
        self, model_name: str, model_path: str, model_config: dict[str, Any], model_result: ModelResult
    ):
        """Store results for a single model with Pydantic validation."""
        try:
            # Validate the model result
            if isinstance(model_result, dict):
                model_result = ModelResult(**model_result)
            elif not isinstance(model_result, ModelResult):
                raise ValueError(f"model_result must be ModelResult or dict, got {type(model_result)}")

            model_hash = self._calculate_model_hash(model_path, model_config)

            self.results_data["models"][model_name] = {
                "model_path": model_path,
                "model_config": model_config,
                "model_hash": model_hash,
                "last_evaluated": datetime.now().isoformat(),
                "results": model_result.dict(),
            }

            self._save_results()
            print(f"   Stored validated results for {model_name}")

        except ValidationError as e:
            print(f"   ERROR: Validation error for {model_name}: {e}")
            raise

    def store_benchmark_run(self, benchmark_results: BenchmarkResults, output_directory: str):
        """Store complete benchmark run with Pydantic validation."""
        try:
            # Validate the benchmark results
            if isinstance(benchmark_results, dict):
                benchmark_results = BenchmarkResults(**benchmark_results)
            elif not isinstance(benchmark_results, BenchmarkResults):
                raise ValueError(f"benchmark_results must be BenchmarkResults or dict")

            run_data = {
                "timestamp": benchmark_results.timestamp.isoformat(),
                "output_directory": output_directory,
                "dataset_info": benchmark_results.dataset_info.dict(),
                "models_evaluated": [result.model_name for result in benchmark_results.model_results],
                "comparative_analysis": benchmark_results.comparative_analysis.dict(),
                "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }

            self.results_data["benchmark_runs"].append(run_data)

            # Update metadata
            self.metadata.update(
                {
                    "last_updated": datetime.now().isoformat(),
                    "total_models": len(self.results_data["models"]),
                    "total_benchmark_runs": len(self.results_data["benchmark_runs"]),
                    "dataset_info": benchmark_results.dataset_info.dict(),
                    "storage_format": self.storage_format,
                }
            )

            self._save_results()
            self._save_metadata()
            print(f"   Stored validated benchmark run: {run_data['run_id']}")

        except ValidationError as e:
            print(f"   ERROR: Validation error for benchmark run: {e}")
            raise

    def get_models_to_evaluate(self, all_models: dict[str, Any]) -> dict[str, Any]:
        """Determine which models need to be evaluated."""
        models_to_evaluate = {}

        for model_name, model_info in all_models.items():
            model_path = model_info.get("path")
            model_config = model_info.get("config", {})

            if not self.is_model_up_to_date(model_name, model_path, model_config):
                models_to_evaluate[model_name] = model_info
                print(f"   {model_name}: Needs evaluation (new or changed)")
            else:
                print(f"   {model_name}: Up to date, skipping")

        return models_to_evaluate

    def is_model_up_to_date(self, model_name: str, model_path: str, model_config: dict[str, Any]) -> bool:
        """Check if model results are up to date."""
        if model_name not in self.results_data["models"]:
            return False

        current_hash = self._calculate_model_hash(model_path, model_config)
        stored_hash = self.results_data["models"][model_name].get("model_hash")

        return current_hash == stored_hash

    def create_combined_benchmark_results(self, dataset_info: DatasetInfo) -> BenchmarkResults:
        """Create a validated BenchmarkResults object from stored data."""
        try:
            # Validate dataset info
            if isinstance(dataset_info, dict):
                dataset_info = DatasetInfo(**dataset_info)

            model_results = []
            for model_name, model_data in self.results_data["models"].items():
                if "results" in model_data:
                    result_data = model_data["results"]
                    result_data["model_name"] = model_name

                    # Create validated ModelResult
                    model_result = ModelResult(**result_data)
                    model_results.append(model_result)

            if not model_results:
                raise ValueError("No model results found in database")

            # Create comparative analysis
            comparative_analysis = self._create_comparative_analysis(model_results)

            return BenchmarkResults(
                timestamp=datetime.now(),
                dataset_info=dataset_info,
                model_results=model_results,
                comparative_analysis=comparative_analysis,
            )

        except ValidationError as e:
            print(f"   ERROR: Validation error creating benchmark results: {e}")
            raise

    def _create_comparative_analysis(self, model_results: list[ModelResult]) -> dict[str, Any]:
        """Create comparative analysis from validated model results."""
        if not model_results:
            return {}

        # Rankings by mAP@0.5
        rankings_map_50 = sorted(
            [(result.model_name, result.detection_metrics.map_50) for result in model_results],
            key=lambda x: x[1],
            reverse=True,
        )

        # Best performer
        best_performer = {"model": rankings_map_50[0][0], "map_0_5": rankings_map_50[0][1]}

        # Class-wise analysis
        class_wise_analysis = {}
        if model_results:
            # Get all classes from first model
            first_result = model_results[0]

            for class_name, score in first_result.detection_metrics.class_metrics.items():
                # Find best model for this class
                class_scores = [
                    (result.model_name, result.detection_metrics.class_metrics.get(class_name, 0))
                    for result in model_results
                ]
                class_scores.sort(key=lambda x: x[1], reverse=True)

                class_wise_analysis[class_name] = {"best_model": class_scores[0][0], "best_score": class_scores[0][1]}

        return {
            "rankings": {"by_map_0_5": rankings_map_50},
            "best_performer": best_performer,
            "class_wise_analysis": class_wise_analysis,
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all results to a pandas DataFrame for analysis."""
        if self.storage_format in ["parquet", "both"] and self.parquet_file.exists():
            return pd.read_parquet(self.parquet_file)
        else:
            # Create DataFrame from JSON data
            flattened_data = []
            for model_name, model_data in self.results_data["models"].items():
                if "results" in model_data:
                    result = model_data["results"]
                    row = {
                        "model_name": model_name,
                        "last_evaluated": model_data.get("last_evaluated", ""),
                        **self._flatten_dict(result, prefix=""),
                    }
                    flattened_data.append(row)

            return pd.DataFrame(flattened_data)

    def _flatten_dict(self, d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dictionary for DataFrame export."""
        flattened = {}
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_dict(value, new_key))
            else:
                flattened[new_key] = value
        return flattened

    def export_results_csv(self, output_path: str):
        """Export model results to CSV format."""
        df = self.export_to_dataframe()
        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f"   Exported {len(df)} model results to {output_path}")
        else:
            print("   No results to export")

    def get_database_summary(self) -> dict[str, Any]:
        """Get summary of database contents."""
        return {
            "database_path": str(self.database_path),
            "storage_format": self.storage_format,
            "database_version": self.metadata.get("database_version", "2.0"),
            "total_models": len(self.results_data["models"]),
            "total_benchmark_runs": len(self.results_data["benchmark_runs"]),
            "last_updated": self.metadata.get("last_updated"),
            "models": list(self.results_data["models"].keys()),
            "recent_runs": [run["run_id"] for run in self.results_data["benchmark_runs"][-5:]],
        }
