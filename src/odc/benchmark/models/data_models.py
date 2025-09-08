"""Pydantic data models for benchmark system validation and type safety.

This module defines the data structures used throughout the benchmark system
with proper validation, type hints, and serialization support.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version or architecture")
    parameters: int = Field(..., ge=0, description="Number of parameters")
    model_size_mb: float = Field(..., ge=0, description="Model size in MB")
    gflops: float | None = Field(None, ge=0, description="GFLOPs for inference")
    framework: str = Field(default="pytorch", description="ML framework used")
    created_at: datetime | None = Field(None, description="Model creation timestamp")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DetectionMetrics(BaseModel):
    """Detection performance metrics."""

    # Main mAP metrics
    map_50: float = Field(..., ge=0, le=1, alias="mAP@0.5", description="mAP at IoU=0.5")
    map_75: float = Field(..., ge=0, le=1, alias="mAP@0.75", description="mAP at IoU=0.75")
    map_coco: float = Field(..., ge=0, le=1, alias="mAP@[0.5:0.05:0.95]", description="COCO-style mAP")

    # Per-class metrics (dynamic based on dataset)
    class_metrics: dict[str, float] = Field(default_factory=dict, description="Per-class AP@0.5 scores")

    # Additional metrics
    precision: float | None = Field(None, ge=0, le=1, description="Overall precision")
    recall: float | None = Field(None, ge=0, le=1, description="Overall recall")
    f1_score: float | None = Field(None, ge=0, le=1, description="F1 score")

    @validator("class_metrics")
    def validate_class_metrics(cls, v):
        """Validate that all class metrics are between 0 and 1."""
        for class_name, score in v.items():
            if not (0 <= score <= 1):
                raise ValueError(f"Class metric for {class_name} must be between 0 and 1, got {score}")
        return v

    class Config:
        allow_population_by_field_name = True


class PerformanceMetrics(BaseModel):
    """Performance and efficiency metrics."""

    # Timing metrics
    inference_time_ms: float = Field(..., ge=0, description="Average inference time in milliseconds")
    fps: float = Field(..., ge=0, description="Frames per second")
    std_inference_time_ms: float | None = Field(None, ge=0, description="Standard deviation of inference time")

    # Resource usage
    memory_usage_mb: float = Field(..., ge=0, description="Memory usage in MB")
    gpu_memory_mb: float | None = Field(None, ge=0, description="GPU memory usage in MB")
    cpu_usage_percent: float | None = Field(None, ge=0, le=100, description="CPU usage percentage")

    # Model complexity
    parameters: int = Field(..., ge=0, description="Number of model parameters")
    model_size_mb: float = Field(..., ge=0, description="Model file size in MB")
    gflops: float | None = Field(None, ge=0, description="GFLOPs for inference")

    @validator("fps")
    def validate_fps(cls, v, values):
        """Validate FPS is consistent with inference time."""
        if "inference_time_ms" in values and values["inference_time_ms"] > 0:
            expected_fps = 1000 / values["inference_time_ms"]
            if abs(v - expected_fps) > 0.1:  # Allow small tolerance
                raise ValueError(f"FPS {v} inconsistent with inference time {values['inference_time_ms']}ms")
        return v


class DetectionStatistics(BaseModel):
    """Statistics about detections made by the model."""

    total_detections: int = Field(..., ge=0, description="Total number of detections")
    avg_detections_per_image: float = Field(..., ge=0, description="Average detections per image")
    confidence_distribution: dict[str, int] | None = Field(None, description="Distribution of confidence scores")
    size_distribution: dict[str, int] | None = Field(None, description="Distribution of detection sizes")

    # Per-class statistics
    class_detection_counts: dict[str, int] = Field(default_factory=dict, description="Detections per class")

    @validator("class_detection_counts")
    def validate_class_counts(cls, v):
        """Validate that all class counts are non-negative."""
        for class_name, count in v.items():
            if count < 0:
                raise ValueError(f"Detection count for {class_name} must be non-negative, got {count}")
        return v


class ModelResult(BaseModel):
    """Complete results for a single model evaluation."""

    model_name: str = Field(..., description="Name of the evaluated model")
    model_metadata: ModelMetadata = Field(..., description="Model metadata")
    detection_metrics: DetectionMetrics = Field(..., description="Detection performance metrics")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance and efficiency metrics")
    statistics: DetectionStatistics = Field(..., description="Detection statistics")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now, description="When evaluation was performed")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DatasetInfo(BaseModel):
    """Information about the dataset used for evaluation."""

    name: str = Field(..., description="Dataset name")
    split: str = Field(..., description="Dataset split (train/val/test)")
    num_images: int = Field(..., ge=0, description="Number of images")
    num_annotations: int = Field(..., ge=0, description="Total number of annotations")
    num_classes: int = Field(..., ge=0, description="Number of classes")
    class_names: list[str] = Field(..., description="List of class names")
    class_counts: dict[str, int] = Field(..., description="Number of annotations per class")

    @validator("class_names")
    def validate_class_names(cls, v, values):
        """Validate class names consistency."""
        if "num_classes" in values and len(v) != values["num_classes"]:
            raise ValueError(f"Number of class names {len(v)} doesn't match num_classes {values['num_classes']}")
        return v

    @validator("class_counts")
    def validate_class_counts(cls, v, values):
        """Validate class counts consistency."""
        if "class_names" in values:
            missing_classes = set(values["class_names"]) - set(v.keys())
            if missing_classes:
                raise ValueError(f"Missing class counts for: {missing_classes}")

        total_from_counts = sum(v.values())
        if "num_annotations" in values and total_from_counts != values["num_annotations"]:
            raise ValueError(
                f"Sum of class counts {total_from_counts} doesn't match num_annotations {values['num_annotations']}"
            )

        return v


class ComparativeAnalysis(BaseModel):
    """Comparative analysis results across models."""

    rankings: dict[str, list[tuple]] = Field(..., description="Model rankings by different metrics")
    best_performer: dict[str, str | float] = Field(..., description="Best performing model info")
    class_wise_analysis: dict[str, dict[str, str | float]] = Field(..., description="Best model per class")
    performance_summary: dict[str, Any] | None = Field(None, description="Summary statistics")

    @validator("rankings")
    def validate_rankings(cls, v):
        """Validate rankings structure."""
        required_keys = ["by_map_0_5"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required ranking key: {key}")
        return v


class BenchmarkResults(BaseModel):
    """Complete benchmark results containing all model evaluations and analysis."""

    timestamp: datetime = Field(default_factory=datetime.now, description="Benchmark execution timestamp")
    dataset_info: DatasetInfo = Field(..., description="Dataset information")
    model_results: list[ModelResult] = Field(..., description="Results for each evaluated model")
    comparative_analysis: ComparativeAnalysis = Field(..., description="Comparative analysis across models")
    config: dict[str, Any] | None = Field(None, description="Benchmark configuration used")

    @validator("model_results")
    def validate_model_results(cls, v):
        """Validate that the list of model results is not empty."""
        if not v:
            raise ValueError("At least one model result is required")
        return v

    @root_validator
    def validate_consistency(cls, values):
        """Validate consistency across all results."""
        if "model_results" in values and "dataset_info" in values:
            dataset_classes = set(values["dataset_info"].class_names)

            for model_result in values["model_results"]:
                result_classes = set(model_result.detection_metrics.class_metrics.keys())
                if not result_classes.issubset(dataset_classes):
                    extra_classes = result_classes - dataset_classes
                    raise ValueError(
                        f"Model {model_result.model_name} has metrics for unknown classes: {extra_classes}"
                    )

        return values

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        return json.loads(self.json())

    def save_to_file(self, filepath: str | Path) -> None:
        """Save results to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(self.json(indent=2))

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> "BenchmarkResults":
        """Load results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""

    # Dataset configuration
    dataset_path: str = Field(..., description="Path to dataset")
    dataset_split: str = Field(default="test", description="Dataset split to use")
    max_samples: int | None = Field(None, ge=1, description="Maximum number of samples to evaluate")

    # Model configuration
    model_files: list[str] = Field(..., description="List of model files to evaluate")
    confidence_threshold: float = Field(default=0.25, ge=0, le=1, description="Confidence threshold for detections")
    iou_threshold: float = Field(default=0.45, ge=0, le=1, description="IoU threshold for NMS")

    # Evaluation configuration
    evaluation_mode: str = Field(default="comprehensive", description="Evaluation mode")
    calculate_size_analysis: bool = Field(default=True, description="Whether to calculate size analysis")
    calculate_density_analysis: bool = Field(default=True, description="Whether to calculate density analysis")

    # Output configuration
    output_directory: str = Field(default="materials/benchmarks", description="Output directory")
    generate_plots: bool = Field(default=True, description="Whether to generate plots")
    generate_reports: bool = Field(default=True, description="Whether to generate reports")

    # Database configuration
    use_database: bool = Field(default=True, description="Whether to use persistent database")
    incremental_updates: bool = Field(default=True, description="Whether to use incremental updates")

    @validator("model_files")
    def validate_model_files(cls, v):
        """Validate that model files exist."""
        for model_file in v:
            model_path = Path(f"models/{model_file}")
            if not model_path.exists():
                raise ValueError(f"Model file not found: {model_path}")
        return v

    class Config:
        extra = "allow"  # Allow additional configuration fields


# Type aliases for convenience
ModelResultDict = dict[str, Any]  # For backward compatibility
DatasetInfoDict = dict[str, Any]  # For backward compatibility
