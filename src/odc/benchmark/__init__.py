"""Object Detection Benchmark System.

A comprehensive, extensible benchmark system for object detection models
with professional LaTeX-compatible reports for thesis integration.
"""

__version__ = "0.1.0"
__author__ = "YgLK"

# Core interfaces
from .models.base import ObjectDetectionModel, Detection, ModelMetadata
from .datasets.base import ObjectDetectionDataset, GroundTruthAnnotation, DatasetSample
from .reporters.base_reporter import BaseReporter, BenchmarkResults

# Implementations
from .models.ultralytics_adapter import UltralyticsModel
from .models.fasterrcnn_adapter import FasterRCNNModel
from .models.rfdetr_adapter import RFDETRModel
from .datasets.skyfusion_loader import SkyFusionDataset
from .metrics.detection_metrics import DetectionMetrics
from .metrics.performance_metrics import PerformanceMetrics, PerformanceEvaluator

# Reporting and Visualization
from .reporters.latex_reporter import LaTeXReporter
from .visualization.plot_generators import PlotGenerator
from .visualization.detection_visualizer import DetectionVisualizer

# Main pipeline
from .pipeline import BenchmarkPipeline

__all__ = [
    # Core interfaces
    "ObjectDetectionModel",
    "Detection",
    "ModelMetadata",
    "ObjectDetectionDataset",
    "GroundTruthAnnotation",
    "DatasetSample",
    "BaseReporter",
    "BenchmarkResults",
    # Implementations
    "UltralyticsModel",
    "FasterRCNNModel",
    "RFDETRModel",
    "SkyFusionDataset",
    "DetectionMetrics",
    "PerformanceMetrics",
    "PerformanceEvaluator",
    # Reporting and Visualization
    "LaTeXReporter",
    "PlotGenerator",
    "DetectionVisualizer",
    # Pipeline
    "BenchmarkPipeline",
]
