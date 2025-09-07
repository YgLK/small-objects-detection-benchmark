"""
Dataset Exploratory Data Analysis (EDA) Module.

This module provides comprehensive tools for analyzing object detection datasets,
generating visualizations, and producing detailed reports.

Key Components:
- Dataset analysis and statistics
- Visualization generation (distributions, heatmaps, samples)
- Quality assessment and invalid box detection
- Report generation with comprehensive findings
"""

from .analyze_dataset_module import DatasetAnalyzer
from .count_invalid_boxes_module import InvalidBoxCounter
from .eda_pipeline import EDAPipeline
from .generate_bbox_heatmaps_module import HeatmapGenerator
from .histogram_visualizations_module import HistogramVisualizer
from .report_generator import ReportGenerator
from .visualize_dataset_module import DatasetVisualizer

__all__ = [
    "DatasetAnalyzer",
    "DatasetVisualizer",
    "InvalidBoxCounter",
    "HeatmapGenerator",
    "HistogramVisualizer",
    "ReportGenerator",
    "EDAPipeline",
]

__version__ = "1.0.0"
__author__ = "DPM3 SkyFusion Team"
