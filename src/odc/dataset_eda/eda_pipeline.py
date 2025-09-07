#!/usr/bin/env python3
"""
EDA Pipeline for SkyFusion Dataset.

This module provides a comprehensive pipeline for running exploratory data analysis
on object detection datasets, orchestrating all analysis components.

Usage:
    python eda_pipeline.py --images_path path/to/images --labels_path path/to/labels --split_name train --output_dir dataset_eda/reports/train

Author: Generated for SkyFusion Dataset Analysis
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .analyze_dataset_module import DatasetAnalyzer
from .count_invalid_boxes_module import InvalidBoxCounter
from .generate_bbox_heatmaps_module import HeatmapGenerator
from .histogram_visualizations_module import HistogramVisualizer
from .report_generator import ReportGenerator
from .visualize_dataset_module import DatasetVisualizer


class EDAPipeline:
    """Comprehensive EDA pipeline for object detection datasets."""

    def __init__(self, dataset_path: str, class_names: dict[int, str], output_base_dir: str):
        """
        Initialize the EDA pipeline.

        Args:
        ----
            dataset_path: Path to the dataset root directory
            class_names: Mapping of class IDs to class names
            output_base_dir: Base directory for output files
        """
        self.dataset_path = Path(dataset_path)
        self.class_names = class_names
        self.output_base_dir = Path(output_base_dir)

        # Ensure output directory exists
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def _get_split_paths(self, split: str) -> tuple[str, str]:
        """
        Get image and label paths for a specific split.

        Args:
        ----
            split: Dataset split name (train, valid, test)


        Returns:
        -------
            Tuple of (images_path, labels_path)
        """
        images_path = str(self.dataset_path / "images" / split)
        labels_path = str(self.dataset_path / "labels" / split)
        return images_path, labels_path

    def _get_combined_paths(self) -> tuple[list[str], list[str]]:
        """
        Get combined paths for all splits.

        Returns
        -------
            Tuple of (image_paths_list, label_paths_list)
        """
        splits = ["train", "valid", "test"]
        image_paths = []
        label_paths = []

        for split in splits:
            img_path, lbl_path = self._get_split_paths(split)

            if os.path.exists(img_path) and os.path.exists(lbl_path):
                image_paths.append(img_path)
                label_paths.append(lbl_path)
                logger.info(f"Added {split} split to combined analysis")
            else:
                logger.warning(f"Skipping {split} split - paths not found")

        return image_paths, label_paths

    def _setup_output_directories(self, split_name: str) -> dict[str, str]:
        """
        Setup output directory structure for a split.

        Args:
        ----
            split_name: Name of the dataset split


        Returns:
        -------
            Dictionary with paths to different output subdirectories
        """
        base_path = self.output_base_dir / split_name

        # Create directory structure
        dirs = {
            "base": str(base_path),
            "visualizations": str(base_path / "visualizations"),
            "statistics": str(base_path / "statistics"),
            "distributions": str(base_path / "visualizations" / "distributions"),
            "examples": str(base_path / "visualizations" / "examples"),
            "examples_size": str(base_path / "visualizations" / "examples" / "size"),
            "examples_count": str(base_path / "visualizations" / "examples" / "object_count"),
            "heatmaps": str(base_path / "visualizations" / "heatmaps"),
            "histograms": str(base_path / "visualizations" / "histograms"),
            "invalid_boxes": str(base_path / "visualizations" / "invalid_boxes"),
        }

        # Create all directories
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        logger.info(f"Created output directory structure at {base_path}")
        return dirs

    def _validate_inputs(self, images_path: str | list[str], labels_path: str | list[str]) -> bool:
        """
        Validate that input paths exist and contain expected files.

        Args:
        ----
            images_path: Path to images directory (or list of paths)
            labels_path: Path to labels directory (or list of paths)


        Returns:
        -------
            True if validation passes, False otherwise
        """
        # Handle both single paths and lists of paths
        image_paths = images_path if isinstance(images_path, list) else [images_path]
        label_paths = labels_path if isinstance(labels_path, list) else [labels_path]

        total_images = 0
        total_labels = 0

        for img_path, lbl_path in zip(image_paths, label_paths):
            if not os.path.exists(img_path):
                logger.error(f"Images path does not exist: {img_path}")
                return False

            if not os.path.exists(lbl_path):
                logger.error(f"Labels path does not exist: {lbl_path}")
                return False

            # Check if directories contain files
            image_files = [f for f in os.listdir(img_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            label_files = [f for f in os.listdir(lbl_path) if f.endswith(".txt")]

            if not image_files:
                logger.error(f"No image files found in {img_path}")
                return False

            if not label_files:
                logger.error(f"No label files found in {lbl_path}")
                return False

            total_images += len(image_files)
            total_labels += len(label_files)

        logger.info(f"Found {total_images} images and {total_labels} label files")
        return True

    def run_comprehensive_analysis(self, split: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Run comprehensive EDA analysis for a dataset split.

        Args:
        ----
            split: Dataset split name (train, valid, test, or combined)
            config: Configuration dictionary for analysis options


        Returns:
        -------
            Dictionary containing analysis results
        """
        if config is None:
            config = {
                "generate_distributions": True,
                "generate_heatmaps": True,
                "generate_histograms": True,
                "generate_samples": True,
                "detect_invalid_boxes": True,
            }

        logger.info(f"Starting comprehensive EDA analysis for {split} split")

        # Get paths for the split
        if split == "combined":
            images_path, labels_path = self._get_combined_paths()
        else:
            images_path, labels_path = self._get_split_paths(split)

        # Validate inputs
        if not self._validate_inputs(images_path, labels_path):
            logger.error(f"Input validation failed for {split} split")
            return {}

        # Setup output directories
        dirs = self._setup_output_directories(split)

        results = {
            "split": split,
            "paths": {"images": images_path, "labels": labels_path},
            "output_dirs": dirs,
            "visualizations": [],
            "reports": [],
        }

        try:
            # Step 1: Basic dataset analysis
            if config.get("generate_distributions", True):
                logger.info("Step 1: Running basic dataset analysis...")
                analyzer = DatasetAnalyzer(images_path, labels_path, dirs["statistics"])
                analysis_results = analyzer.analyze()
                results["analysis"] = analysis_results

            # Step 2: Generate visualizations
            if config.get("generate_samples", True):
                logger.info("Step 2: Generating dataset visualizations...")
                visualizer = DatasetVisualizer(images_path, labels_path, dirs)
                visualizer.generate_all_visualizations()
                results["visualizations"].extend(["distributions", "examples"])

            # Step 3: Count invalid boxes
            if config.get("detect_invalid_boxes", True):
                logger.info("Step 3: Analyzing invalid bounding boxes...")
                invalid_counter = InvalidBoxCounter(images_path, labels_path, dirs["invalid_boxes"])
                invalid_results = invalid_counter.analyze()
                results["invalid_boxes"] = invalid_results

            # Step 4: Generate heatmaps
            if config.get("generate_heatmaps", True):
                logger.info("Step 4: Generating spatial heatmaps...")
                heatmap_generator = HeatmapGenerator(labels_path, dirs["heatmaps"])
                heatmap_generator.generate_all_heatmaps()
                results["visualizations"].append("heatmaps")

            # Step 5: Generate histograms
            if config.get("generate_histograms", True):
                logger.info("Step 5: Generating histogram visualizations...")
                histogram_visualizer = HistogramVisualizer(images_path, labels_path, dirs["histograms"])
                histogram_visualizer.generate_all_histograms()
                results["visualizations"].append("histograms")

            # Step 6: Generate markdown report
            logger.info("Step 6: Generating markdown report...")
            report_generator = ReportGenerator(
                split_name=split,
                analysis_results=results.get("analysis", {}),
                invalid_results=results.get("invalid_boxes", {}),
                output_dirs=dirs,
            )
            report_path = report_generator.generate_report()
            results["reports"].append(report_path)

            logger.success(f"EDA analysis completed for {split} split")

        except Exception as e:
            logger.error(f"Error during EDA analysis for {split}: {e}")
            results["error"] = str(e)

        return results

    def generate_combined_report(self, split_results: dict[str, dict[str, Any]]) -> str:
        """
        Generate a combined report from multiple split results.

        Args:
        ----
            split_results: Dictionary mapping split names to their results


        Returns:
        -------
            Path to the generated combined report
        """
        logger.info("Generating combined EDA report...")

        # Create combined output directory
        combined_dir = self.output_base_dir / "combined_report"
        combined_dir.mkdir(exist_ok=True)

        # Generate combined report
        report_path = combined_dir / "combined_eda_report.md"

        with open(report_path, "w") as f:
            f.write("# Combined Dataset EDA Report\n\n")
            f.write("This report combines EDA results from multiple dataset splits.\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Splits analyzed**: {len(split_results)}\n")
            analysis_date = "N/A"
            if hasattr(logger, "_core"):
                analysis_date = "Current session"
            f.write(f"- **Analysis date**: {analysis_date}\n\n")

            for split_name, results in split_results.items():
                if results and "error" not in results:
                    f.write(f"### {split_name.title()} Split\n\n")
                    f.write(f"- **Visualizations**: {len(results.get('visualizations', []))}\n")
                    f.write(f"- **Reports**: {len(results.get('reports', []))}\n")

                    if "analysis" in results:
                        analysis = results["analysis"]
                        f.write(f"- **Total images**: {analysis.get('total_images', 'N/A')}\n")
                        f.write(f"- **Total objects**: {analysis.get('total_objects', 'N/A')}\n")

                    f.write("\n")

        logger.success(f"Combined report generated: {report_path}")
        return str(report_path)
