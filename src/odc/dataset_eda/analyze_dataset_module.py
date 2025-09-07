"""
Dataset Analysis Module for SkyFusion Dataset.

This module provides a class-based interface for analyzing YOLO format datasets.
It extracts statistics about classes, bounding boxes, and object distributions.

Author: Generated for SkyFusion Dataset Analysis
"""

from collections import Counter
import csv
import json
import os
from typing import Any, Dict, List, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


class DatasetAnalyzer:
    """Analyzes YOLO format datasets and generates comprehensive statistics."""

    def __init__(self, images_path, labels_path, output_dir: str):
        """
        Initialize the dataset analyzer.

        Args:
        ----
            images_path: Path to the images directory (or list of paths for combined)
            labels_path: Path to the labels directory (or list of paths for combined)
            output_dir: Directory to save analysis results
        """
        # Handle both single paths and lists of paths
        self.images_paths = images_path if isinstance(images_path, list) else [images_path]
        self.labels_paths = labels_path if isinstance(labels_path, list) else [labels_path]
        self.output_dir = output_dir

        # Define class names from dataset.yaml
        self.class_names = {0: "aircraft", 1: "ship", 2: "vehicle"}

        # Initialize data structures
        self.class_counts = Counter()
        self.bbox_count_per_image = []
        self.bbox_sizes = []
        self.bbox_sizes_by_class = {0: [], 1: [], 2: []}
        self.aspect_ratios = []
        self.aspect_ratios_by_class = {0: [], 1: [], 2: []}

        # Results storage
        self.results = {}

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def _process_label_files(self) -> None:
        """Process all label files and extract statistics."""
        total_files = 0
        empty_images = 0

        # Process all label directories
        for labels_path in self.labels_paths:
            label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]

            logger.info(f"Processing {len(label_files)} label files from {labels_path}...")

            for filename in tqdm(label_files, desc=f"Processing {os.path.basename(labels_path)}"):
                total_files += 1
                file_path = os.path.join(labels_path, filename)

                bboxes_in_image = 0
                try:
                    with open(file_path) as f:
                        lines = f.readlines()
                        bboxes_in_image = len(lines)

                        if bboxes_in_image == 0:
                            empty_images += 1

                        self.bbox_count_per_image.append(bboxes_in_image)

                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                self.class_counts[class_id] += 1

                                # Extract normalized bbox dimensions
                                x_center, y_center, width, height = map(float, parts[1:5])
                                area = width * height
                                self.bbox_sizes.append(area)
                                self.bbox_sizes_by_class[class_id].append(area)

                                # Calculate aspect ratio (width/height)
                                if height > 0:
                                    aspect = width / height
                                    self.aspect_ratios.append(aspect)
                                    self.aspect_ratios_by_class[class_id].append(aspect)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")

        self.results["total_files"] = total_files
        self.results["empty_images"] = empty_images

    def _get_image_sizes(self) -> list[tuple[int, int]]:
        """Get sample image sizes to determine dataset image dimensions."""
        img_sizes = []

        # Sample from all image directories
        for images_path in self.images_paths:
            sample_files = [f for f in os.listdir(images_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))][
                :2
            ]  # 2 samples per directory

            for sample_file in sample_files:
                try:
                    with Image.open(os.path.join(images_path, sample_file)) as img:
                        img_sizes.append(img.size)
                except Exception as e:
                    logger.warning(f"Error opening image {sample_file}: {e}")

        return img_sizes

    def _generate_text_report(self) -> str:
        """Generate a text analysis report."""
        img_sizes = self._get_image_sizes()

        report_file = os.path.join(self.output_dir, "dataset_analysis_report.txt")

        with open(report_file, "w") as f:
            f.write("===== SkyFusion YOLO Dataset Analysis =====\n\n")

            f.write("Dataset Overview:\n")
            f.write(f"  Total images: {self.results['total_files']}\n")
            f.write(
                f"  Images with no objects: {self.results['empty_images']} "
                f"({self.results['empty_images'] / self.results['total_files'] * 100:.2f}%)\n"
            )
            f.write(f"  Total objects: {sum(self.class_counts.values())}\n")
            f.write(f"  Sample image sizes: {img_sizes}\n\n")

            f.write("Class distribution:\n")
            for class_id, count in sorted(self.class_counts.items()):
                percentage = count / sum(self.class_counts.values()) * 100
                f.write(f"  {self.class_names[class_id]} (Class {class_id}): {count} ({percentage:.2f}%)\n")

            f.write("\nBounding boxes per image:\n")
            f.write(f"  Min: {min(self.bbox_count_per_image) if self.bbox_count_per_image else 0}\n")
            f.write(f"  Max: {max(self.bbox_count_per_image) if self.bbox_count_per_image else 0}\n")
            f.write(f"  Avg: {np.mean(self.bbox_count_per_image) if self.bbox_count_per_image else 0:.2f}\n")
            f.write(f"  Median: {np.median(self.bbox_count_per_image) if self.bbox_count_per_image else 0:.1f}\n")

            f.write("\nBounding box sizes (normalized area):\n")
            f.write(f"  Min: {min(self.bbox_sizes) if self.bbox_sizes else 0:.6f}\n")
            f.write(f"  Max: {max(self.bbox_sizes) if self.bbox_sizes else 0:.6f}\n")
            f.write(f"  Avg: {np.mean(self.bbox_sizes) if self.bbox_sizes else 0:.6f}\n")

            f.write("\nBounding box aspect ratios (width/height):\n")
            f.write(f"  Min: {min(self.aspect_ratios) if self.aspect_ratios else 0:.2f}\n")
            f.write(f"  Max: {max(self.aspect_ratios) if self.aspect_ratios else 0:.2f}\n")
            f.write(f"  Avg: {np.mean(self.aspect_ratios) if self.aspect_ratios else 0:.2f}\n")

            # Class-specific statistics
            for class_id in sorted(self.bbox_sizes_by_class.keys()):
                sizes = self.bbox_sizes_by_class[class_id]
                aspects = self.aspect_ratios_by_class[class_id]

                if sizes:
                    f.write(f"\n{self.class_names[class_id]} (Class {class_id}) statistics:\n")
                    f.write(f"  Count: {len(sizes)}\n")
                    f.write(f"  Bbox size - Min: {min(sizes):.6f}, Max: {max(sizes):.6f}, Avg: {np.mean(sizes):.6f}\n")
                    if aspects:
                        f.write(
                            f"  Aspect ratio - Min: {min(aspects):.2f}, Max: {max(aspects):.2f}, Avg: {np.mean(aspects):.2f}\n"
                        )

            # Objects per image distribution
            counts = pd.Series(self.bbox_count_per_image).value_counts().sort_index()
            f.write("\nObjects per image distribution:\n")
            for count, frequency in counts.items():
                f.write(
                    f"  {count} objects: {frequency} images ({frequency / self.results['total_files'] * 100:.2f}%)\n"
                )

        logger.info(f"Text analysis report generated: {report_file}")
        return report_file

    def _generate_json_metadata(self) -> str:
        """Generate JSON dataset metadata for programmatic use."""
        img_sizes = self._get_image_sizes()

        metadata = {
            "dataset_info": {
                "name": "SkyFusion YOLO",
                "total_images": self.results["total_files"],
                "empty_images": self.results["empty_images"],
                "total_objects": sum(self.class_counts.values()),
                "image_size": img_sizes[0] if img_sizes else [640, 640],
                "classes": {
                    class_id: {
                        "name": self.class_names[class_id],
                        "count": count,
                        "percentage": count / sum(self.class_counts.values()) * 100,
                    }
                    for class_id, count in sorted(self.class_counts.items())
                },
            },
            "bounding_box_stats": {
                "overall": {
                    "min_size": min(self.bbox_sizes) if self.bbox_sizes else 0,
                    "max_size": max(self.bbox_sizes) if self.bbox_sizes else 0,
                    "avg_size": np.mean(self.bbox_sizes) if self.bbox_sizes else 0,
                    "min_aspect": min(self.aspect_ratios) if self.aspect_ratios else 0,
                    "max_aspect": max(self.aspect_ratios) if self.aspect_ratios else 0,
                    "avg_aspect": np.mean(self.aspect_ratios) if self.aspect_ratios else 0,
                }
            },
            "objects_per_image": {
                "min": min(self.bbox_count_per_image) if self.bbox_count_per_image else 0,
                "max": max(self.bbox_count_per_image) if self.bbox_count_per_image else 0,
                "avg": np.mean(self.bbox_count_per_image) if self.bbox_count_per_image else 0,
                "median": float(np.median(self.bbox_count_per_image)) if self.bbox_count_per_image else 0,
            },
        }

        # Add class-specific statistics to metadata
        for class_id in sorted(self.bbox_sizes_by_class.keys()):
            sizes = self.bbox_sizes_by_class[class_id]
            aspects = self.aspect_ratios_by_class[class_id]

            if sizes:
                metadata["bounding_box_stats"][self.class_names[class_id]] = {
                    "min_size": min(sizes),
                    "max_size": max(sizes),
                    "avg_size": np.mean(sizes),
                    "min_aspect": min(aspects) if aspects else 0,
                    "max_aspect": max(aspects) if aspects else 0,
                    "avg_aspect": np.mean(aspects) if aspects else 0,
                }

        # Save metadata as JSON
        metadata_file = os.path.join(self.output_dir, "dataset_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset metadata JSON generated: {metadata_file}")
        return metadata_file

    def _generate_csv_statistics(self) -> str:
        """Generate dataset statistics CSV."""
        stats_data = []

        # Overall statistics
        stats_data.append(
            {
                "Type": "Overall",
                "Class": "All",
                "Count": sum(self.class_counts.values()),
                "Percentage": 100.0,
                "Min_Size": min(self.bbox_sizes) if self.bbox_sizes else 0,
                "Max_Size": max(self.bbox_sizes) if self.bbox_sizes else 0,
                "Avg_Size": np.mean(self.bbox_sizes) if self.bbox_sizes else 0,
                "Min_Aspect": min(self.aspect_ratios) if self.aspect_ratios else 0,
                "Max_Aspect": max(self.aspect_ratios) if self.aspect_ratios else 0,
                "Avg_Aspect": np.mean(self.aspect_ratios) if self.aspect_ratios else 0,
            }
        )

        # Class-specific statistics
        for class_id in sorted(self.bbox_sizes_by_class.keys()):
            sizes = self.bbox_sizes_by_class[class_id]
            aspects = self.aspect_ratios_by_class[class_id]
            count = self.class_counts[class_id]

            if sizes:
                stats_data.append(
                    {
                        "Type": "Class",
                        "Class": self.class_names[class_id],
                        "Count": count,
                        "Percentage": count / sum(self.class_counts.values()) * 100,
                        "Min_Size": min(sizes),
                        "Max_Size": max(sizes),
                        "Avg_Size": np.mean(sizes),
                        "Min_Aspect": min(aspects) if aspects else 0,
                        "Max_Aspect": max(aspects) if aspects else 0,
                        "Avg_Aspect": np.mean(aspects) if aspects else 0,
                    }
                )

        # Save as CSV
        csv_file = os.path.join(self.output_dir, "dataset_statistics.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats_data[0].keys())
            writer.writeheader()
            writer.writerows(stats_data)

        logger.info(f"Dataset statistics CSV generated: {csv_file}")
        return csv_file

    def analyze(self) -> dict[str, Any]:
        """
        Run the complete dataset analysis.

        Returns
        -------
            Dictionary containing analysis results
        """
        logger.info("Starting dataset analysis...")

        # Process label files
        self._process_label_files()

        # Generate reports
        text_report = self._generate_text_report()
        json_metadata = self._generate_json_metadata()
        csv_statistics = self._generate_csv_statistics()

        # Compile results
        analysis_results = {
            "total_files": self.results["total_files"],
            "empty_images": self.results["empty_images"],
            "total_objects": sum(self.class_counts.values()),
            "class_counts": dict(self.class_counts),
            "class_names": self.class_names,
            "bbox_count_per_image": self.bbox_count_per_image,
            "bbox_sizes": self.bbox_sizes,
            "bbox_sizes_by_class": self.bbox_sizes_by_class,
            "aspect_ratios": self.aspect_ratios,
            "aspect_ratios_by_class": self.aspect_ratios_by_class,
            "image_sizes": self._get_image_sizes(),
            "output_files": {
                "text_report": text_report,
                "json_metadata": json_metadata,
                "csv_statistics": csv_statistics,
            },
        }

        logger.success("Dataset analysis completed successfully")
        return analysis_results
