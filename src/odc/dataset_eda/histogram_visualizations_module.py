"""
Histogram Visualizations Module for SkyFusion Dataset.

This module provides a class-based interface for generating histogram
visualizations and size analysis for YOLO format datasets.

Author: Generated for SkyFusion Dataset Analysis
"""

import os
from typing import Any, Dict, List, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


class HistogramVisualizer:
    """Generates histogram visualizations and size analysis for YOLO format datasets."""

    def __init__(self, images_path, labels_path, output_dir: str):
        """
        Initialize the histogram visualizer.

        Args:
        ----
            images_path: Path to the images directory (or list of paths for combined)
            labels_path: Path to the labels directory (or list of paths for combined)
            output_dir: Directory to save histogram visualizations
        """
        # Handle both single paths and lists of paths
        self.images_paths = images_path if isinstance(images_path, list) else [images_path]
        self.labels_paths = labels_path if isinstance(labels_path, list) else [labels_path]
        self.output_dir = output_dir

        # Define class names
        self.class_names = {0: "aircraft", 1: "ship", 2: "vehicle"}

        # Get image dimensions
        self.image_size = self._get_image_size()

        # Storage for object data
        self.object_data = []

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def _get_image_size(self) -> tuple[int, int]:
        """Get the size of images in the dataset."""
        # Sample from first available images directory
        for images_path in self.images_paths:
            sample_files = [f for f in os.listdir(images_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            if sample_files:
                sample_img_path = os.path.join(images_path, sample_files[0])
                try:
                    with Image.open(sample_img_path) as img:
                        return img.size
                except Exception as e:
                    logger.warning(f"Error reading sample image: {e}")

        # Default to 640x640 if unable to determine
        return (640, 640)

    def _collect_object_data(self) -> None:
        """Collect object data from all label files."""
        # Process all label directories
        for labels_path in self.labels_paths:
            label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]

            logger.info(f"Collecting object data from {len(label_files)} label files in {labels_path}...")

            for filename in tqdm(label_files, desc=f"Processing {os.path.basename(labels_path)}"):
                file_path = os.path.join(labels_path, filename)

                try:
                    with open(file_path) as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])

                            # Calculate pixel dimensions
                            pixel_width = width * self.image_size[0]
                            pixel_height = height * self.image_size[1]
                            pixel_area = pixel_width * pixel_height

                            self.object_data.append(
                                {
                                    "file": filename,
                                    "class_id": class_id,
                                    "class_name": self.class_names[class_id],
                                    "x_center": x_center,
                                    "y_center": y_center,
                                    "norm_width": width,
                                    "norm_height": height,
                                    "norm_area": width * height,
                                    "pixel_width": pixel_width,
                                    "pixel_height": pixel_height,
                                    "pixel_area": pixel_area,
                                    "aspect_ratio": width / height if height > 0 else 0,
                                }
                            )

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")

    def generate_size_distribution_histogram(self) -> str:
        """Generate histogram of object size distribution."""
        if not self.object_data:
            return ""

        # Separate data by class
        class_areas = {class_id: [] for class_id in self.class_names.keys()}

        for obj in self.object_data:
            class_areas[obj["class_id"]].append(obj["pixel_area"])

        # Create histogram
        plt.figure(figsize=(14, 8))

        colors = ["red", "green", "blue"]
        bins = np.logspace(0, 5, 50)  # Log scale bins from 1 to 100,000

        for i, (class_id, areas) in enumerate(class_areas.items()):
            if areas:
                plt.hist(areas, bins=bins, alpha=0.6, label=self.class_names[class_id], color=colors[i], density=True)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Object Area (pixels²)")
        plt.ylabel("Density")
        plt.title("Object Size Distribution by Class (Log Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "size_distribution_histogram.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Size distribution histogram saved to {output_path}")
        return output_path

    def generate_aspect_ratio_histogram(self) -> str:
        """Generate histogram of aspect ratio distribution."""
        if not self.object_data:
            return ""

        # Separate data by class
        class_ratios = {class_id: [] for class_id in self.class_names.keys()}

        for obj in self.object_data:
            # Filter extreme outliers
            if 0.1 <= obj["aspect_ratio"] <= 10:
                class_ratios[obj["class_id"]].append(obj["aspect_ratio"])

        # Create histogram
        plt.figure(figsize=(12, 8))

        colors = ["red", "green", "blue"]
        bins = np.linspace(0, 5, 50)

        for i, (class_id, ratios) in enumerate(class_ratios.items()):
            if ratios:
                plt.hist(ratios, bins=bins, alpha=0.6, label=self.class_names[class_id], color=colors[i], density=True)

        plt.axvline(x=1, color="black", linestyle="--", alpha=0.7, label="Square (1:1)")
        plt.xlabel("Aspect Ratio (width/height)")
        plt.ylabel("Density")
        plt.title("Aspect Ratio Distribution by Class")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "aspect_ratio_histogram.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Aspect ratio histogram saved to {output_path}")
        return output_path

    def generate_size_category_analysis(self) -> str:
        """Generate analysis of objects by size categories."""
        if not self.object_data:
            return ""

        # Define size categories (in pixels²)
        size_categories = {
            "Very Small": (0, 25),
            "Small": (25, 100),
            "Medium": (100, 400),
            "Medium-Large": (400, 1600),
            "Large": (1600, 6400),
            "Very Large": (6400, float("inf")),
        }

        # Count objects in each category by class
        category_counts = {}
        for category in size_categories:
            category_counts[category] = {class_id: 0 for class_id in self.class_names.keys()}

        for obj in self.object_data:
            area = obj["pixel_area"]
            class_id = obj["class_id"]

            for category, (min_size, max_size) in size_categories.items():
                if min_size <= area < max_size:
                    category_counts[category][class_id] += 1
                    break

        # Create stacked bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Absolute counts
        categories = list(size_categories.keys())
        class_ids = sorted(self.class_names.keys())
        colors = ["red", "green", "blue"]

        bottom = np.zeros(len(categories))
        for i, class_id in enumerate(class_ids):
            counts = [category_counts[cat][class_id] for cat in categories]
            ax1.bar(categories, counts, bottom=bottom, label=self.class_names[class_id], color=colors[i], alpha=0.8)
            bottom += counts

        ax1.set_title("Object Count by Size Category")
        ax1.set_ylabel("Number of Objects")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)

        # Percentage distribution
        for i, class_id in enumerate(class_ids):
            total_class_objects = sum(category_counts[cat][class_id] for cat in categories)
            if total_class_objects > 0:
                percentages = [category_counts[cat][class_id] / total_class_objects * 100 for cat in categories]
                ax2.plot(
                    categories,
                    percentages,
                    marker="o",
                    label=self.class_names[class_id],
                    color=colors[i],
                    linewidth=2,
                    markersize=6,
                )

        ax2.set_title("Percentage Distribution by Size Category")
        ax2.set_ylabel("Percentage (%)")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "size_category_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Size category analysis saved to {output_path}")
        return output_path

    def find_and_visualize_extreme_objects(self, n_examples: int = 6) -> tuple[str, str]:
        """Find and visualize the largest and smallest objects."""
        if not self.object_data:
            return "", ""

        # Sort by area
        sorted_objects = sorted(self.object_data, key=lambda x: x["pixel_area"])

        # Get smallest and largest objects
        smallest_objects = sorted_objects[:n_examples]
        largest_objects = sorted_objects[-n_examples:]

        # Visualize smallest objects
        smallest_path = self._visualize_object_examples(
            smallest_objects, os.path.join(self.output_dir, "smallest_objects_grid.png"), "Smallest Objects in Dataset"
        )

        # Visualize largest objects
        largest_path = self._visualize_object_examples(
            largest_objects, os.path.join(self.output_dir, "largest_objects_grid.png"), "Largest Objects in Dataset"
        )

        return smallest_path, largest_path

    def _visualize_object_examples(self, objects: list[dict], output_path: str, title: str) -> str:
        """Visualize a grid of object examples."""
        if not objects:
            return ""

        grid_size = int(np.ceil(np.sqrt(len(objects))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, obj in enumerate(objects):
            if i >= len(axes):
                break

            # Find the correct image path for this file
            image_name = obj["file"].replace(".txt", ".jpg")
            image_path = None

            # Search across all image directories
            for images_dir in self.images_paths:
                potential_image_path = os.path.join(images_dir, image_name)
                if os.path.exists(potential_image_path):
                    image_path = potential_image_path
                    break

            if not image_path:
                axes[i].axis("off")
                continue

            try:
                img = Image.open(image_path)

                # Calculate crop region around the object
                x_center, y_center = obj["x_center"], obj["y_center"]
                norm_width, norm_height = obj["norm_width"], obj["norm_height"]

                # Add padding around the object
                padding = 0.1
                crop_width = max(norm_width + padding, 0.2)
                crop_height = max(norm_height + padding, 0.2)

                # Ensure crop doesn't exceed image boundaries
                x1 = max(0, x_center - crop_width / 2)
                y1 = max(0, y_center - crop_height / 2)
                x2 = min(1, x_center + crop_width / 2)
                y2 = min(1, y_center + crop_height / 2)

                # Convert to pixel coordinates
                img_width, img_height = img.size
                crop_box = (int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height))

                # Crop image
                cropped_img = img.crop(crop_box)

                # Draw bounding box on cropped image
                draw = ImageDraw.Draw(cropped_img)

                # Adjust coordinates for cropped image
                crop_x_center = (x_center - x1) / (x2 - x1)
                crop_y_center = (y_center - y1) / (y2 - y1)
                crop_norm_width = norm_width / (x2 - x1)
                crop_norm_height = norm_height / (y2 - y1)

                crop_width_px, crop_height_px = cropped_img.size
                box_x1 = int((crop_x_center - crop_norm_width / 2) * crop_width_px)
                box_y1 = int((crop_y_center - crop_norm_height / 2) * crop_height_px)
                box_x2 = int((crop_x_center + crop_norm_width / 2) * crop_width_px)
                box_y2 = int((crop_y_center + crop_norm_height / 2) * crop_height_px)

                color = ["red", "green", "blue"][obj["class_id"]]
                draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline=color, width=3)

                # Display image
                axes[i].imshow(np.array(cropped_img))
                axes[i].set_title(f"{obj['class_name']}\n{obj['pixel_area']:.0f} px²")
                axes[i].axis("off")

            except Exception as e:
                logger.warning(f"Error visualizing object in {image_path}: {e}")
                axes[i].axis("off")

        # Hide unused subplots
        for i in range(len(objects), len(axes)):
            axes[i].axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"{title} visualization saved to {output_path}")
        return output_path

    def generate_size_comparison_chart(self) -> str:
        """Generate a visual comparison of object sizes across classes."""
        if not self.object_data:
            return ""

        # Calculate statistics by class
        class_stats = {}
        for class_id in self.class_names.keys():
            class_objects = [obj for obj in self.object_data if obj["class_id"] == class_id]
            if class_objects:
                areas = [obj["pixel_area"] for obj in class_objects]
                class_stats[class_id] = {
                    "min": min(areas),
                    "max": max(areas),
                    "mean": np.mean(areas),
                    "median": np.median(areas),
                    "std": np.std(areas),
                    "count": len(areas),
                }

        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Box plot
        class_areas = []
        class_labels = []
        colors = ["red", "green", "blue"]

        for class_id in sorted(self.class_names.keys()):
            if class_id in class_stats:
                areas = [obj["pixel_area"] for obj in self.object_data if obj["class_id"] == class_id]
                class_areas.append(areas)
                class_labels.append(self.class_names[class_id])

        box_plot = ax1.boxplot(class_areas, labels=class_labels, patch_artist=True)
        for patch, color in zip(box_plot["boxes"], colors[: len(class_areas)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax1.set_yscale("log")
        ax1.set_title("Object Size Distribution by Class")
        ax1.set_ylabel("Area (pixels²)")
        ax1.grid(True, alpha=0.3)

        # Statistics bar chart
        stats_to_plot = ["min", "median", "mean", "max"]
        x_pos = np.arange(len(class_labels))
        width = 0.2

        for i, stat in enumerate(stats_to_plot):
            values = [
                class_stats[class_id][stat] for class_id in sorted(self.class_names.keys()) if class_id in class_stats
            ]
            ax2.bar(x_pos + i * width, values, width, label=stat.title(), alpha=0.8)

        ax2.set_yscale("log")
        ax2.set_title("Size Statistics by Class")
        ax2.set_ylabel("Area (pixels²)")
        ax2.set_xlabel("Class")
        ax2.set_xticks(x_pos + width * 1.5)
        ax2.set_xticklabels(class_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "size_comparison_chart.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Size comparison chart saved to {output_path}")
        return output_path

    def generate_all_histograms(self) -> dict[str, Any]:
        """
        Generate all histogram visualizations.

        Returns
        -------
            Dictionary containing paths to generated visualizations
        """
        logger.info("Starting histogram visualization generation...")

        # Collect object data
        self._collect_object_data()

        if not self.object_data:
            logger.warning("No object data found, skipping histogram generation")
            return {}

        results = {}

        # Generate histograms
        results["size_distribution"] = self.generate_size_distribution_histogram()
        results["aspect_ratio"] = self.generate_aspect_ratio_histogram()
        results["size_category"] = self.generate_size_category_analysis()
        results["size_comparison"] = self.generate_size_comparison_chart()

        # Generate extreme object visualizations
        smallest_path, largest_path = self.find_and_visualize_extreme_objects()
        if smallest_path:
            results["smallest_objects"] = smallest_path
        if largest_path:
            results["largest_objects"] = largest_path

        logger.success("All histogram visualizations generated successfully")
        return results
