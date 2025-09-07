"""
Dataset Visualization Module for SkyFusion Dataset.

This module provides a class-based interface for generating visualizations
of YOLO format datasets including distributions, examples, and sample images.

Author: Generated for SkyFusion Dataset Analysis
"""

from collections import Counter
import os
from typing import Any, Dict, List

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import seaborn as sns
from tqdm import tqdm


class DatasetVisualizer:
    """Generates comprehensive visualizations for YOLO format datasets."""

    def __init__(self, images_path, labels_path, output_dirs: dict[str, str]):
        """
        Initialize the dataset visualizer.

        Args:
        ----
            images_path: Path to the images directory (or list of paths for combined)
            labels_path: Path to the labels directory (or list of paths for combined)
            output_dirs: Dictionary with output directory paths
        """
        # Handle both single paths and lists of paths
        self.images_paths = images_path if isinstance(images_path, list) else [images_path]
        self.labels_paths = labels_path if isinstance(labels_path, list) else [labels_path]
        self.output_dirs = output_dirs

        # Define class names and colors for visualization
        self.class_names = {0: "aircraft", 1: "ship", 2: "vehicle"}
        self.class_colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

        # Initialize data structures
        self.class_counts = Counter()
        self.bbox_count_per_image = []
        self.bbox_sizes_by_class = {0: [], 1: [], 2: []}
        self.aspect_ratios_by_class = {0: [], 1: [], 2: []}

        # Ensure output directories exist
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def _collect_statistics(self) -> None:
        """Collect statistics from label files for visualization."""
        # Process all label directories
        for labels_path in self.labels_paths:
            label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]

            logger.info(f"Collecting statistics from {len(label_files)} label files in {labels_path}...")

            for filename in tqdm(label_files, desc=f"Processing {os.path.basename(labels_path)}"):
                file_path = os.path.join(labels_path, filename)

                bboxes_in_image = 0
                try:
                    with open(file_path) as f:
                        lines = f.readlines()
                        bboxes_in_image = len(lines)
                        self.bbox_count_per_image.append(bboxes_in_image)

                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                self.class_counts[class_id] += 1

                                # Extract normalized bbox dimensions
                                x_center, y_center, width, height = map(float, parts[1:5])

                                # Calculate area and aspect ratio
                                area = width * height
                                aspect = width / height if height > 0 else 0

                                self.bbox_sizes_by_class[class_id].append(area)
                                self.aspect_ratios_by_class[class_id].append(aspect)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")

    def generate_class_distribution(self) -> str:
        """Generate class distribution bar chart."""
        plt.figure(figsize=(10, 6))
        classes = [self.class_names[i] for i in sorted(self.class_counts.keys())]
        counts = [self.class_counts[i] for i in sorted(self.class_counts.keys())]
        bars = plt.bar(classes, counts, color=["red", "green", "blue"])

        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{height:,}", ha="center", va="bottom")

        plt.title("Class Distribution in SkyFusion Dataset")
        plt.ylabel("Number of Objects")
        plt.ylim(0, max(counts) * 1.1)  # Add 10% padding
        plt.tight_layout()

        output_path = os.path.join(self.output_dirs["distributions"], "class_distribution.png")
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Class distribution chart saved to {output_path}")
        return output_path

    def generate_objects_per_image_distribution(self) -> str:
        """Generate objects per image histogram."""
        plt.figure(figsize=(12, 6))
        data = np.array(self.bbox_count_per_image)
        data_filtered = data[data <= 100]  # Filter extremely high values for better visualization
        plt.hist(data_filtered, bins=30, alpha=0.7, color="blue")
        plt.title("Objects per Image Distribution (images with ≤ 100 objects)")
        plt.xlabel("Number of Objects")
        plt.ylabel("Number of Images")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dirs["distributions"], "objects_per_image.png")
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Objects per image distribution saved to {output_path}")
        return output_path

    def generate_bbox_size_distribution(self) -> str:
        """Generate bounding box size distribution by class."""
        plt.figure(figsize=(12, 6))
        for class_id, sizes in self.bbox_sizes_by_class.items():
            if sizes:
                sns.kdeplot(np.array(sizes), label=self.class_names[class_id])
        plt.title("Bounding Box Size Distribution by Class")
        plt.xlabel("Normalized Box Area")
        plt.xscale("log")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dirs["distributions"], "bbox_size_distribution.png")
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Bounding box size distribution saved to {output_path}")
        return output_path

    def generate_aspect_ratio_distribution(self) -> str:
        """Generate aspect ratio distribution by class."""
        plt.figure(figsize=(12, 6))
        for class_id, ratios in self.aspect_ratios_by_class.items():
            filtered_ratios = [r for r in ratios if 0.1 <= r <= 10]  # Filter extreme outliers
            if filtered_ratios:
                sns.kdeplot(np.array(filtered_ratios), label=self.class_names[class_id])
        plt.title("Bounding Box Aspect Ratio Distribution by Class")
        plt.xlabel("Aspect Ratio (width/height)")
        plt.axvline(x=1, color="black", linestyle="--", alpha=0.5)  # Add reference line for square boxes
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dirs["distributions"], "aspect_ratio_distribution.png")
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Aspect ratio distribution saved to {output_path}")
        return output_path

    def generate_box_size_violin_plot(self) -> str:
        """Generate violin plot for box sizes by class."""
        # Prepare data for violin plot
        plot_data = []
        for class_id, sizes in self.bbox_sizes_by_class.items():
            if sizes:
                # Convert to pixel area (assuming 640x640 images)
                pixel_sizes = [size * 640 * 640 for size in sizes]
                for size in pixel_sizes:
                    plot_data.append({"Class": self.class_names[class_id], "Size (px²)": size})

        if plot_data:
            import pandas as pd

            df = pd.DataFrame(plot_data)

            plt.figure(figsize=(12, 8))
            sns.violinplot(data=df, x="Class", y="Size (px²)")
            plt.yscale("log")
            plt.title("Distribution of Object Sizes by Class")
            plt.ylabel("Bounding Box Area (pixels²)")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            output_path = os.path.join(self.output_dirs["distributions"], "box_size_violin_plot.png")
            plt.savefig(output_path)
            plt.close()

            logger.info(f"Box size violin plot saved to {output_path}")
            return output_path

        return ""

    def visualize_image_with_boxes(self, image_path: str, label_path: str, output_path: str) -> None:
        """Visualize a single image with bounding boxes."""
        # Load image
        img = Image.open(image_path)
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)

        # Load labels
        with open(label_path) as f:
            lines = f.readlines()

        # Draw bounding boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # Convert normalized coordinates to pixel values
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                # Draw rectangle and label
                color = self.class_colors[class_id]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1 - 10), self.class_names[class_id], fill=color)

        # Save the image
        img.save(output_path)

    def generate_sample_images_by_object_count(self) -> list[str]:
        """Generate sample images with different numbers of objects."""
        sample_counts = [1, 5, 10, 20, 50]
        output_paths = []

        for target_count in sample_counts:
            found = False
            # Search across all label directories
            for labels_path, images_path in zip(self.labels_paths, self.images_paths):
                if found:
                    break

                for filename in os.listdir(labels_path):
                    if not filename.endswith(".txt"):
                        continue

                    file_path = os.path.join(labels_path, filename)
                    with open(file_path) as f:
                        n_boxes = len(f.readlines())

                    if n_boxes == target_count:
                        image_name = filename.replace(".txt", ".jpg")
                        image_path = os.path.join(images_path, image_name)

                        if os.path.exists(image_path):
                            output_path = os.path.join(
                                self.output_dirs["examples_count"], f"sample_image_{target_count}_objects.png"
                            )
                            try:
                                self.visualize_image_with_boxes(image_path, file_path, output_path)
                                output_paths.append(output_path)
                                logger.info(f"Saved visualization for image with {target_count} objects")
                                found = True
                                break
                            except Exception as e:
                                logger.warning(f"Error visualizing {image_path}: {e}")

        return output_paths

    def generate_sample_images_by_class(self) -> list[str]:
        """Generate sample images with single class types."""
        class_combinations = ["aircraft_only", "ship_only", "vehicle_only"]
        output_paths = []

        for combo in class_combinations:
            target_class = combo.split("_")[0]
            target_class_id = None
            for cid, name in self.class_names.items():
                if name == target_class:
                    target_class_id = cid
                    break

            if target_class_id is None:
                continue

            found = False
            # Search across all label directories
            for labels_path, images_path in zip(self.labels_paths, self.images_paths):
                if found:
                    break

                for filename in os.listdir(labels_path):
                    if not filename.endswith(".txt"):
                        continue

                    file_path = os.path.join(labels_path, filename)
                    classes_in_image = set()

                    with open(file_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                classes_in_image.add(class_id)

                    # Check if image contains only the target class
                    if classes_in_image == {target_class_id}:
                        image_name = filename.replace(".txt", ".jpg")
                        image_path = os.path.join(images_path, image_name)

                        if os.path.exists(image_path):
                            output_path = os.path.join(self.output_dirs["examples_count"], f"sample_{combo}.png")
                            try:
                                self.visualize_image_with_boxes(image_path, file_path, output_path)
                                output_paths.append(output_path)
                                logger.info(f"Saved visualization for {combo}")
                                found = True
                                break
                            except Exception as e:
                                logger.warning(f"Error visualizing {image_path}: {e}")

        return output_paths

    def create_diverse_samples_grid(self, n_samples: int = 4) -> str:
        """Create a grid of diverse sample images."""
        # Find diverse examples across all directories
        diverse_files = []

        # Collect files from all label directories
        all_label_files = []
        for labels_path, images_path in zip(self.labels_paths, self.images_paths):
            label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]
            for filename in label_files:
                all_label_files.append((filename, labels_path, images_path))

        # Sample files with different characteristics
        step = max(1, len(all_label_files) // n_samples)
        for i in range(0, len(all_label_files), step):
            if len(diverse_files) >= n_samples:
                break

            filename, labels_path, images_path = all_label_files[i]
            image_name = filename.replace(".txt", ".jpg")
            image_path = os.path.join(images_path, image_name)
            label_path = os.path.join(labels_path, filename)

            if os.path.exists(image_path):
                diverse_files.append((image_path, label_path))

        if len(diverse_files) < n_samples:
            # Fill remaining slots with random samples
            remaining = n_samples - len(diverse_files)
            for i, (filename, labels_path, images_path) in enumerate(all_label_files[:remaining]):
                image_name = filename.replace(".txt", ".jpg")
                image_path = os.path.join(images_path, image_name)
                label_path = os.path.join(labels_path, filename)

                if os.path.exists(image_path):
                    diverse_files.append((image_path, label_path))

        # Create grid
        grid_size = int(np.ceil(np.sqrt(len(diverse_files))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (image_path, label_path) in enumerate(diverse_files):
            if i >= len(axes):
                break

            # Load and process image
            img = Image.open(image_path)
            img_array = np.array(img)

            # Load labels and draw boxes
            with open(label_path) as f:
                lines = f.readlines()

            ax = axes[i]
            ax.imshow(img_array)

            # Draw bounding boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Convert to pixel coordinates
                    img_width, img_height = img.size
                    x1 = (x_center - width / 2) * img_width
                    y1 = (y_center - height / 2) * img_height
                    x2 = (x_center + width / 2) * img_width
                    y2 = (y_center + height / 2) * img_height

                    # Draw rectangle
                    color = ["red", "green", "blue"][class_id]
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
                    ax.add_patch(rect)

            ax.set_title(f"Sample {i + 1} ({len(lines)} objects)")
            ax.axis("off")

        # Hide unused subplots
        for i in range(len(diverse_files), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        output_path = os.path.join(self.output_dirs["examples_count"], "diverse_samples_grid.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Diverse samples grid saved to {output_path}")
        return output_path

    def generate_all_visualizations(self) -> dict[str, Any]:
        """Generate all visualizations."""
        logger.info("Starting visualization generation...")

        # Collect statistics first
        self._collect_statistics()

        results = {}

        # Generate distribution plots
        results["class_distribution"] = self.generate_class_distribution()
        results["objects_per_image"] = self.generate_objects_per_image_distribution()
        results["bbox_size_distribution"] = self.generate_bbox_size_distribution()
        results["aspect_ratio_distribution"] = self.generate_aspect_ratio_distribution()
        results["box_size_violin"] = self.generate_box_size_violin_plot()

        # Generate example images
        results["sample_by_count"] = self.generate_sample_images_by_object_count()
        results["sample_by_class"] = self.generate_sample_images_by_class()
        results["diverse_grid"] = self.create_diverse_samples_grid()

        logger.success("All visualizations generated successfully")
        return results
