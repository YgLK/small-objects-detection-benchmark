"""
Invalid Bounding Box Counter Module for SkyFusion Dataset.

This module provides a class-based interface for analyzing and counting
invalid bounding boxes in YOLO format datasets.

Author: Generated for SkyFusion Dataset Analysis
"""

import os
from typing import Any, Dict, List

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


class InvalidBoxCounter:
    """Analyzes and counts invalid bounding boxes in YOLO format datasets."""

    def __init__(self, images_path, labels_path, output_dir: str):
        """
        Initialize the invalid box counter.

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

        # Define class names
        self.class_names = {0: "aircraft", 1: "ship", 2: "vehicle"}

        # Results storage
        self.invalid_boxes = []
        self.total_boxes = 0
        self.invalid_by_class = {0: 0, 1: 0, 2: 0}
        self.total_by_class = {0: 0, 1: 0, 2: 0}

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def _is_box_too_small(
        self, width: float, height: float, img_width: int, img_height: int, min_pixel_size: int = 1
    ) -> bool:
        """
        Check if a bounding box is too small to be meaningful.

        Args:
        ----
            width: Normalized width of the box
            height: Normalized height of the box
            img_width: Image width in pixels
            img_height: Image height in pixels
            min_pixel_size: Minimum size in pixels for a valid box


        Returns:
        -------
            True if the box is too small, False otherwise
        """
        pixel_width = width * img_width
        pixel_height = height * img_height
        pixel_area = pixel_width * pixel_height

        return pixel_area <= min_pixel_size

    def _analyze_label_file(self, label_path: str, image_path: str) -> list[dict[str, Any]]:
        """
        Analyze a single label file for invalid boxes.

        Args:
        ----
            label_path: Path to the label file
            image_path: Path to the corresponding image file


        Returns:
        -------
            List of invalid box information
        """
        invalid_boxes = []

        try:
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            # Process label file
            with open(label_path) as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    self.total_boxes += 1
                    self.total_by_class[class_id] += 1

                    # Check if box is too small
                    if self._is_box_too_small(width, height, img_width, img_height):
                        invalid_info = {
                            "file": os.path.basename(label_path),
                            "line": line_idx + 1,
                            "class_id": class_id,
                            "class_name": self.class_names[class_id],
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "normalized_area": width * height,
                            "pixel_area": width * height * img_width * img_height,
                            "img_width": img_width,
                            "img_height": img_height,
                        }
                        invalid_boxes.append(invalid_info)
                        self.invalid_by_class[class_id] += 1

        except Exception as e:
            logger.warning(f"Error processing {label_path}: {e}")

        return invalid_boxes

    def _generate_invalid_boxes_report(self) -> str:
        """Generate a detailed report of invalid boxes."""
        report_path = os.path.join(self.output_dir, "invalid_boxes_report.txt")

        with open(report_path, "w") as f:
            f.write("===== Invalid Bounding Boxes Analysis =====\n\n")

            f.write("Summary:\n")
            f.write(f"  Total bounding boxes: {self.total_boxes}\n")
            f.write(f"  Invalid bounding boxes: {len(self.invalid_boxes)}\n")
            f.write(f"  Percentage invalid: {len(self.invalid_boxes) / self.total_boxes * 100:.4f}%\n\n")

            f.write("Invalid boxes by class:\n")
            for class_id in sorted(self.class_names.keys()):
                invalid_count = self.invalid_by_class[class_id]
                total_count = self.total_by_class[class_id]
                percentage = invalid_count / total_count * 100 if total_count > 0 else 0
                f.write(f"  {self.class_names[class_id]}: {invalid_count}/{total_count} ({percentage:.4f}%)\n")

            if self.invalid_boxes:
                f.write("\nDetailed invalid boxes:\n")
                f.write("File\tLine\tClass\tNorm_Area\tPixel_Area\tWidth\tHeight\n")

                for box in self.invalid_boxes:
                    f.write(
                        f"{box['file']}\t{box['line']}\t{box['class_name']}\t"
                        f"{box['normalized_area']:.8f}\t{box['pixel_area']:.2f}\t"
                        f"{box['width']:.6f}\t{box['height']:.6f}\n"
                    )

        logger.info(f"Invalid boxes report saved to {report_path}")
        return report_path

    def _generate_invalid_boxes_visualization(self) -> str:
        """Generate visualization of invalid box statistics."""
        if not self.invalid_boxes:
            logger.info("No invalid boxes found, skipping visualization")
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Invalid boxes by class
        classes = [self.class_names[i] for i in sorted(self.invalid_by_class.keys())]
        invalid_counts = [self.invalid_by_class[i] for i in sorted(self.invalid_by_class.keys())]

        axes[0, 0].bar(classes, invalid_counts, color=["red", "green", "blue"])
        axes[0, 0].set_title("Invalid Boxes by Class")
        axes[0, 0].set_ylabel("Number of Invalid Boxes")

        # Add value labels
        for i, count in enumerate(invalid_counts):
            axes[0, 0].text(i, count + 0.1, str(count), ha="center", va="bottom")

        # 2. Invalid box sizes distribution
        pixel_areas = [box["pixel_area"] for box in self.invalid_boxes]
        axes[0, 1].hist(pixel_areas, bins=20, alpha=0.7, color="orange")
        axes[0, 1].set_title("Distribution of Invalid Box Sizes")
        axes[0, 1].set_xlabel("Pixel Area")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_yscale("log")

        # 3. Percentage of invalid boxes by class
        percentages = []
        for class_id in sorted(self.invalid_by_class.keys()):
            invalid_count = self.invalid_by_class[class_id]
            total_count = self.total_by_class[class_id]
            percentage = invalid_count / total_count * 100 if total_count > 0 else 0
            percentages.append(percentage)

        axes[1, 0].bar(classes, percentages, color=["red", "green", "blue"])
        axes[1, 0].set_title("Percentage of Invalid Boxes by Class")
        axes[1, 0].set_ylabel("Percentage (%)")

        # Add value labels
        for i, pct in enumerate(percentages):
            axes[1, 0].text(i, pct + 0.01, f"{pct:.3f}%", ha="center", va="bottom")

        # 4. Normalized area distribution
        norm_areas = [box["normalized_area"] for box in self.invalid_boxes]
        axes[1, 1].hist(norm_areas, bins=20, alpha=0.7, color="purple")
        axes[1, 1].set_title("Distribution of Invalid Box Normalized Areas")
        axes[1, 1].set_xlabel("Normalized Area")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_xscale("log")

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "invalid_boxes_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Invalid boxes visualization saved to {output_path}")
        return output_path

    def _save_invalid_boxes_csv(self) -> str:
        """Save invalid boxes data to CSV."""
        if not self.invalid_boxes:
            return ""

        import csv

        csv_path = os.path.join(self.output_dir, "invalid_boxes.csv")

        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "file",
                "line",
                "class_id",
                "class_name",
                "x_center",
                "y_center",
                "width",
                "height",
                "normalized_area",
                "pixel_area",
                "img_width",
                "img_height",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.invalid_boxes)

        logger.info(f"Invalid boxes CSV saved to {csv_path}")
        return csv_path

    def _visualize_sample_invalid_boxes(self, max_samples: int = 9) -> str:
        """Visualize sample images with invalid boxes highlighted."""
        if not self.invalid_boxes:
            return ""

        # Group invalid boxes by file
        boxes_by_file = {}
        for box in self.invalid_boxes:
            filename = box["file"]
            if filename not in boxes_by_file:
                boxes_by_file[filename] = []
            boxes_by_file[filename].append(box)

        # Select sample files
        sample_files = list(boxes_by_file.keys())[:max_samples]

        if not sample_files:
            return ""

        # Create grid
        grid_size = int(np.ceil(np.sqrt(len(sample_files))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, filename in enumerate(sample_files):
            if i >= len(axes):
                break

            # Find the correct image path for this file
            image_name = filename.replace(".txt", ".jpg")
            image_path = None
            label_path = None

            # Search across all directories
            for labels_dir, images_dir in zip(self.labels_paths, self.images_paths):
                potential_image_path = os.path.join(images_dir, image_name)
                potential_label_path = os.path.join(labels_dir, filename)

                if os.path.exists(potential_image_path) and os.path.exists(potential_label_path):
                    image_path = potential_image_path
                    label_path = potential_label_path
                    break

            if not image_path or not label_path:
                continue

            try:
                img = Image.open(image_path)
                img_array = np.array(img)

                ax = axes[i]
                ax.imshow(img_array)

                # Draw all boxes (valid and invalid)
                with open(label_path) as f:
                    lines = f.readlines()

                invalid_boxes_in_file = boxes_by_file[filename]
                invalid_lines = {box["line"] - 1 for box in invalid_boxes_in_file}  # Convert to 0-indexed

                for line_idx, line in enumerate(lines):
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

                        # Choose color based on validity
                        if line_idx in invalid_lines:
                            color = "red"
                            linewidth = 3
                        else:
                            color = ["blue", "green", "orange"][class_id]
                            linewidth = 1

                        # Draw rectangle
                        rect = plt.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1, linewidth=linewidth, edgecolor=color, facecolor="none"
                        )
                        ax.add_patch(rect)

                ax.set_title(f"{filename}\n({len(invalid_boxes_in_file)} invalid boxes)")
                ax.axis("off")

            except Exception as e:
                logger.warning(f"Error visualizing {image_path}: {e}")
                axes[i].axis("off")

        # Hide unused subplots
        for i in range(len(sample_files), len(axes)):
            axes[i].axis("off")

        plt.suptitle("Sample Images with Invalid Boxes (Red = Invalid)", fontsize=16)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "sample_invalid_boxes.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Sample invalid boxes visualization saved to {output_path}")
        return output_path

    def analyze(self) -> dict[str, Any]:
        """
        Run the complete invalid boxes analysis.

        Returns
        -------
            Dictionary containing analysis results
        """
        logger.info("Starting invalid boxes analysis...")

        # Process all label directories
        for labels_path, images_path in zip(self.labels_paths, self.images_paths):
            # Get list of label files
            label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]

            logger.info(f"Analyzing {len(label_files)} label files from {labels_path}...")

            # Process each label file
            for filename in tqdm(label_files, desc=f"Analyzing {os.path.basename(labels_path)}"):
                label_path = os.path.join(labels_path, filename)
                image_name = filename.replace(".txt", ".jpg")
                image_path = os.path.join(images_path, image_name)

                if os.path.exists(image_path):
                    invalid_boxes = self._analyze_label_file(label_path, image_path)
                    self.invalid_boxes.extend(invalid_boxes)

        # Generate reports and visualizations
        text_report = self._generate_invalid_boxes_report()
        visualization = self._generate_invalid_boxes_visualization()
        csv_file = self._save_invalid_boxes_csv()
        sample_viz = self._visualize_sample_invalid_boxes()

        # Compile results
        results = {
            "total_boxes": self.total_boxes,
            "invalid_boxes_count": len(self.invalid_boxes),
            "invalid_percentage": len(self.invalid_boxes) / self.total_boxes * 100 if self.total_boxes > 0 else 0,
            "invalid_by_class": dict(self.invalid_by_class),
            "total_by_class": dict(self.total_by_class),
            "invalid_boxes_details": self.invalid_boxes,
            "output_files": {
                "text_report": text_report,
                "visualization": visualization,
                "csv_file": csv_file,
                "sample_visualization": sample_viz,
            },
        }

        logger.success(
            f"Invalid boxes analysis completed. Found {len(self.invalid_boxes)} invalid boxes out of {self.total_boxes} total boxes"
        )
        return results
