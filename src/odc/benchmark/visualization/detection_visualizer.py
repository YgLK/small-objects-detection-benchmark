"""Detection visualization for showing model predictions on sample images."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from ..datasets.base import DatasetSample, GroundTruthAnnotation
from ..models.base import Detection


class DetectionVisualizer:
    """Visualize detection results on sample images."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the detection visualizer.

        Args:
            config: Configuration dictionary with visualization options:
                - colors: Color mapping for classes
                - line_thickness: Thickness of bounding box lines
                - font_size: Font size for labels
                - confidence_threshold: Minimum confidence to display
        """
        self.config = config
        self.line_thickness = config.get("line_thickness", 2)
        self.font_size = config.get("font_size", 12)
        self.confidence_threshold = config.get("confidence_threshold", 0.25)

        # Default colors for classes
        self.colors = config.get(
            "colors",
            {
                "aircraft": (255, 0, 0),  # Red
                "ship": (0, 255, 0),  # Green
                "vehicle": (0, 0, 255),  # Blue
                "ground_truth": (255, 255, 0),  # Yellow for ground truth
                "prediction": (255, 0, 255),  # Magenta for predictions
            },
        )

    def visualize_sample_detections(
        self,
        sample: DatasetSample,
        detections: list[Detection],
        output_path: str,
        show_ground_truth: bool = True,
        show_predictions: bool = True,
    ) -> str:
        """Visualize detections on a single sample image.

        Args:
            sample: Dataset sample with image and ground truth
            detections: Model predictions
            output_path: Path to save the visualization
            show_ground_truth: Whether to show ground truth boxes
            show_predictions: Whether to show prediction boxes

        Returns:
            Path to saved visualization
        """
        # Load image
        if sample.image is not None:
            image = sample.image.copy()
        else:
            image = cv2.imread(sample.image_path)

        if image is None:
            raise ValueError(f"Could not load image from {sample.image_path}")

        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)

        # Draw ground truth boxes
        if show_ground_truth and sample.annotations:
            for annotation in sample.annotations:
                self._draw_bbox(
                    ax,
                    annotation.bbox,
                    annotation.class_name,
                    color=self.colors.get("ground_truth", (255, 255, 0)),
                    label_prefix="GT: ",
                    linestyle="--",
                )

        # Draw prediction boxes
        if show_predictions:
            for detection in detections:
                if detection.confidence >= self.confidence_threshold:
                    self._draw_bbox(
                        ax,
                        detection.bbox,
                        detection.class_name,
                        color=self.colors.get(detection.class_name, (255, 0, 255)),
                        confidence=detection.confidence,
                        label_prefix="Pred: ",
                    )

        # Set title and labels
        ax.set_title(f"Detection Results - {sample.image_id}", fontsize=14, fontweight="bold")
        ax.set_xlabel(
            f"Ground Truth: {len(sample.annotations)} objects, "
            f"Predictions: {len([d for d in detections if d.confidence >= self.confidence_threshold])} objects"
        )

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add legend
        legend_elements = []
        if show_ground_truth:
            legend_elements.append(
                patches.Patch(
                    color=np.array(self.colors.get("ground_truth", (255, 255, 0))) / 255, label="Ground Truth"
                )
            )
        if show_predictions:
            for class_name in set(d.class_name for d in detections):
                legend_elements.append(
                    patches.Patch(
                        color=np.array(self.colors.get(class_name, (255, 0, 255))) / 255, label=f"Pred: {class_name}"
                    )
                )

        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _draw_bbox(
        self,
        ax,
        bbox: tuple[float, float, float, float],
        class_name: str,
        color: tuple[int, int, int],
        confidence: float | None = None,
        label_prefix: str = "",
        linestyle: str = "-",
    ):
        """Draw a bounding box on the axes.

        Args:
            ax: Matplotlib axes
            bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
            class_name: Object class name
            color: RGB color tuple
            confidence: Detection confidence (optional)
            label_prefix: Prefix for the label
            linestyle: Line style for the box
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Normalize color to 0-1 range
        color_norm = tuple(c / 255.0 for c in color)

        # Draw rectangle
        rect = Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=self.line_thickness,
            edgecolor=color_norm,
            facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)

        # Create label
        if confidence is not None:
            label = f"{label_prefix}{class_name} ({confidence:.2f})"
        else:
            label = f"{label_prefix}{class_name}"

        # Add text label
        ax.text(
            x_min,
            y_min - 5,
            label,
            fontsize=self.font_size,
            color=color_norm,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    def create_comparison_grid(
        self,
        samples: list[DatasetSample],
        model_detections: dict[str, list[list[Detection]]],
        output_path: str,
        max_samples: int = 4,
    ) -> str:
        """Create a grid comparing detections from multiple models.

        Args:
            samples: List of dataset samples
            model_detections: Dictionary mapping model names to their detections
            output_path: Path to save the comparison grid
            max_samples: Maximum number of samples to show

        Returns:
            Path to saved comparison grid
        """
        num_models = len(model_detections)
        num_samples = min(len(samples), max_samples)

        # Create figure with subplots
        fig, axes = plt.subplots(num_samples, num_models + 1, figsize=(4 * (num_models + 1), 4 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        model_names = list(model_detections.keys())

        for sample_idx in range(num_samples):
            sample = samples[sample_idx]

            # Load image
            if sample.image is not None:
                image = sample.image.copy()
            else:
                image = cv2.imread(sample.image_path)

            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # First column: Ground truth
            ax = axes[sample_idx, 0]
            ax.imshow(image_rgb)

            # Draw ground truth boxes
            for annotation in sample.annotations:
                self._draw_bbox_simple(
                    ax, annotation.bbox, annotation.class_name, color=self.colors.get("ground_truth", (255, 255, 0))
                )

            ax.set_title(f"Ground Truth\n{len(sample.annotations)} objects", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            # Subsequent columns: Model predictions
            for model_idx, model_name in enumerate(model_names):
                ax = axes[sample_idx, model_idx + 1]
                ax.imshow(image_rgb)

                # Draw predictions
                detections = model_detections[model_name][sample_idx]
                valid_detections = [d for d in detections if d.confidence >= self.confidence_threshold]

                for detection in valid_detections:
                    self._draw_bbox_simple(
                        ax,
                        detection.bbox,
                        detection.class_name,
                        color=self.colors.get(detection.class_name, (255, 0, 255)),
                    )

                ax.set_title(f"{model_name}\n{len(valid_detections)} detections", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle("Model Detection Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _draw_bbox_simple(
        self, ax, bbox: tuple[float, float, float, float], class_name: str, color: tuple[int, int, int]
    ):
        """Draw a simple bounding box without labels (for grid comparisons)."""
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Normalize color to 0-1 range
        color_norm = tuple(c / 255.0 for c in color)

        # Draw rectangle
        rect = Rectangle(
            (x_min, y_min), width, height, linewidth=self.line_thickness, edgecolor=color_norm, facecolor="none"
        )
        ax.add_patch(rect)

    def visualize_class_distribution(self, samples: list[DatasetSample], output_path: str) -> str:
        """Visualize the distribution of object classes in the dataset.

        Args:
            samples: List of dataset samples
            output_path: Path to save the visualization

        Returns:
            Path to saved visualization
        """
        # Count classes
        class_counts = {}
        for sample in samples:
            for annotation in sample.annotations:
                class_name = annotation.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Pie chart
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors_list = [np.array(self.colors.get(cls, (128, 128, 128))) / 255.0 for cls in classes]

        ax1.pie(counts, labels=classes, colors=colors_list, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Class Distribution (Pie Chart)")

        # Bar chart
        ax2.bar(classes, counts, color=colors_list)
        ax2.set_title("Class Distribution (Bar Chart)")
        ax2.set_xlabel("Object Classes")
        ax2.set_ylabel("Number of Instances")

        # Add value labels on bars
        for i, count in enumerate(counts):
            ax2.text(i, count + max(counts) * 0.01, str(count), ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path
