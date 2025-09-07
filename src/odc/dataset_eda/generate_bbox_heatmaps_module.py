"""
Bounding Box Heatmap Generator Module for SkyFusion Dataset.

This module provides a class-based interface for generating spatial heatmaps
showing the distribution of objects across images in YOLO format datasets.

Author: Generated for SkyFusion Dataset Analysis
"""

import os
from typing import Any, Dict, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class HeatmapGenerator:
    """Generates spatial heatmaps for object distributions in YOLO format datasets."""

    def __init__(self, labels_path, output_dir: str, image_size: tuple[int, int] = (640, 640)):
        """
        Initialize the heatmap generator.

        Args:
        ----
            labels_path: Path to the labels directory (or list of paths for combined)
            output_dir: Directory to save heatmap visualizations
            image_size: Size of images (width, height) for coordinate conversion
        """
        # Handle both single paths and lists of paths
        self.labels_paths = labels_path if isinstance(labels_path, list) else [labels_path]
        self.output_dir = output_dir
        self.image_size = image_size

        # Define class names
        self.class_names = {0: "aircraft", 1: "ship", 2: "vehicle"}

        # Storage for object centers by class
        self.object_centers = {0: [], 1: [], 2: []}
        self.all_centers = []

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def _collect_object_centers(self) -> None:
        """Collect object center coordinates from all label files."""
        # Process all label directories
        for labels_path in self.labels_paths:
            label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]

            logger.info(f"Collecting object centers from {len(label_files)} label files in {labels_path}...")

            for filename in tqdm(label_files, desc=f"Processing {os.path.basename(labels_path)}"):
                file_path = os.path.join(labels_path, filename)

                try:
                    with open(file_path) as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center = map(float, parts[1:3])

                            # Convert normalized coordinates to pixel coordinates
                            x_pixel = x_center * self.image_size[0]
                            y_pixel = y_center * self.image_size[1]

                            self.object_centers[class_id].append((x_pixel, y_pixel))
                            self.all_centers.append((x_pixel, y_pixel))

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")

    def generate_class_heatmap(self, class_id: int, bins: int = 50) -> str:
        """
        Generate heatmap for a specific class.

        Args:
        ----
            class_id: ID of the class to generate heatmap for
            bins: Number of bins for the heatmap grid


        Returns:
        -------
            Path to the saved heatmap image
        """
        centers = self.object_centers[class_id]

        if not centers:
            logger.warning(f"No objects found for class {self.class_names[class_id]}")
            return ""

        # Extract x and y coordinates
        x_coords = [center[0] for center in centers]
        y_coords = [center[1] for center in centers]

        # Create heatmap
        plt.figure(figsize=(12, 10))

        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=bins, range=[[0, self.image_size[0]], [0, self.image_size[1]]]
        )

        # Plot heatmap
        plt.imshow(
            heatmap.T,
            origin="lower",
            cmap="hot",
            interpolation="bilinear",
            extent=[0, self.image_size[0], 0, self.image_size[1]],
        )
        plt.colorbar(label="Object Density")
        plt.title(f"{self.class_names[class_id].title()} Spatial Distribution Heatmap\n({len(centers)} objects)")
        plt.xlabel("X Coordinate (pixels)")
        plt.ylabel("Y Coordinate (pixels)")

        # Invert y-axis to match image coordinates
        plt.gca().invert_yaxis()

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"{self.class_names[class_id]}_heatmap.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"{self.class_names[class_id].title()} heatmap saved to {output_path}")
        return output_path

    def generate_all_classes_heatmap(self, bins: int = 50) -> str:
        """
        Generate combined heatmap for all classes.

        Args:
        ----
            bins: Number of bins for the heatmap grid


        Returns:
        -------
            Path to the saved heatmap image
        """
        if not self.all_centers:
            logger.warning("No objects found for combined heatmap")
            return ""

        # Extract x and y coordinates
        x_coords = [center[0] for center in self.all_centers]
        y_coords = [center[1] for center in self.all_centers]

        # Create heatmap
        plt.figure(figsize=(12, 10))

        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=bins, range=[[0, self.image_size[0]], [0, self.image_size[1]]]
        )

        # Plot heatmap
        plt.imshow(
            heatmap.T,
            origin="lower",
            cmap="hot",
            interpolation="bilinear",
            extent=[0, self.image_size[0], 0, self.image_size[1]],
        )
        plt.colorbar(label="Object Density")
        plt.title(f"All Classes Spatial Distribution Heatmap\n({len(self.all_centers)} objects)")
        plt.xlabel("X Coordinate (pixels)")
        plt.ylabel("Y Coordinate (pixels)")

        # Invert y-axis to match image coordinates
        plt.gca().invert_yaxis()

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "all_classes_heatmap.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"All classes heatmap saved to {output_path}")
        return output_path

    def generate_class_comparison_heatmap(self, bins: int = 50) -> str:
        """
        Generate side-by-side comparison of class heatmaps.

        Args:
        ----
            bins: Number of bins for the heatmap grid


        Returns:
        -------
            Path to the saved comparison image
        """
        # Create subplots for each class
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        # Generate heatmap for each class
        for i, class_id in enumerate(sorted(self.class_names.keys())):
            centers = self.object_centers[class_id]

            if centers:
                x_coords = [center[0] for center in centers]
                y_coords = [center[1] for center in centers]

                # Create 2D histogram
                heatmap, xedges, yedges = np.histogram2d(
                    x_coords, y_coords, bins=bins, range=[[0, self.image_size[0]], [0, self.image_size[1]]]
                )

                # Plot heatmap
                im = axes[i].imshow(
                    heatmap.T,
                    origin="lower",
                    cmap="hot",
                    interpolation="bilinear",
                    extent=[0, self.image_size[0], 0, self.image_size[1]],
                )
                axes[i].set_title(f"{self.class_names[class_id].title()}\n({len(centers)} objects)")
                axes[i].set_xlabel("X Coordinate (pixels)")
                axes[i].set_ylabel("Y Coordinate (pixels)")
                axes[i].invert_yaxis()

                # Add colorbar
                plt.colorbar(im, ax=axes[i], label="Density")
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    f"No {self.class_names[class_id]} objects found",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{self.class_names[class_id].title()}\n(0 objects)")

        # Generate combined heatmap for the fourth subplot
        if self.all_centers:
            x_coords = [center[0] for center in self.all_centers]
            y_coords = [center[1] for center in self.all_centers]

            heatmap, xedges, yedges = np.histogram2d(
                x_coords, y_coords, bins=bins, range=[[0, self.image_size[0]], [0, self.image_size[1]]]
            )

            im = axes[3].imshow(
                heatmap.T,
                origin="lower",
                cmap="hot",
                interpolation="bilinear",
                extent=[0, self.image_size[0], 0, self.image_size[1]],
            )
            axes[3].set_title(f"All Classes Combined\n({len(self.all_centers)} objects)")
            axes[3].set_xlabel("X Coordinate (pixels)")
            axes[3].set_ylabel("Y Coordinate (pixels)")
            axes[3].invert_yaxis()

            plt.colorbar(im, ax=axes[3], label="Density")

        plt.suptitle("Spatial Distribution Heatmaps by Class", fontsize=16)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "class_comparison_heatmaps.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Class comparison heatmaps saved to {output_path}")
        return output_path

    def generate_density_statistics(self) -> str:
        """Generate statistics about object density distribution."""
        stats_path = os.path.join(self.output_dir, "density_statistics.txt")

        with open(stats_path, "w") as f:
            f.write("===== Spatial Density Statistics =====\n\n")

            # Overall statistics
            f.write(f"Total objects: {len(self.all_centers)}\n")
            f.write(f"Image dimensions: {self.image_size[0]} x {self.image_size[1]} pixels\n")
            f.write(
                f"Overall density: {len(self.all_centers) / (self.image_size[0] * self.image_size[1]):.8f} objects/pixel\n\n"
            )

            # Class-specific statistics
            for class_id in sorted(self.class_names.keys()):
                centers = self.object_centers[class_id]
                f.write(f"{self.class_names[class_id].title()} statistics:\n")
                f.write(f"  Count: {len(centers)}\n")

                if centers:
                    x_coords = [center[0] for center in centers]
                    y_coords = [center[1] for center in centers]

                    f.write(f"  X range: {min(x_coords):.1f} - {max(x_coords):.1f}\n")
                    f.write(f"  Y range: {min(y_coords):.1f} - {max(y_coords):.1f}\n")
                    f.write(f"  X mean: {np.mean(x_coords):.1f} ± {np.std(x_coords):.1f}\n")
                    f.write(f"  Y mean: {np.mean(y_coords):.1f} ± {np.std(y_coords):.1f}\n")
                    f.write(
                        f"  Density: {len(centers) / (self.image_size[0] * self.image_size[1]):.8f} objects/pixel\n"
                    )
                f.write("\n")

        logger.info(f"Density statistics saved to {stats_path}")
        return stats_path

    def generate_all_heatmaps(self) -> dict[str, Any]:
        """
        Generate all heatmap visualizations.

        Returns
        -------
            Dictionary containing paths to generated heatmaps
        """
        logger.info("Starting heatmap generation...")

        # Collect object centers
        self._collect_object_centers()

        if not self.all_centers:
            logger.warning("No objects found, skipping heatmap generation")
            return {}

        results = {}

        # Generate individual class heatmaps
        for class_id in sorted(self.class_names.keys()):
            heatmap_path = self.generate_class_heatmap(class_id)
            if heatmap_path:
                results[f"{self.class_names[class_id]}_heatmap"] = heatmap_path

        # Generate combined heatmap
        all_classes_path = self.generate_all_classes_heatmap()
        if all_classes_path:
            results["all_classes_heatmap"] = all_classes_path

        # Generate comparison heatmap
        comparison_path = self.generate_class_comparison_heatmap()
        if comparison_path:
            results["comparison_heatmap"] = comparison_path

        # Generate statistics
        stats_path = self.generate_density_statistics()
        if stats_path:
            results["density_statistics"] = stats_path

        logger.success("All heatmaps generated successfully")
        return results
