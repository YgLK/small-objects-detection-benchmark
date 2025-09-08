"""SkyFusion dataset loader for YOLO format data."""

import os
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm
import yaml

from .base import DatasetSample, GroundTruthAnnotation, ObjectDetectionDataset
from .data_utils import (
    calculate_bbox_area,
    find_matching_image,
    load_image_safe,
    parse_yolo_label_line,
    validate_yolo_annotation,
    yolo_to_xyxy,
)


class SkyFusionDataset(ObjectDetectionDataset):
    """SkyFusion dataset loader for YOLO format data.

    This loader handles the SkyFusion dataset in YOLO format with support for
    train/test/valid splits and integration with the benchmark system.
    """

    def __init__(self, dataset_path: str, split: str, config: dict[str, Any]):
        """Initialize the SkyFusion dataset loader.

        Args:
            dataset_path: Path to the SkyFusion_yolo dataset root directory
            split: Dataset split ('train', 'test', 'valid')
            config: Configuration dictionary with optional parameters:
                - load_images: Whether to load images into memory (default: True)
                - validate_annotations: Whether to validate annotations (default: True)
                - max_samples: Maximum number of samples to load (default: None for all)
        """
        super().__init__(dataset_path, split, config)

        # Configuration options
        self.load_images = config.get("load_images", True)
        self.validate_annotations = config.get("validate_annotations", True)
        self.max_samples = config.get("max_samples", None)

        # Dataset paths
        self.images_dir = os.path.join(dataset_path, "images", split)
        self.labels_dir = os.path.join(dataset_path, "labels", split)
        self.dataset_yaml_path = os.path.join(dataset_path, "dataset.yaml")

        # Class information
        self._class_id_to_name = {}
        self._load_class_info()

        # Load dataset
        self._load_dataset()

    def _load_class_info(self) -> None:
        """Load class information from dataset.yaml file."""
        try:
            if os.path.exists(self.dataset_yaml_path):
                with open(self.dataset_yaml_path) as f:
                    dataset_config = yaml.safe_load(f)

                if "names" in dataset_config:
                    self._class_id_to_name = dataset_config["names"]
                    self._class_names = [self._class_id_to_name[i] for i in sorted(self._class_id_to_name.keys())]
                else:
                    raise ValueError("No 'names' section found in dataset.yaml")
            else:
                # Fallback to default SkyFusion classes
                self._class_id_to_name = {0: "aircraft", 1: "ship", 2: "vehicle"}
                self._class_names = ["aircraft", "ship", "vehicle"]
                print(f"Warning: dataset.yaml not found at {self.dataset_yaml_path}, using default classes")

        except Exception as e:
            # Fallback to default classes
            self._class_id_to_name = {0: "aircraft", 1: "ship", 2: "vehicle"}
            self._class_names = ["aircraft", "ship", "vehicle"]
            print(f"Warning: Failed to load class info from dataset.yaml: {e}, using default classes")

    def _load_dataset(self) -> None:
        """Load the dataset from the specified paths."""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        print(f"Loading SkyFusion {self.split} dataset from {self.dataset_path}...")

        # Get all label files
        label_files = [f for f in os.listdir(self.labels_dir) if f.endswith(".txt")]

        if self.max_samples is not None:
            label_files = label_files[: self.max_samples]

        print(f"Found {len(label_files)} label files for {self.split} split")

        # Process each label file
        valid_samples = 0
        invalid_samples = 0

        for label_file in tqdm(label_files, desc=f"Loading {self.split} samples"):
            try:
                sample = self._load_sample(label_file)
                if sample is not None:
                    self._samples.append(sample)
                    valid_samples += 1
                else:
                    invalid_samples += 1
            except Exception as e:
                print(f"Error loading sample {label_file}: {e}")
                invalid_samples += 1

        print(f"Loaded {valid_samples} valid samples, {invalid_samples} invalid samples")
        self._is_loaded = True

    def _load_sample(self, label_file: str) -> DatasetSample | None:
        """Load a single dataset sample.

        Args:
            label_file: Name of the label file

        Returns:
            DatasetSample object or None if loading failed
        """
        label_path = os.path.join(self.labels_dir, label_file)

        # Find corresponding image
        image_path = find_matching_image(label_path, self.images_dir)
        if image_path is None:
            print(f"Warning: No matching image found for {label_file}")
            return None

        # Load image if requested
        image = None
        if self.load_images:
            image = load_image_safe(image_path)
            if image is None:
                print(f"Warning: Failed to load image {image_path}")
                return None
        else:
            # Just check if image exists and get dimensions
            if not os.path.exists(image_path):
                return None
            # Create a dummy image array for consistency
            image = np.zeros((640, 640, 3), dtype=np.uint8)

        # Get image dimensions
        if self.load_images:
            img_height, img_width = image.shape[:2]
        else:
            # For SkyFusion, images are typically 640x640
            img_width, img_height = 640, 640
            # Try to get actual dimensions
            try:
                temp_image = cv2.imread(image_path)
                if temp_image is not None:
                    img_height, img_width = temp_image.shape[:2]
            except:
                pass

        # Load annotations
        annotations = self._load_annotations(label_path, img_width, img_height)
        if annotations is None:
            # This indicates an invalid annotation was found and should skip this sample.
            return None

        # Create image ID from filename
        image_id = os.path.splitext(label_file)[0]

        # Create metadata
        metadata = {
            "image_width": img_width,
            "image_height": img_height,
            "num_annotations": len(annotations),
            "split": self.split,
        }

        return DatasetSample(
            image_id=image_id, image_path=image_path, image=image, annotations=annotations, metadata=metadata
        )

    def _load_annotations(self, label_path: str, img_width: int, img_height: int) -> list[GroundTruthAnnotation] | None:
        """Load annotations from a YOLO format label file.

        Args:
            label_path: Path to the label file
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List of GroundTruthAnnotation objects or None if validation fails
        """
        annotations = []

        try:
            with open(label_path) as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse YOLO format line
                    class_id, x_center, y_center, width, height = parse_yolo_label_line(line)

                    # Validate annotation if requested
                    if self.validate_annotations:
                        if not validate_yolo_annotation(
                            class_id, x_center, y_center, width, height, len(self._class_names)
                        ):
                            print(
                                f"Warning: Invalid annotation in {label_path} line {line_num + 1}: {line}. Skipping sample."
                            )
                            return None

                    # Convert to absolute coordinates
                    x_min, y_min, x_max, y_max = yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height)

                    # Calculate area
                    area = calculate_bbox_area(width, height, img_width, img_height)

                    # Get class name
                    class_name = self._class_id_to_name.get(class_id, f"class_{class_id}")

                    # Create annotation
                    annotation = GroundTruthAnnotation(
                        image_id=os.path.splitext(os.path.basename(label_path))[0],
                        bbox=(x_min, y_min, x_max, y_max),
                        class_id=class_id,
                        class_name=class_name,
                        area=area,
                        is_crowd=False,
                    )

                    annotations.append(annotation)

                except Exception as e:
                    print(
                        f"Warning: Failed to parse annotation in {label_path} line {line_num + 1}: {line}, error: {e}"
                    )
                    continue

        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")

        return annotations

    def get_class_names(self) -> list[str]:
        """Get the list of class names in the dataset."""
        return self._class_names.copy()

    def get_dataset_info(self) -> dict[str, Any]:
        """Get comprehensive dataset information and statistics."""
        if not self._is_loaded:
            self._load_dataset()

        # Calculate basic statistics
        total_annotations = sum(len(sample.annotations) for sample in self._samples)
        class_counts = {}
        size_stats = {"min_area": float("inf"), "max_area": 0, "areas": []}

        for sample in self._samples:
            for annotation in sample.annotations:
                class_name = annotation.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # Track size statistics
                area = annotation.area
                size_stats["areas"].append(area)
                size_stats["min_area"] = min(size_stats["min_area"], area)
                size_stats["max_area"] = max(size_stats["max_area"], area)

        # Calculate size statistics
        if size_stats["areas"]:
            size_stats["avg_area"] = np.mean(size_stats["areas"])
            size_stats["median_area"] = np.median(size_stats["areas"])
            size_stats["std_area"] = np.std(size_stats["areas"])
        else:
            size_stats.update({"avg_area": 0, "median_area": 0, "std_area": 0})

        # Object density statistics
        objects_per_image = [len(sample.annotations) for sample in self._samples]
        density_stats = {
            "min_objects_per_image": min(objects_per_image) if objects_per_image else 0,
            "max_objects_per_image": max(objects_per_image) if objects_per_image else 0,
            "avg_objects_per_image": np.mean(objects_per_image) if objects_per_image else 0,
            "median_objects_per_image": np.median(objects_per_image) if objects_per_image else 0,
        }

        return {
            "dataset_path": self.dataset_path,
            "split": self.split,
            "num_images": len(self._samples),
            "num_annotations": total_annotations,
            "num_classes": len(self._class_names),
            "class_names": self._class_names,
            "class_counts": class_counts,
            "class_id_to_name": self._class_id_to_name,
            "size_statistics": size_stats,
            "density_statistics": density_stats,
            "images_dir": self.images_dir,
            "labels_dir": self.labels_dir,
            "config": self.config,
        }
