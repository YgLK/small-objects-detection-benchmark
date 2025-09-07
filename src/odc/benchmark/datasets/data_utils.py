"""Utility functions for dataset loading and processing."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_yolo_label_line(line: str) -> tuple[int, float, float, float, float]:
    """Parse a single line from a YOLO format label file.

    Args:
        line: Line from label file in format "class_id x_center y_center width height"

    Returns:
        Tuple of (class_id, x_center, y_center, width, height) in normalized coordinates
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label format: {line}")

    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    return class_id, x_center, y_center, width, height


def yolo_to_xyxy(
    x_center: float, y_center: float, width: float, height: float, img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """Convert YOLO normalized coordinates to absolute xyxy format.

    Args:
        x_center: Normalized x center coordinate (0-1)
        y_center: Normalized y center coordinate (0-1)
        width: Normalized width (0-1)
        height: Normalized height (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in absolute pixel coordinates
    """
    # Convert normalized coordinates to absolute
    abs_x_center = x_center * img_width
    abs_y_center = y_center * img_height
    abs_width = width * img_width
    abs_height = height * img_height

    # Convert center+size to corner coordinates
    x_min = abs_x_center - abs_width / 2
    y_min = abs_y_center - abs_height / 2
    x_max = abs_x_center + abs_width / 2
    y_max = abs_y_center + abs_height / 2

    return x_min, y_min, x_max, y_max


def load_image_safe(image_path: str) -> np.ndarray | None:
    """Safely load an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Loaded image as numpy array (H, W, C) in BGR format, or None if failed
    """
    try:
        if not os.path.exists(image_path):
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None

        return image
    except Exception:
        return None


def get_image_dimensions(image_path: str) -> tuple[int, int] | None:
    """Get image dimensions without loading the full image.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height
    except Exception:
        return None


def find_matching_image(label_path: str, images_dir: str) -> str | None:
    """Find the corresponding image file for a label file.

    Args:
        label_path: Path to the label file (.txt)
        images_dir: Directory containing image files

    Returns:
        Path to the matching image file, or None if not found
    """
    # Get the base name without extension
    label_basename = os.path.splitext(os.path.basename(label_path))[0]

    # Common image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    for ext in image_extensions:
        image_path = os.path.join(images_dir, label_basename + ext)
        if os.path.exists(image_path):
            return image_path

    return None


def validate_yolo_annotation(
    class_id: int, x_center: float, y_center: float, width: float, height: float, num_classes: int
) -> bool:
    """Validate a YOLO format annotation.

    Args:
        class_id: Class identifier
        x_center: Normalized x center coordinate
        y_center: Normalized y center coordinate
        width: Normalized width
        height: Normalized height
        num_classes: Total number of classes in the dataset

    Returns:
        True if annotation is valid, False otherwise
    """
    # Check class ID is valid
    if class_id < 0 or class_id >= num_classes:
        return False

    # Check coordinates are normalized (0-1)
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        return False

    # Check dimensions are positive and normalized
    if not (0 < width <= 1 and 0 < height <= 1):
        return False

    # Check that bounding box doesn't exceed image boundaries
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
        return False

    return True


def calculate_bbox_area(width: float, height: float, img_width: int, img_height: int) -> float:
    """Calculate the area of a bounding box in pixels.

    Args:
        width: Normalized width (0-1)
        height: Normalized height (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Area in pixels
    """
    abs_width = width * img_width
    abs_height = height * img_height
    return abs_width * abs_height


def get_dataset_splits(dataset_root: str) -> dict[str, dict[str, str]]:
    """Get the paths for different dataset splits.

    Args:
        dataset_root: Root directory of the dataset

    Returns:
        Dictionary mapping split names to their image and label directories
    """
    splits = {}

    for split in ["train", "test", "valid"]:
        images_dir = os.path.join(dataset_root, "images", split)
        labels_dir = os.path.join(dataset_root, "labels", split)

        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            splits[split] = {"images": images_dir, "labels": labels_dir}

    return splits
