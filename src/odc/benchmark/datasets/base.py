"""Base classes for object detection datasets."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, overload

import numpy as np


@dataclass
class GroundTruthAnnotation:
    """Standardized ground truth annotation format.

    Attributes:
        image_id: Unique identifier for the image
        bbox: Bounding box coordinates as (x_min, y_min, x_max, y_max) in absolute pixels
        class_id: Integer class identifier
        class_name: String name of the class
        area: Area of the bounding box in pixels
        is_crowd: Whether this is a crowd annotation (for COCO-style datasets)
    """

    image_id: str
    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    class_id: int
    class_name: str
    area: float
    is_crowd: bool = False


@dataclass
class DatasetSample:
    """Single dataset sample containing image and annotations.

    Attributes:
        image_id: Unique identifier for the image
        image_path: Path to the image file
        image: Loaded image as numpy array (H, W, C) in BGR format
        annotations: List of ground truth annotations for this image
        metadata: Additional metadata about the image/sample
    """

    image_id: str
    image_path: str
    image: np.ndarray
    annotations: list[GroundTruthAnnotation]
    metadata: dict[str, Any]


class ObjectDetectionDataset(ABC):
    """Abstract base class for object detection datasets.

    This interface ensures all dataset loaders provide consistent functionality
    for loading images, annotations, and metadata.
    """

    def __init__(self, dataset_path: str, split: str, config: dict[str, Any]):
        """Initialize the dataset.

        Args:
            dataset_path: Path to the dataset root directory
            split: Dataset split ('train', 'test', 'valid', etc.)
            config: Dataset-specific configuration parameters
        """
        self.dataset_path = dataset_path
        self.split = split
        self.config = config
        self._samples = []
        self._class_names = []
        self._is_loaded = False

    @abstractmethod
    def _load_dataset(self) -> None:
        """Load the dataset from the specified path.

        This method should populate self._samples and self._class_names.
        Should set self._is_loaded = True when successful.
        """
        pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if not self._is_loaded:
            self._load_dataset()
        return len(self._samples)

    @overload
    def __getitem__(self, idx: int) -> DatasetSample: ...

    @overload
    def __getitem__(self, idx: slice) -> list[DatasetSample]: ...

    def __getitem__(self, idx: int | slice) -> DatasetSample | list[DatasetSample]:
        """Get a dataset sample by index or slice.

        Args:
            idx: Index or slice of the sample(s) to retrieve

        Returns:
            DatasetSample object or a list of DatasetSample objects
        """
        if not self._is_loaded:
            self._load_dataset()

        if isinstance(idx, int):
            if idx < 0 or idx >= len(self._samples):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self._samples)}")

        return self._samples[idx]

    def __iter__(self) -> Iterator[DatasetSample]:
        """Iterate over all samples in the dataset."""
        if not self._is_loaded:
            self._load_dataset()

        for sample in self._samples:
            yield sample

    def get_class_names(self) -> list[str]:
        """Get the list of class names in the dataset.

        Returns:
            List of class names in order of their class IDs
        """
        if not self._is_loaded:
            self._load_dataset()
        return self._class_names.copy()

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information and statistics.

        Returns:
            Dictionary containing dataset metadata and statistics
        """
        if not self._is_loaded:
            self._load_dataset()

        # Calculate basic statistics
        total_annotations = sum(len(sample.annotations) for sample in self._samples)
        class_counts = {}
        for sample in self._samples:
            for annotation in sample.annotations:
                class_name = annotation.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            "dataset_path": self.dataset_path,
            "split": self.split,
            "num_images": len(self._samples),
            "num_annotations": total_annotations,
            "num_classes": len(self._class_names),
            "class_names": self._class_names,
            "class_counts": class_counts,
            "avg_annotations_per_image": total_annotations / len(self._samples) if self._samples else 0,
        }

    def get_sample_by_id(self, image_id: str) -> DatasetSample:
        """Get a dataset sample by image ID.

        Args:
            image_id: ID of the image to retrieve

        Returns:
            DatasetSample object for the specified image

        Raises:
            ValueError: If image ID is not found
        """
        if not self._is_loaded:
            self._load_dataset()

        for sample in self._samples:
            if sample.image_id == image_id:
                return sample

        raise ValueError(f"Image ID '{image_id}' not found in dataset")

    def filter_by_class(self, class_names: list[str]) -> list[DatasetSample]:
        """Filter samples that contain annotations for specific classes.

        Args:
            class_names: List of class names to filter by

        Returns:
            List of samples containing at least one annotation from the specified classes
        """
        if not self._is_loaded:
            self._load_dataset()

        filtered_samples = []
        for sample in self._samples:
            for annotation in sample.annotations:
                if annotation.class_name in class_names:
                    filtered_samples.append(sample)
                    break

        return filtered_samples

    def is_loaded(self) -> bool:
        """Check if the dataset is loaded and ready for use."""
        return self._is_loaded
