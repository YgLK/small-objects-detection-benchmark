"""Base classes for object detection models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass
class Detection:
    """Standardized detection format for all models.

    Attributes:
        bbox: Bounding box coordinates as (x_min, y_min, x_max, y_max) in absolute pixels
        class_id: Integer class identifier
        confidence: Detection confidence score (0.0 to 1.0)
        class_name: Optional string name of the class
    """

    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    class_id: int
    confidence: float
    class_name: str | None = None


@dataclass
class ModelMetadata:
    """Model metadata for reporting and analysis.

    Attributes:
        name: Human-readable model name
        version: Model version or variant (e.g., 'yolov8m', 'yolov8s')
        parameters: Total number of model parameters
        model_size_mb: Model file size in megabytes
        input_size: Expected input image size as (width, height)
        framework: Framework used (e.g., 'ultralytics', 'detectron2')
        additional_info: Any additional model-specific information
    """

    name: str
    version: str
    parameters: int
    model_size_mb: float
    input_size: tuple[int, int]
    framework: str
    additional_info: dict[str, Any]


class ObjectDetectionModel(ABC):
    """Abstract base class for object detection models.

    This interface ensures all model adapters provide consistent functionality
    for loading, inference, and metadata extraction.
    """

    def __init__(self, model_path: str, config: dict[str, Any]):
        """Initialize the model.

        Args:
            model_path: Path to the model weights or configuration
            config: Model-specific configuration parameters
        """
        self.model_path = model_path
        self.config = config
        self._is_loaded = False
        self._warmup_done = False

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model from the specified path.

        This method should handle the actual model loading and initialization.
        Should set self._is_loaded = True when successful.
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> list[Detection]:
        """Perform inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of Detection objects for all detected objects
        """
        pass

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Perform inference on a batch of images.

        Default implementation processes images one by one.
        Subclasses can override for optimized batch processing.

        Args:
            images: List of input images as numpy arrays

        Returns:
            List of detection lists, one for each input image
        """
        return [self.predict(image) for image in images]

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata for reporting.

        Returns:
            ModelMetadata object with model information
        """
        pass

    def warmup(self, num_iterations: int = 2) -> None:
        """Warmup the model for accurate speed measurements.

        Runs inference on dummy data to ensure GPU is ready and
        any lazy loading is completed.

        Args:
            num_iterations: Number of warmup iterations
        """
        if not self._is_loaded:
            self._load_model()

        # Create dummy image matching expected input size
        metadata = self.get_metadata()
        width, height = metadata.input_size
        dummy_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        print(f"Warming up model for {num_iterations} iterations...")
        for i in range(num_iterations):
            _ = self.predict(dummy_image)

        self._warmup_done = True
        print("Model warmup completed.")

    def measure_inference_time(self, image: np.ndarray, num_runs: int = 10) -> dict[str, float]:
        """Measure inference time statistics.

        Args:
            image: Input image for timing measurements
            num_runs: Number of inference runs for averaging

        Returns:
            Dictionary with timing statistics (mean, std, min, max) in milliseconds
        """
        if not self._warmup_done:
            self.warmup()

        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = self.predict(image)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000.0 / np.mean(times),  # Frames per second
        }

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_loaded
