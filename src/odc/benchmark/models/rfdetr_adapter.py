"""RF-DETR model adapter using the rfdetr library."""

import os
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

from .base import Detection, ModelMetadata, ObjectDetectionModel


class RFDETRModel(ObjectDetectionModel):
    """RF-DETR model adapter for models trained with the rfdetr library.

    This adapter provides a standardized interface for RF-DETR models
    trained with the rfdetr library.
    """

    def __init__(self, model_path: str, config: dict[str, Any]):
        """Initialize RF-DETR model.

        Args:
            model_path: Path to the RF-DETR model weights (.pth file)
            config: Configuration dictionary with optional parameters:
                - conf_threshold: Confidence threshold for detections (default: 0.25)
                - iou_threshold: IoU threshold for NMS (default: 0.45)
                - device: Device to run inference on ('cpu', 'cuda', 'auto')
                - verbose: Whether to print verbose output (default: False)
                - class_map: Mapping from dataset class IDs to model class IDs
                - num_classes: Number of classes in the model
        """
        super().__init__(model_path, config)

        # Set default configuration values
        self.conf_threshold = config.get("conf_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_map = config.get("class_map", {1: 0, 2: 1, 3: 2})  # COCO to model mapping
        self.yolo_to_coco_map = {0: 1, 1: 2, 2: 3}  # YOLO (0,1,2) to COCO (1,2,3)
        self.verbose = config.get("verbose", False)
        self.num_classes = config.get("num_classes", 3)

        # Log model parameters for verification
        model_name = os.path.basename(model_path)
        print(f"\nInitializing RF-DETR Model: {model_name}")
        print(f"   Confidence Threshold: {self.conf_threshold}")
        print(f"   IoU Threshold: {self.iou_threshold}")
        print(f"   Device: {self.device}")
        print(f"   Num Classes: {self.num_classes}")
        print(f"   Verbose: {self.verbose}\n")

        # Model will be loaded lazily
        self.model = None

    def _load_model(self):
        """Load the RF-DETR model from the specified path."""
        try:
            from rfdetr import RFDETRBase
        except ImportError:
            raise ImportError("rfdetr library is required for RF-DETR models. Install it with: pip install rfdetr")

        if self.verbose:
            print(f"Loading RF-DETR model from {self.model_path}")

        # Initialize the model
        self.model = RFDETRBase(
            device=self.device,
            # num_classes=self.num_classes, # FIXME
            pretrain_weights=self.model_path,
        )
        self._is_loaded = True

        if self.verbose:
            print("RF-DETR model loaded successfully")

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Perform inference on a single image using RF-DETR.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of Detection objects for all detected objects
        """
        if not self._is_loaded:
            self._load_model()

        # convert BGR to RGB if needed
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input image must have 3 channels (BGR format)")

        # convert to PIL.Image as RF-DETR supports it directly
        image_pil = Image.fromarray(image_rgb)

        # run inference
        detections = self.model.predict(image_pil, threshold=self.conf_threshold)

        # convert detections to standardized format
        detection_list: list[Detection] = []

        if hasattr(detections, "xyxy") and len(detections.xyxy) > 0:
            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = bbox
                class_id = detections.class_id[i]
                confidence = detections.confidence[i]

                yolo_class_id = class_id - 1

                class_names = {-1: "background", 0: "aircraft", 1: "ship", 2: "vehicle"}
                class_name = class_names.get(yolo_class_id, f"Class_{yolo_class_id}")

                detection = Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    class_id=int(yolo_class_id),
                    confidence=float(confidence),
                    class_name=class_name,
                )
                detection_list.append(detection)

        return detection_list

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata for reporting.

        Returns:
            ModelMetadata object with model information
        """
        if not self._is_loaded:
            self._load_model()

        # Get model file size
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        return ModelMetadata(
            name="RF-DETR",
            version="base",
            parameters=self._estimate_parameters(),
            model_size_mb=model_size_mb,
            input_size=(640, 640),  # Standard RF-DETR input size
            framework="rfdetr",
            additional_info={
                "class_map": self.class_map,
                "yolo_to_coco_map": self.yolo_to_coco_map,
                "num_classes": self.num_classes,
                "conf_threshold": self.conf_threshold,
            },
        )

    def _estimate_parameters(self) -> int:
        """Estimate the number of model parameters.

        Returns:
            Estimated number of parameters
        """
        if not self._is_loaded:
            self._load_model()

        try:
            # Try to get actual parameter count if model has the attribute
            if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
                total_params = sum(p.numel() for p in self.model.model.parameters())
                return total_params
        except:
            pass

        # Fallback: estimate based on model file size
        # Rough estimate: 4 bytes per parameter (float32)
        model_size_bytes = os.path.getsize(self.model_path)
        estimated_params = model_size_bytes // 4
        return estimated_params

    def warmup(self, num_iterations: int = 2):
        """Warmup the model for accurate speed measurements.

        Args:
            num_iterations: Number of warmup iterations
        """
        if not self._is_loaded:
            self._load_model()

        if self.verbose:
            print(f"Warming up RF-DETR model with {num_iterations} iterations...")

        # Create dummy image for warmup
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        for i in range(num_iterations):
            _ = self.predict(dummy_image)

        self._warmup_done = True

        if self.verbose:
            print("RF-DETR model warmup completed")
