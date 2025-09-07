"""Ultralytics model adapter for YOLO and RT-DETR models."""

import os
from typing import Any, Dict, List, Literal

import cv2
import numpy as np

from .base import Detection, ModelMetadata, ObjectDetectionModel


class UltralyticsModel(ObjectDetectionModel):
    """Ultralytics model adapter for YOLO and RT-DETR models.

    This adapter provides a standardized interface for models
    trained with the ultralytics library, including YOLO v8/v9/v10/v11, RT-DETR, and other supported architectures.
    """

    def __init__(self, model_path: str, config: dict[str, Any]):
        """Initialize Ultralytics model.

        Args:
            model_path: Path to the Ultralytics model weights (.pt file)
            config: Configuration dictionary with optional parameters:
                - conf_threshold: Confidence threshold for detections (default: 0.25)
                - iou_threshold: IoU threshold for NMS (default: 0.45)
                - device: Device to run inference on ('cpu', 'cuda', 'auto')
                - verbose: Whether to print verbose output (default: False)
        """
        super().__init__(model_path, config)

        # Set default configuration values
        self.conf_threshold = config.get("conf_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)  # NOTE: RT-DETR doesn't use the IOU threshold
        self.device = config.get("device", "auto")
        self.verbose = config.get("verbose", False)

        # Log model parameters for verification
        model_name = os.path.basename(model_path)
        print(f"\nInitializing Ultralytics Model: {model_name}")
        print(f"   Confidence Threshold: {self.conf_threshold}")
        print(f"   IoU Threshold: {self.iou_threshold}")
        print(f"   Device: {self.device}")
        print(f"   Verbose: {self.verbose}\n")

        # Model will be loaded lazily
        self.model = None

        # Load model immediately
        if "yolo" in model_path:
            self._load_model("yolo")
        elif "rtdetr" in model_path:
            self._load_model("rtdetr")
        else:
            raise ValueError(f"Unsupported model type: {model_path}")

    def _load_model(self, model_type: Literal["yolo", "rtdetr"]) -> None:
        """Load the Ultralytics model from the specified path."""
        print(f"Loading {model_type} model from {self.model_path}...")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            if model_type == "yolo":
                from ultralytics import YOLO

                self.model = YOLO(self.model_path)
                self.model_type = model_type
            elif model_type == "rtdetr":
                from ultralytics import RTDETR

                self.model = RTDETR(self.model_path)
                self.model_type = model_type
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Configure model settings
            if hasattr(self.model, "conf"):
                self.model.conf = self.conf_threshold
            if hasattr(self.model, "iou") and model_type != "rtdetr":
                self.model.iou = self.iou_threshold

            print(f"{model_type} model loaded successfully. Classes: {self.model.names}")
            self._is_loaded = True

        except ImportError:
            raise ImportError(
                f"ultralytics library is required for {model_type} models. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Ultralytics model: {str(e)}")

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Perform inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of Detection objects for all detected objects
        """
        if not self._is_loaded:
            self._load_model()

        # Run inference
        results = self.model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold)

        # Extract detections from results
        detections = []
        if len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                # Extract detection data
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                # Convert to Detection objects
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    conf = float(confidences[i])
                    cls_id = int(class_ids[i])
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        class_id=cls_id,
                        confidence=conf,
                        class_name=cls_name,
                    )
                    detections.append(detection)

        return detections

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Perform inference on a batch of images.

        Ultralytics models support batch processing, so we override the default implementation.

        Args:
            images: List of input images as numpy arrays

        Returns:
            List of detection lists, one for each input image
        """
        if not self._is_loaded:
            self._load_model()

        if not images:
            return []

        # Run batch inference
        results = self.model(images, verbose=self.verbose)

        # Process results for each image
        all_detections = []
        for result in results:
            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                # Extract detection data
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                # Convert to Detection objects
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    conf = float(confidences[i])
                    cls_id = int(class_ids[i])
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        class_id=cls_id,
                        confidence=conf,
                        class_name=cls_name,
                    )
                    detections.append(detection)

            all_detections.append(detections)

        return all_detections

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata for reporting.

        Returns:
            ModelMetadata object with Ultralytics model information
        """
        if not self._is_loaded:
            self._load_model()

        # Extract model information
        model_info = self.model.info()  # Returns (layers, parameters, gradients, GFLOPs)
        layers, parameters, gradients, gflops = model_info

        # Get model file size
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        # Extract model variant from filename or path
        model_name = os.path.basename(self.model_path)
        model_name_lower = model_name.lower()

        # Determine model version/type
        if "yolov8n" in model_name_lower:
            version = "YOLOv8n"
        elif "yolov8s" in model_name_lower:
            version = "YOLOv8s"
        elif "yolov8m" in model_name_lower:
            version = "YOLOv8m"
        elif "yolov8l" in model_name_lower:
            version = "YOLOv8l"
        elif "yolov8x" in model_name_lower:
            version = "YOLOv8x"
        elif "yolov9" in model_name_lower:
            version = "YOLOv9"
        elif "yolov10" in model_name_lower:
            version = "YOLOv10"
        elif "rtdetr" in model_name_lower:
            version = "RT-DETR"
        elif "yolo" in model_name_lower:
            version = "YOLO"
        else:
            version = "Ultralytics"

        # Most Ultralytics models use 640x640 input size by default
        input_size = (640, 640)

        additional_info = {
            "model_file": model_name,
            "layers": layers,
            "gflops": gflops,
            "gradients": gradients,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "class_names": self.model.names,
            "num_classes": len(self.model.names),
            "model_type": self.model_type,
        }

        return ModelMetadata(
            name=model_name.replace(".pt", ""),
            version=version,
            parameters=parameters,
            model_size_mb=model_size_mb,
            input_size=input_size,
            framework="ultralytics",
            additional_info=additional_info,
        )
