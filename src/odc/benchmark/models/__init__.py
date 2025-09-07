"""Model adapters for different object detection frameworks."""

from .base import ObjectDetectionModel, Detection, ModelMetadata
from .ultralytics_adapter import UltralyticsModel
from .fasterrcnn_adapter import FasterRCNNModel
from .rfdetr_adapter import RFDETRModel

__all__ = ["ObjectDetectionModel", "Detection", "ModelMetadata", "UltralyticsModel", "FasterRCNNModel", "RFDETRModel"]
