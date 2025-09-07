"""FasterRCNN model adapter using PyTorch Lightning checkpoints."""

import os
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from .base import Detection, ModelMetadata, ObjectDetectionModel


class FasterRCNNModel(ObjectDetectionModel):
    """FasterRCNN model adapter for PyTorch Lightning checkpoints.

    This adapter provides a standardized interface for FasterRCNN models
    trained with PyTorch Lightning.
    """

    def __init__(self, model_path: str, config: dict[str, Any]):
        """Initialize FasterRCNN model.

        Args:
            model_path: Path to the PyTorch Lightning checkpoint (.ckpt file)
            config: Configuration dictionary with optional parameters:
                - conf_threshold: Confidence threshold for detections (default: 0.25)
                - iou_threshold: IoU threshold for NMS (default: 0.45)
                - device: Device to run inference on ('cpu', 'cuda', 'auto')
                - verbose: Whether to print verbose output (default: False)
        """
        super().__init__(model_path, config)

        # Set default configuration values
        self.conf_threshold = config.get("conf_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.device = config.get("device", "auto")
        self.verbose = config.get("verbose", False)

        # Log model parameters for verification
        model_name = os.path.basename(model_path)
        print(f"\nInitializing FasterRCNN Model: {model_name}")
        print(f"   Confidence Threshold: {self.conf_threshold}")
        print(f"   IoU Threshold: {self.iou_threshold}")
        print(f"   Device: {self.device}")
        print(f"   Verbose: {self.verbose}\n")

        # Model will be loaded lazily
        self.model = None
        self.lightning_module = None

        # Class mapping for SkyFusion dataset
        # FasterRCNN outputs: 0=background, 1=aircraft, 2=ship, 3=vehicle
        # Dataset expects: 0=aircraft, 1=ship, 2=vehicle
        self.class_names = {1: "aircraft", 2: "ship", 3: "vehicle"}
        self.class_id_mapping = {1: 0, 2: 1, 3: 2}  # Model ID -> Dataset ID

        # Load model immediately
        self._load_model()

    def _load_model(self) -> None:
        """Load the FasterRCNN model from the Lightning checkpoint."""
        try:
            import pytorch_lightning as pl
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            print(f"Loading FasterRCNN model from {self.model_path}...")

            # Define Lightning Module class (same as in training script)
            class SkyFusionModule(pl.LightningModule):
                def __init__(self, conf_threshold: float, iou_threshold: float):
                    super().__init__()
                    self.conf_threshold = conf_threshold
                    self.iou_threshold = iou_threshold
                    self.model = self._create_model()

                def _create_model(self):
                    model = fasterrcnn_resnet50_fpn(
                        pretrained=True,
                        box_score_thresh=self.conf_threshold,
                        box_nms_thresh=self.iou_threshold,
                    )
                    in_features = model.roi_heads.box_predictor.cls_score.in_features
                    # 4 classes: background + aircraft + ship + vehicle
                    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
                    return model

                def forward(self, image):
                    return self.model(image)

            # Load the Lightning module from checkpoint
            self.lightning_module = SkyFusionModule.load_from_checkpoint(
                self.model_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                map_location="cpu",
            )

            # Extract the actual model
            self.model = self.lightning_module.model

            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model.to(self.device)
            self.model.eval()

            # Set up transforms
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

            self._is_loaded = True
            print(f"FasterRCNN model loaded successfully. Classes: {self.class_names}")

        except ImportError as e:
            raise ImportError(
                f"Required libraries not available: {str(e)}. Ensure pytorch-lightning and torchvision are installed."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FasterRCNN model: {str(e)}")

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Perform inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of Detection objects for all detected objects
        """
        if not self._is_loaded:
            self._load_model()

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image and apply transforms
        pil_image = Image.fromarray(image_rgb)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Extract detections from predictions
        detections = []
        if len(predictions) > 0:
            pred = predictions[0]

            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(scores[i])
                model_cls_id = int(labels[i])

                # Convert FasterRCNN class ID to dataset format
                if model_cls_id in self.class_id_mapping:
                    dataset_cls_id = self.class_id_mapping[model_cls_id]
                    cls_name = self.class_names.get(model_cls_id, f"class_{model_cls_id}")

                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        class_id=dataset_cls_id,  # Use dataset format class ID
                        confidence=conf,
                        class_name=cls_name,
                    )
                    detections.append(detection)

        return detections

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Perform inference on a batch of images.

        Args:
            images: List of input images as numpy arrays

        Returns:
            List of detection lists, one for each input image
        """
        if not self._is_loaded:
            self._load_model()

        if not images:
            return []

        # Convert images to tensors
        image_tensors = []
        for image in images:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            image_tensor = self.transform(pil_image)
            image_tensors.append(image_tensor)

        # Stack into batch
        batch_tensor = torch.stack(image_tensors).to(self.device)

        # Run batch inference
        with torch.no_grad():
            predictions = self.model(batch_tensor)

        # Process results for each image
        all_detections = []
        for pred in predictions:
            detections = []

            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(scores[i])
                cls_id = int(labels[i])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

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
            ModelMetadata object with FasterRCNN model information
        """
        if not self._is_loaded:
            self._load_model()

        # Count parameters
        parameters = sum(p.numel() for p in self.model.parameters())

        # Get model file size
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        # Extract model name from filename
        model_name = os.path.basename(self.model_path)

        # FasterRCNN typically uses variable input size, but we'll use a common size
        input_size = (800, 600)  # Common FasterRCNN input size

        additional_info = {
            "model_file": model_name,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "class_names": self.class_names,
            "num_classes": len(self.class_names) + 1,  # +1 for background
            "framework": "pytorch_lightning",
            "backbone": "ResNet50-FPN",
        }

        return ModelMetadata(
            name=model_name.replace(".ckpt", ""),
            version="FasterRCNN-ResNet50-FPN",
            parameters=parameters,
            model_size_mb=model_size_mb,
            input_size=input_size,
            framework="pytorch_lightning",
            additional_info=additional_info,
        )
