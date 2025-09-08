import json
import os
from pathlib import Path
import sys

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from tqdm import tqdm


# Add project root to path to allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.odc.benchmark import (
    Detection,
    RFDETRModel,
    SkyFusionDataset,
    UltralyticsModel,
)


# --- Configuration ---
MODELS_CONFIG = [
    {
        "name": "yolov11m-p2-aug_20250603",
        "path": str(PROJECT_ROOT / "models/yolov11m-p2-aug_20250603.pt"),
        "params": {
            "conf_thr": 0.052566007120515956,
            "nms_iou": 0.49317179138811856,
        },
        "type": "yolo",
    },
    {
        "name": "rt-detr",
        "path": str(PROJECT_ROOT / "models/rtdetr-aug_best.pt"),
        "params": {
            "conf_thr": 0.2704984199324548,
        },
        "type": "rtdetr",
    },
]

DATA_DIR = str(PROJECT_ROOT / "datasets/SkyFusion_yolo")
OUTPUT_DIR = PROJECT_ROOT / "output/tta"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_FAST_DEV = False  # Set to True to run on a small subset of data

# --- TTA Augmentations ---
TTA_TRANSFORMS = {
    "hflip": A.HorizontalFlip(p=1.0),
    "brightness": A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
}

# --- Utility Functions (adapted from 08_robustness.py) ---


def instantiate_model_with_thresholds(model_info, conf_thr, nms_thr):
    model_type = model_info["type"]
    path = model_info["path"]
    config = {"conf_threshold": conf_thr, "iou_threshold": nms_thr}

    if model_type == "yolo" or model_type == "rtdetr":
        return UltralyticsModel(path, config)
    elif model_type == "rfdetr":
        return RFDETRModel(path, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def convert_ground_truth_to_coco_format(dataset):
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_gt["categories"].append({"id": 0, "name": "aircraft", "supercategory": "none"})
    annotation_id = 0

    for i, sample in enumerate(dataset):
        h, w, _ = sample.image.shape
        coco_gt["images"].append({"id": i, "width": w, "height": h, "file_name": Path(sample.image_path).name})

        for ann in sample.annotations:
            x1, y1, x2, y2 = ann.bbox
            width = x2 - x1
            height = y2 - y1
            coco_gt["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": i,
                    "category_id": 0,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return coco_gt


def evaluate_coco(gt_coco, pred_coco):
    coco_eval = COCOeval(gt_coco, pred_coco, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_val = coco_eval.stats[0]  # AP @[ IoU=0.50:0.95 | area=all ]
    aps_val = coco_eval.stats[3]  # AP @[ IoU=0.50:0.95 | area=small ]
    return map_val, aps_val


def convert_predictions_to_coco_format(dataset, model):
    """Runs baseline inference and converts predictions to COCO format."""
    coco_predictions = []

    for i, sample in enumerate(tqdm(dataset, desc=f"Baseline Inferring {Path(model.model_path).name}")):
        image = sample.image
        predictions = model.predict(image)
        image_id = i  # Use the integer index for the image_id

        for pred in predictions:
            x1, y1, x2, y2 = pred.bbox
            width = x2 - x1
            height = y2 - y1
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 0,  # Assuming single class 'aircraft'
                    "bbox": [x1, y1, width, height],
                    "score": pred.confidence,
                }
            )
    return coco_predictions


# --- TTA Core Functions ---


def deaugment_predictions(predictions, transform_name, image_width):
    """De-augments predictions from a transformed image back to original coordinates."""
    # If the transform is not a geometric one that changes coordinates, return the predictions as is.
    if transform_name != "hflip":
        return predictions

    deaug_preds = []
    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox
        # Flip the horizontal coordinates
        x1_new = image_width - x2
        x2_new = image_width - x1
        deaug_preds.append(Detection(bbox=(x1_new, y1, x2_new, y2), confidence=pred.confidence, class_id=pred.class_id))
    return deaug_preds


def merge_predictions(all_predictions, nms_iou_threshold=0.5):
    """Merges predictions from multiple sources (original + TTA) using NMS."""
    if not all_predictions:
        return []

    boxes = torch.tensor([p.bbox for p in all_predictions], dtype=torch.float32)
    scores = torch.tensor([p.confidence for p in all_predictions])
    # Assuming single class, so all labels are 0
    labels = torch.tensor([0 for _ in all_predictions])

    # Use torchvision's batched NMS. Since there's only one class, it's equivalent to standard NMS.
    # Note: torchvision.ops.nms is for single-class NMS.
    keep_indices = torch.ops.torchvision.nms(boxes, scores, nms_iou_threshold)

    final_predictions = [all_predictions[i] for i in keep_indices]
    return final_predictions


def predict_with_tta(model, image):
    """Runs inference on an image and its augmented versions, then merges results."""
    h, w, _ = image.shape
    all_preds = []

    # 1. Original image
    original_preds = model.predict(image)
    all_preds.extend(original_preds)

    # 2. Augmented images
    for name, transform in TTA_TRANSFORMS.items():
        augmented_image = transform(image=image)["image"]
        tta_preds = model.predict(augmented_image)
        deaug_preds = deaugment_predictions(tta_preds, name, w)
        all_preds.extend(deaug_preds)

    # 3. Merge all predictions
    # Use the model's own NMS threshold for merging TTA results, with a fallback
    nms_threshold = model.config.get("iou_threshold")
    if nms_threshold is None:
        nms_threshold = 0.5  # Default fallback
    final_preds = merge_predictions(all_preds, nms_iou_threshold=nms_threshold)
    return final_preds


def convert_tta_predictions_to_coco(dataset, model):
    """Runs TTA inference and converts final predictions to COCO format."""
    coco_predictions = []

    for i, sample in enumerate(tqdm(dataset, desc=f"TTA Inferring {Path(model.model_path).name}")):
        image = sample.image
        final_predictions = predict_with_tta(model, image)
        image_id = i  # Use the integer index for the image_id

        for pred in final_predictions:
            x1, y1, x2, y2 = pred.bbox
            width = x2 - x1
            height = y2 - y1
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 0,  # Assuming single class 'aircraft'
                    "bbox": [x1, y1, width, height],
                    "score": pred.confidence,
                }
            )
    return coco_predictions


# --- Main Execution Logic ---


def main():
    print("Starting TTA evaluation...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset and create a definitive list of valid samples
    print("Loading and filtering dataset...")
    dataset_config = {"classes": ["aircraft"]}
    full_dataset = SkyFusionDataset(DATA_DIR, config=dataset_config, split="test")
    # This is the single source of truth for our dataset to ensure consistency
    valid_samples = [s for s in full_dataset if s is not None]
    dataset = valid_samples[:10] if RUN_FAST_DEV else valid_samples
    print(f"Using {len(dataset)} images for evaluation.")

    # 2. Create and load the ground truth file to ensure consistency.
    gt_path = OUTPUT_DIR / "ground_truth.json"
    print("Creating ground truth COCO file...")
    coco_gt_data = convert_ground_truth_to_coco_format(dataset)
    with open(gt_path, "w") as f:
        json.dump(coco_gt_data, f)

    # Load the generated GT file to create the official COCO object for evaluation.
    gt_coco = COCO(str(gt_path))

    results = []
    for model_config in MODELS_CONFIG:
        model_name = model_config["name"]
        params = model_config["params"]
        model = instantiate_model_with_thresholds(model_config, params["conf_thr"], params.get("nms_iou"))

        # --- Baseline Evaluation ---
        print(f"\nRunning baseline evaluation for {model_name}...")
        baseline_preds_path = OUTPUT_DIR / f"{model_name}_baseline.json"
        if not baseline_preds_path.exists():
            baseline_preds = convert_predictions_to_coco_format(dataset, model)
            with open(baseline_preds_path, "w") as f:
                json.dump(baseline_preds, f)

        pred_coco_base = gt_coco.loadRes(str(baseline_preds_path))
        map_base, aps_base = evaluate_coco(gt_coco, pred_coco_base)
        results.append(
            {
                "model": model_name,
                "method": "Baseline",
                "mAP": map_base,
                "AP_S": aps_base,
            }
        )

        # --- TTA Evaluation ---
        print(f"\nRunning TTA evaluation for {model_name}...")
        tta_preds_path = OUTPUT_DIR / f"{model_name}_tta.json"
        if not tta_preds_path.exists():
            tta_preds = convert_tta_predictions_to_coco(dataset, model)
            with open(tta_preds_path, "w") as f:
                json.dump(tta_preds, f)

        pred_coco = gt_coco.loadRes(str(tta_preds_path))
        map_tta, aps_tta = evaluate_coco(gt_coco, pred_coco)
        results.append(
            {
                "model": model_name,
                "method": "TTA",
                "mAP": map_tta,
                "AP_S": aps_tta,
            }
        )

    # --- Save Summary ---
    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nTTA evaluation complete. Summary saved to {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
