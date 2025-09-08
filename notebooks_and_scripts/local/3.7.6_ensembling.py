import json
import logging
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# Add project root to path to allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.odc.benchmark import (
    RFDETRModel,
    SkyFusionDataset,
    UltralyticsModel,
)


# Attempt to import ensemble_boxes, will be required for WBF
try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    print("Please install ensemble_boxes: pip install ensemble-boxes")
    sys.exit(1)

# --- Configuration ---

# Using the same top 2 models for ensembling
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
OUTPUT_DIR = PROJECT_ROOT / "output/ensembling"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_FAST_DEV = False  # Set to True to run on a small subset of data
CREATE_EXAMPLES = True
NUM_EXAMPLES = 5

# WBF parameters
WBF_IOU_THR = 0.55

# --- Utility Functions (adapted from 10_tta.py) ---


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


def convert_ground_truth_to_coco_format(dataset, image_path_to_id):
    coco_gt = {"images": [], "annotations": [], "categories": []}
    # Add all three classes to COCO categories
    coco_gt["categories"].append({"id": 0, "name": "aircraft", "supercategory": "none"})
    coco_gt["categories"].append({"id": 1, "name": "ship", "supercategory": "none"})
    coco_gt["categories"].append({"id": 2, "name": "vehicle", "supercategory": "none"})
    annotation_id = 0

    for sample in dataset:
        image_filename = Path(sample.image_path).name
        image_id = image_path_to_id[image_filename]
        h, w, _ = sample.image.shape
        coco_gt["images"].append({"id": image_id, "width": w, "height": h, "file_name": image_filename})

        for ann in sample.annotations:
            x1, y1, x2, y2 = ann.bbox
            width = x2 - x1
            height = y2 - y1
            coco_gt["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann.class_id,
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


# --- WBF Core Functions ---


def run_wbf_on_image(models, image, iou_thr, pbar=None):
    """Runs inference with multiple models and applies WBF to the results."""
    h, w, _ = image.shape

    boxes_list = []
    scores_list = []
    labels_list = []

    log_func = pbar.write if pbar else print

    for model in models:
        predictions = model.predict(image)

        boxes = []
        scores = []
        labels = []

        for pred in predictions:
            x1, y1, x2, y2 = pred.bbox
            # Normalize boxes for WBF
            boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            scores.append(pred.confidence)
            labels.append(pred.class_id)

        boxes_list.append(boxes)
        log_func(f"   - Model {Path(model.model_path).name}: Found {len(boxes)} boxes")
        scores_list.append(scores)
        labels_list.append(labels)

    # Define weights for each model (e.g., equal weights)
    weights = [1] * len(models)

    # Apply WBF
    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=0.0,  # Do not skip any boxes based on score
    )

    # De-normalize boxes
    wbf_boxes[:, 0] *= w
    wbf_boxes[:, 1] *= h
    wbf_boxes[:, 2] *= w
    wbf_boxes[:, 3] *= h

    log_func(f"  - After WBF: Found {len(wbf_boxes)} boxes")
    return wbf_boxes, wbf_scores, wbf_labels.astype(int)


def convert_wbf_predictions_to_coco(sample, models, wbf_iou_thr, image_id, pbar=None):
    """Runs WBF inference and converts final predictions to COCO format."""
    coco_predictions = []
    image = sample.image
    log_func = pbar.write if pbar else print
    log_func(f"\nRunning WBF for image {image_id}...")
    wbf_boxes, wbf_scores, wbf_labels = run_wbf_on_image(models, image, wbf_iou_thr, pbar=pbar)

    log_func(f" > WBF generated {len(wbf_boxes)} boxes before filtering.")
    for box, score, label in zip(wbf_boxes, wbf_scores, wbf_labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        coco_predictions.append(
            {
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, width, height],
                "score": float(score),
            }
        )
    log_func(f" > Found {len(coco_predictions)} predictions to save.")
    return coco_predictions


def convert_single_model_predictions_to_coco(sample, model, image_id):
    """Runs baseline inference for a single model and converts predictions to COCO format."""
    coco_predictions = []
    image = sample.image
    predictions = model.predict(image)
    for pred in predictions:
        # Include all classes, not just 'aircraft'
        x1, y1, x2, y2 = pred.bbox
        width = x2 - x1
        height = y2 - y1
        coco_predictions.append(
            {
                "image_id": image_id,
                "category_id": pred.class_id,
                "bbox": [x1, y1, width, height],
                "score": pred.confidence,
            }
        )
    return coco_predictions


# --- Visualization Functions ---


def draw_predictions(image, boxes, scores, title, color):
    """Draws bounding boxes and a title on an image."""
    img_copy = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        label = f"{score:.2f}"
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add title
    cv2.putText(img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img_copy


def save_wbf_example_images(dataset, models, output_dir, num_examples):
    """Creates and saves example images comparing individual and WBF predictions."""
    examples_dir = output_dir / "wbf_examples"
    examples_dir.mkdir(exist_ok=True)
    print(f"\nGenerating {num_examples} example images in {examples_dir}...")

    model1, model2 = models[0], models[1]
    model1_name = Path(model1.model_path).stem
    model2_name = Path(model2.model_path).stem

    for i in range(min(num_examples, len(dataset))):
        sample = dataset[i]
        image = sample.image

        # Get predictions for model 1
        preds1 = [p for p in model1.predict(image) if p.class_id == 0]
        boxes1 = [p.bbox for p in preds1]
        scores1 = [p.confidence for p in preds1]
        img1 = draw_predictions(image, boxes1, scores1, model1_name, (0, 255, 0))  # Green

        # Get predictions for model 2
        preds2 = [p for p in model2.predict(image) if p.class_id == 0]
        boxes2 = [p.bbox for p in preds2]
        scores2 = [p.confidence for p in preds2]
        img2 = draw_predictions(image, boxes2, scores2, model2_name, (255, 0, 0))  # Blue

        # Get WBF predictions
        wbf_boxes, wbf_scores, _ = run_wbf_on_image(models, image, WBF_IOU_THR)
        img3 = draw_predictions(image, wbf_boxes, wbf_scores, "WBF Ensemble", (0, 0, 255))  # Red

        # Combine and save
        combined_image = cv2.hconcat([img1, img2, img3])
        save_path = examples_dir / f"wbf_example_{i}.jpg"
        cv2.imwrite(str(save_path), combined_image)


# --- Main Execution Logic ---


def main():
    # Setup logging
    log_path = OUTPUT_DIR / "ensembling_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    log_func = logging.info
    log_to_file = lambda msg: logging.info(msg)

    log_func("Starting Ensembling (WBF) evaluation...")

    # 1. Load dataset and create GT file from VALID samples
    log_func(f"Loading SkyFusion test dataset from {DATA_DIR}...")
    CLASSES = ["aircraft", "ship", "vehicle"]
    dataset_config = {"classes": CLASSES}
    full_dataset = SkyFusionDataset(DATA_DIR, config=dataset_config, split="test")

    # Get all valid samples and create stable image_id mapping
    all_samples = [s for s in full_dataset if s is not None]
    log_func(f"Loaded {len(all_samples)} valid samples.")
    image_path_to_id = {Path(sample.image_path).name: i for i, sample in enumerate(all_samples)}

    # The ground truth file MUST contain all images for COCO evaluation to work correctly.
    log_func("Creating ground truth COCO file from all valid samples...")
    gt_coco_data = convert_ground_truth_to_coco_format(all_samples, image_path_to_id)
    gt_path = OUTPUT_DIR / "gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt_coco_data, f, indent=4)
    gt_coco = COCO(str(gt_path))

    # For dev mode, run predictions on a subset of the data.
    dataset = all_samples[:10] if RUN_FAST_DEV else all_samples
    if RUN_FAST_DEV:
        log_func(f"Using {len(dataset)} images for predictions.")

    # 2. Instantiate models
    models = []
    for config in MODELS_CONFIG:
        model = instantiate_model_with_thresholds(config, config["params"]["conf_thr"], config["params"].get("nms_iou"))
        models.append(model)

    if CREATE_EXAMPLES:
        # Use a small slice for generating examples, not the main dataset variable
        example_dataset = full_dataset[:NUM_EXAMPLES]
        save_wbf_example_images(example_dataset, models, OUTPUT_DIR, NUM_EXAMPLES)

    # 3. Run WBF evaluation
    print(f"\nRunning WBF evaluation with IoU threshold {WBF_IOU_THR}...")
    wbf_preds_path = OUTPUT_DIR / f"wbf_iou_{WBF_IOU_THR}.json"
    if not wbf_preds_path.exists():
        all_wbf_predictions = []
        for sample in tqdm(dataset, desc="Running WBF evaluation"):
            image_filename = Path(sample.image_path).name
            image_id = image_path_to_id[image_filename]
            wbf_coco_preds = convert_wbf_predictions_to_coco(sample, models, WBF_IOU_THR, image_id)
            all_wbf_predictions.extend(wbf_coco_preds)
        with open(wbf_preds_path, "w") as f:
            json.dump(all_wbf_predictions, f)

    results = []
    with open(wbf_preds_path) as f:
        wbf_preds_data = json.load(f)

    if not wbf_preds_data:
        print(f"\n⚠️  WBF predictions file is empty. Skipping evaluation for IoU={WBF_IOU_THR}.")
        results.append({"model": "WBF Ensemble", "mAP": 0, "AP_S": 0})
    else:
        pred_coco_wbf = gt_coco.loadRes(str(wbf_preds_path))
        map_wbf, aps_wbf = evaluate_coco(gt_coco, pred_coco_wbf)
        results.append({"model": "WBF Ensemble", "mAP": map_wbf, "AP_S": aps_wbf})

    # 4. Run baseline for individual models for comparison
    print("\nRunning baseline for individual models...")
    for model in models:
        model_name = Path(model.model_path).name
        baseline_preds_path = OUTPUT_DIR / f"{model_name}_baseline.json"
        if not baseline_preds_path.exists():
            all_model_predictions = []
            for sample in tqdm(dataset, desc=f"Running baseline for {model_name}"):
                image_filename = Path(sample.image_path).name
                image_id = image_path_to_id[image_filename]
                coco_preds = convert_single_model_predictions_to_coco(sample, model, image_id)
                all_model_predictions.extend(coco_preds)
            with open(baseline_preds_path, "w") as f:
                json.dump(all_model_predictions, f)
        with open(baseline_preds_path) as f:
            baseline_preds_data = json.load(f)

        if not baseline_preds_data:
            print(f"\n⚠️  Baseline predictions file for {model_name} is empty. Skipping evaluation.")
            results.append({"model": model_name, "mAP": 0, "AP_S": 0})
        else:
            pred_coco_base = gt_coco.loadRes(str(baseline_preds_path))
            map_base, aps_base = evaluate_coco(gt_coco, pred_coco_base)
            results.append({"model": model_name, "mAP": map_base, "AP_S": aps_base})

    # 5. Save summary
    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nEnsembling evaluation complete. Summary saved to {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
