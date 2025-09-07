import json
import logging
import os
from pathlib import Path
import sys

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


# --- Configuration ---
DATA_DIR = str(PROJECT_ROOT / "datasets/SkyFusion_yolo")
OUTPUT_DIR = PROJECT_ROOT / "output/tiling"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_FAST_DEV = False  # Use a small subset for quick tests

# Use the same parameters as dev_notebook/08_robustness.py
MODELS_CONFIG = [
    {
        "name": "yolov11m-p2-aug_20250603",
        "path": str(PROJECT_ROOT / "models/yolov11m-p2-aug_20250603.pt"),
        "params": {"conf_thr": 0.052566007120515956, "nms_iou": 0.49317179138811856},
        "type": "yolo",
    },
    {
        "name": "rt-detr",
        "path": str(PROJECT_ROOT / "models/rtdetr-aug_best.pt"),
        "params": {"conf_thr": 0.2704984199324548},
        "type": "rtdetr",
    },
]

# --- Utility Functions (mirroring 12_tiling.py) ---


def instantiate_model_with_thresholds(model_info, conf_thr, nms_thr):
    model_type = model_info["type"]
    path = model_info["path"]
    config = {"conf_threshold": conf_thr, "iou_threshold": nms_thr}

    if model_type in {"yolo", "rtdetr"}:
        return UltralyticsModel(path, config)
    elif model_type == "rfdetr":
        return RFDETRModel(path, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def convert_ground_truth_to_coco_format(dataset):
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_gt["categories"].append({"id": 0, "name": "aircraft", "supercategory": "none"})
    annotation_id = 0

    for i, sample in tqdm(enumerate(dataset), desc="Converting GT to COCO", total=len(dataset), leave=False):
        image_filename = Path(sample.image_path).name
        h, w, _ = sample.image.shape
        coco_gt["images"].append({"id": i, "width": w, "height": h, "file_name": image_filename})

        for ann in sample.annotations:
            x1, y1, x2, y2 = ann.bbox
            width = x2 - x1
            height = y2 - y1
            coco_gt["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": i,
                    "category_id": int(ann.class_id),
                    "bbox": [x1, y1, width, height],
                    "area": float(width * height),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return coco_gt


def convert_predictions_to_coco(dataset, model):
    coco_predictions = []
    for i, sample in tqdm(enumerate(dataset), desc="Running inference", total=len(dataset)):
        image_id = i
        preds = model.predict(sample.image)
        for pred in preds:
            # Only class 0 exists in this dataset
            if int(pred.class_id) != 0:
                continue
            x1, y1, x2, y2 = pred.bbox
            width = x2 - x1
            height = y2 - y1
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(pred.class_id),
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(pred.confidence),
                }
            )
    return coco_predictions


def evaluate_coco(gt_coco, pred_coco):
    coco_eval = COCOeval(gt_coco, pred_coco, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_val = coco_eval.stats[0]  # AP @[ IoU=0.50:0.95 | area=all ]
    aps_val = coco_eval.stats[3]  # AP @[ IoU=0.50:0.95 | area=small ]
    return map_val, aps_val


# --- Main ---


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "non_tiled_eval.log"
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("--- Non-Tiled COCO Evaluation (mAP@[0.5:0.95], AP_S) ---")

    # Use TEST split to match tiling Stage 2
    test_dataset_full = SkyFusionDataset(DATA_DIR, config={"classes": ["aircraft"]}, split="test")
    test_dataset = test_dataset_full[:10] if RUN_FAST_DEV else test_dataset_full
    print(f"Using {len(test_dataset)} test images.")

    # Prepare COCO GT
    gt_test_path = OUTPUT_DIR / "ground_truth_test_non_tiled.json"
    coco_gt_test_data = convert_ground_truth_to_coco_format(test_dataset)
    with open(gt_test_path, "w") as f:
        json.dump(coco_gt_test_data, f)
    gt_coco_test = COCO(str(gt_test_path))

    results = []
    for model_config in MODELS_CONFIG:
        model = instantiate_model_with_thresholds(
            model_config,
            model_config["params"].get("conf_thr"),
            model_config["params"].get("nms_iou"),
        )
        model_name = Path(model.model_path).name
        print(f"\nEvaluating model (non-tiling): {model_name}")

        preds = convert_predictions_to_coco(test_dataset, model)
        pred_path = OUTPUT_DIR / f"preds_non_tiled_{model_name}.json"
        with open(pred_path, "w") as f:
            json.dump(preds, f)

        if not preds:
            logging.warning(f"Predictions file for model {model_name} is empty. Skipping evaluation.")
            map_test, aps_test = 0.0, 0.0
        else:
            pred_coco = gt_coco_test.loadRes(str(pred_path))
            map_test, aps_test = evaluate_coco(gt_coco_test, pred_coco)

        results.append({"model": model_name, "mAP": map_test, "AP_S": aps_test})

    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "baseline_non_tiled_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nNon-tiled evaluation complete. Summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved to {summary_path}")


if __name__ == "__main__":
    main()
