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


try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    print("Please install ensemble_boxes: pip install ensemble-boxes")
    sys.exit(1)

# --- Configuration ---
MODELS_CONFIG = [
    {
        "name": "rt-detr",
        "path": str(PROJECT_ROOT / "models/rtdetr-aug_best.pt"),
        "params": {
            "conf_thr": 0.2704984199324548,
        },
        "type": "rtdetr",
    },
    {
        "name": "yolov11m-p2-aug_20250603",
        "path": str(PROJECT_ROOT / "models/yolov11m-p2-aug_20250603.pt"),
        "params": {
            "conf_thr": 0.052566007120515956,
            "nms_iou": 0.49317179138811856,
        },
        "type": "yolo",
    },
]

DATA_DIR = str(PROJECT_ROOT / "datasets/SkyFusion_yolo")
OUTPUT_DIR = PROJECT_ROOT / "output/tiling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tiling parameters
TILE_SIZE = 512
OVERLAP_PERCENTAGES = [0.1, 0.2, 0.3]  # For validation run
FINAL_OVERLAP = 0.2  # Chosen after validation
WBF_IOU_THR = 0.55

RUN_FAST_DEV = False  # Use a small subset for quick tests

# --- Utility Functions (from previous scripts) ---


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


# --- Tiling Core Functions ---


def get_tiles(image, tile_size, overlap_percent):
    """Splits an image into overlapping tiles."""
    h, w, _ = image.shape
    overlap_px = int(tile_size * overlap_percent)
    stride = tile_size - overlap_px

    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1, y1 = x, y
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)

            # Ensure tiles are at least a minimum size
            if (x2 - x1) > overlap_px and (y2 - y1) > overlap_px:
                tiles.append((image[y1:y2, x1:x2], (x1, y1)))
    return tiles


def run_inference_on_tiles(model, image, tile_size, overlap_percent, wbf_iou_thr):
    """Runs inference on tiles and merges results using WBF."""
    h, w, _ = image.shape
    tiles = get_tiles(image, tile_size, overlap_percent)

    all_boxes = []
    all_scores = []
    all_labels = []

    for tile, (x_offset, y_offset) in tqdm(tiles, desc="  - Processing Tiles", leave=False):
        tile_h, tile_w, _ = tile.shape
        predictions = model.predict(tile)

        for pred in predictions:
            if int(pred.class_id) == 0:
                x1, y1, x2, y2 = pred.bbox
                # Convert box from tile-local to image-global coordinates
                global_x1 = (x1 + x_offset) / w
                global_y1 = (y1 + y_offset) / h
                global_x2 = (x2 + x_offset) / w
                global_y2 = (y2 + y_offset) / h

                all_boxes.append([global_x1, global_y1, global_x2, global_y2])
                all_scores.append(pred.confidence)
                all_labels.append(pred.class_id)

    if not all_boxes:
        return [], [], []

    # Merge boxes from all tiles using WBF
    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
        [all_boxes], [all_scores], [all_labels], weights=None, iou_thr=wbf_iou_thr, skip_box_thr=0.0
    )

    # Denormalize boxes
    wbf_boxes[:, 0] *= w
    wbf_boxes[:, 1] *= h
    wbf_boxes[:, 2] *= w
    wbf_boxes[:, 3] *= h

    return wbf_boxes, wbf_scores, wbf_labels.astype(int)


def convert_tiled_predictions_to_coco(dataset, model, tile_size, overlap, wbf_iou):
    coco_predictions = []
    for i, sample in tqdm(enumerate(dataset), desc="Tiled Inferring", total=len(dataset)):
        image_id = i  # Use the integer index for the image_id
        boxes, scores, labels = run_inference_on_tiles(model, sample.image, tile_size, overlap, wbf_iou)

        for box, score, label in zip(boxes, scores, labels):
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
    return coco_predictions


# --- Main Execution Logic ---


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "debug.log"
    logging.basicConfig(
        level=logging.INFO, filename=log_path, filemode="w", format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting Tiling evaluation...")
    try:
        # --- 1. Validation Run to Find Best Overlap ---
        print("--- Stage 1: Validation Run to Find Best Overlap ---")
        logging.info("--- Stage 1: Validation Run to Find Best Overlap ---")
        val_dataset_full = SkyFusionDataset(DATA_DIR, config={"classes": ["aircraft"]}, split="valid")
        val_dataset = val_dataset_full[:10] if RUN_FAST_DEV else val_dataset_full
        print(f"Using {len(val_dataset)} validation images.")

        gt_val_path = OUTPUT_DIR / "ground_truth_val.json"
        coco_gt_val_data = convert_ground_truth_to_coco_format(val_dataset)
        with open(gt_val_path, "w") as f:
            json.dump(coco_gt_val_data, f)
        gt_coco_val = COCO(str(gt_val_path))

        rt_detr_config = next(c for c in MODELS_CONFIG if c["name"] == "rt-detr")
        rt_detr_model = instantiate_model_with_thresholds(rt_detr_config, rt_detr_config["params"]["conf_thr"], None)

        val_results = []
        for overlap in tqdm(OVERLAP_PERCENTAGES, desc="Validating Overlaps"):  # Outer loop for overlaps
            logging.info(f"\nEvaluating with overlap: {overlap * 100}%")
            preds = convert_tiled_predictions_to_coco(val_dataset, rt_detr_model, TILE_SIZE, overlap, WBF_IOU_THR)
            pred_path = OUTPUT_DIR / f"preds_val_overlap_{overlap}.json"
            with open(pred_path, "w") as f:
                json.dump(preds, f)

            if not preds:
                logging.warning(f"Tiled predictions file for overlap {overlap} is empty. Skipping evaluation.")
                map_val, aps_val = 0, 0
            else:
                pred_coco = gt_coco_val.loadRes(str(pred_path))
                map_val, aps_val = evaluate_coco(gt_coco_val, pred_coco)

            val_results.append({"overlap": overlap, "mAP": map_val, "AP_S": aps_val})

        val_df = pd.DataFrame(val_results)
        print("\nValidation Results:")
        print(val_df.to_string())
        logging.info("\nValidation Results:")
        logging.info(f"\n{val_df.to_string()}")
        best_overlap = val_df.loc[val_df["AP_S"].idxmax()]["overlap"]
        print(f"\nBest overlap based on AP_S: {best_overlap * 100}%")
        logging.info(f"\nBest overlap based on AP_S: {best_overlap * 100}%")

        # --- 2. Test Run with Best Overlap ---
        print(f"\n--- Stage 2: Test Run with Best Overlap ({best_overlap * 100}%) ---")
        logging.info(f"\n--- Stage 2: Test Run with Best Overlap ({best_overlap * 100}%) ---")
        test_dataset_full = SkyFusionDataset(DATA_DIR, config={"classes": ["aircraft"]}, split="test")
        test_dataset = test_dataset_full[:10] if RUN_FAST_DEV else test_dataset_full
        print(f"Using {len(test_dataset)} test images.")

        gt_test_path = OUTPUT_DIR / "ground_truth_test.json"
        coco_gt_test_data = convert_ground_truth_to_coco_format(test_dataset)
        with open(gt_test_path, "w") as f:
            json.dump(coco_gt_test_data, f)
        gt_coco_test = COCO(str(gt_test_path))

        test_results = []
        for model_config in tqdm(MODELS_CONFIG, desc="Testing Models"):  # Outer loop for models
            model = instantiate_model_with_thresholds(
                model_config, model_config["params"]["conf_thr"], model_config["params"].get("nms_iou")
            )
            model_name = Path(model.model_path).name
            logging.info(f"\nRunning test evaluation for {model_name}...")

            preds = convert_tiled_predictions_to_coco(test_dataset, model, TILE_SIZE, best_overlap, WBF_IOU_THR)
            pred_path = OUTPUT_DIR / f"preds_test_{model_name}.json"
            with open(pred_path, "w") as f:
                json.dump(preds, f)

            if not preds:
                logging.warning(f"Tiled predictions file for model {model_name} is empty. Skipping evaluation.")
                map_test, aps_test = 0, 0
            else:
                pred_coco = gt_coco_test.loadRes(str(pred_path))
                map_test, aps_test = evaluate_coco(gt_coco_test, pred_coco)

            test_results.append({"model": model_name, "mAP": map_test, "AP_S": aps_test})

        summary_df = pd.DataFrame(test_results)
        summary_path = OUTPUT_DIR / "summary_tiling.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nTiling evaluation complete. Summary saved to {summary_path}")
        print(f"\n{summary_df.to_string()}")
        logging.info(f"\nTiling evaluation complete. Summary saved to {summary_path}")
        logging.info(f"\n{summary_df.to_string()}")

    except Exception as e:
        import traceback

        logging.error("An error occurred during execution:", exc_info=True)
        print("--- SCRIPT FAILED WITH AN EXCEPTION ---")
        traceback.print_exc()
        print("---------------------------------------")


if __name__ == "__main__":
    main()
