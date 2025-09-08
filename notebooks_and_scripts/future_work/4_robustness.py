import json
import os
from pathlib import Path
import sys

import albumentations as A
import cv2
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
OUTPUT_DIR = PROJECT_ROOT / "output/robustness"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_FAST_DEV = False  # Set to True to run on a small subset of data

CORRUPTION_LEVELS = {
    "jpeg": [30, 50, 70, 90],
    "noise": [5, 10, 15, 20],
}
CORRUPTIONS = {
    "jpeg": [A.Compose([A.ImageCompression(quality_range=(q, q), p=1.0)]) for q in CORRUPTION_LEVELS["jpeg"]],
    "noise": [A.Compose([A.GaussNoise(var_limit=(s * s, s * s), mean=0, p=1.0)]) for s in CORRUPTION_LEVELS["noise"]],
}

# --- Utility Functions (adapted from 07_multi.py) ---


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
    # AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
    map_val = coco_eval.stats[0]
    # AP @[ IoU=0.50:0.95 | area=small | maxDets=100 ]
    aps_val = coco_eval.stats[3]
    return map_val, aps_val


def save_corruption_examples(dataset, output_dir):
    """Saves visual examples of corruptions applied to a sample image."""
    examples_dir = output_dir / "corruption_examples"
    examples_dir.mkdir(exist_ok=True)

    # Use the first image as an example
    sample = dataset[0]
    original_image = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)

    # Save original
    cv2.imwrite(str(examples_dir / "original.png"), original_image)

    # Save JPEG examples
    for level in CORRUPTION_LEVELS["jpeg"]:
        transform = A.Compose([A.ImageCompression(quality_lower=level, quality_upper=level, p=1.0)])
        corrupted_image = transform(image=sample.image)["image"]
        corrupted_image_bgr = cv2.cvtColor(corrupted_image, cv2.COLOR_RGB2BGR)
        filename = f"jpeg_quality_{level}.png"
        cv2.imwrite(str(examples_dir / filename), corrupted_image_bgr)

    # Save Noise examples
    for level in CORRUPTION_LEVELS["noise"]:
        transform = A.Compose([A.GaussNoise(var_limit=(level * level, level * level), mean=0, p=1.0)])
        corrupted_image = transform(image=sample.image)["image"]
        corrupted_image_bgr = cv2.cvtColor(corrupted_image, cv2.COLOR_RGB2BGR)
        filename = f"noise_std_{level}.png"
        cv2.imwrite(str(examples_dir / filename), corrupted_image_bgr)

    print(f"Corruption examples saved to {examples_dir}")


# --- Main Execution Logic ---


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {OUTPUT_DIR}")

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
    # This ensures the image_ids in the evaluator match the predictions perfectly.
    gt_coco = COCO(str(gt_path))

    # Save visual examples of corruptions
    save_corruption_examples(full_dataset, OUTPUT_DIR)

    # 2. Run baseline and robustness tests
    results = []
    for model_config in MODELS_CONFIG:
        model_name = model_config["name"]
        params = model_config["params"]
        model = instantiate_model_with_thresholds(model_config, params["conf_thr"], params.get("nms_iou"))

        # Baseline (no corruption)
        print(f"\nRunning baseline for {model_name}...")
        baseline_preds = []
        for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Baseline Preds"):
            if RUN_FAST_DEV and i >= 10:
                break
            image = sample.image
            predictions = model.predict(image)
            image_id = i  # Use the integer index for the image_id
            for pred in predictions:
                x1, y1, x2, y2 = pred.bbox
                width = x2 - x1
                height = y2 - y1
                baseline_preds.append(
                    {
                        "image_id": image_id,
                        "category_id": 0,  # Assuming single class 'aircraft'
                        "bbox": [x1, y1, width, height],
                        "score": pred.confidence,
                    }
                )
        baseline_preds_path = OUTPUT_DIR / f"{model_name}_baseline.json"
        with open(baseline_preds_path, "w") as f:
            json.dump(baseline_preds, f)

        pred_coco = gt_coco.loadRes(str(baseline_preds_path))
        map_base, aps_base = evaluate_coco(gt_coco, pred_coco)
        results.append(
            {
                "model": model_name,
                "corruption": "baseline",
                "level": 0,
                "mAP": map_base,
                "AP_S": aps_base,
                "map_delta": 0,
                "aps_delta": 0,
            }
        )

        # Corruptions
        for corr_type, transforms in CORRUPTIONS.items():
            for i, transform in enumerate(transforms):
                level = CORRUPTION_LEVELS[corr_type][i]
                print(f"Running {model_name} with {corr_type} level {level}...")
                corrupted_preds = []
                for j, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"{corr_type} Preds"):
                    if RUN_FAST_DEV and j >= 10:
                        break
                    image = sample.image
                    if transform:
                        image = transform(image=image)["image"]
                    predictions = model.predict(image)
                    image_id = j  # Use the integer index for the image_id
                    for pred in predictions:
                        x1, y1, x2, y2 = pred.bbox
                        width = x2 - x1
                        height = y2 - y1
                        corrupted_preds.append(
                            {
                                "image_id": image_id,
                                "category_id": 0,  # Assuming single class 'aircraft'
                                "bbox": [x1, y1, width, height],
                                "score": pred.confidence,
                            }
                        )
                preds_path = OUTPUT_DIR / f"{model_name}_{corr_type}_{level}.json"
                with open(preds_path, "w") as f:
                    json.dump(corrupted_preds, f)

                pred_coco = gt_coco.loadRes(str(preds_path))
                map_corr, aps_corr = evaluate_coco(gt_coco, pred_coco)
                results.append(
                    {
                        "model": model_name,
                        "corruption": corr_type,
                        "level": level,
                        "mAP": map_corr,
                        "AP_S": aps_corr,
                        "map_delta": map_corr - map_base,
                        "aps_delta": aps_corr - aps_base,
                    }
                )

    # 3. Save summary
    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nRobustness evaluation complete. Summary saved to {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
