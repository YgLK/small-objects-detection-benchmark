#!/usr/bin/env python3
"""
Export COCO-format ground truth and predictions for selected models using
predefined thresholds from models/MODELSv2.yaml, to enable bootstrap-based
confidence intervals and pairwise Δ computations without re-running inference later.

Outputs (recommended):
- Ground truth: <output_annotations>
- Predictions per model: <output_predictions_dir>/<model_name>.json

Example:
  uv run scripts/stats/export_predictions.py \
    --models-yaml models/MODELSv2.yaml \
    --models yolov11m-p2-aug_20250603 rt-detr \
    --dataset-path datasets/SkyFusion_yolo \
    --split test \
    --output-annotations /home/yglk/coding/local/masters-thesis-local/data/annotations/test.json \
    --output-predictions-dir /home/yglk/coding/local/masters-thesis-local/data/predictions
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from tqdm import tqdm
import yaml


# Ensure we can import project src modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import benchmark components
from odc.benchmark import FasterRCNNModel, RFDETRModel, SkyFusionDataset, UltralyticsModel  # type: ignore


def load_models_from_yaml(models_yaml: Path, include_names: list[str]) -> list[dict[str, Any]]:
    data = yaml.safe_load(models_yaml.read_text())
    selected = []
    for m in data.get("models", []):
        name = m.get("name")
        if include_names and name not in include_names:
            continue
        params = m.get("params", {}) or m.get("config", {}) or {}
        # Translate to internal names
        translated = {}
        for k, v in params.items():
            if k == "conf_thr":
                translated["conf_threshold"] = v
            elif k == "nms_iou":
                translated["iou_threshold"] = v
            else:
                translated[k] = v
        selected.append(
            {
                "name": name,
                "path": m.get("path"),
                "type": m.get("type"),
                "config": translated,
            }
        )
    return selected


def instantiate_model(entry: dict[str, Any]):
    mtype = entry["type"]
    path = entry["path"]
    cfg = entry.get("config", {})
    if mtype in {"yolo", "rtdetr"}:
        return UltralyticsModel(path, cfg)
    elif mtype == "rfdetr":
        return RFDETRModel(path, cfg)
    elif mtype == "faster-rcnn":
        return FasterRCNNModel(path, cfg)
    else:
        raise ValueError(f"Unsupported model type: {mtype}")


def to_coco_gt(dataset: SkyFusionDataset) -> dict[str, Any]:
    images = []
    annotations = []
    # Build categories generically from observed class ids; names are optional for COCOeval
    observed_class_ids = set()
    ann_id = 0
    for img_id, sample in enumerate(dataset):
        h, w = sample.image.shape[:2]
        images.append({"id": img_id, "width": int(w), "height": int(h), "file_name": Path(sample.image_path).name})
        for ann in sample.annotations:
            x1, y1, x2, y2 = ann.bbox
            w_box = float(x2 - x1)
            h_box = float(y2 - y1)
            cid = int(ann.class_id)
            observed_class_ids.add(cid)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [float(x1), float(y1), w_box, h_box],
                    "area": float(w_box * h_box),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    cats = sorted(list(observed_class_ids)) or [0]
    categories = [{"id": cid, "name": f"class_{cid}", "supercategory": "none"} for cid in cats]
    return {"images": images, "annotations": annotations, "categories": categories}


def to_coco_dets(dataset: SkyFusionDataset, model) -> list[dict[str, Any]]:
    preds_json = []
    for img_id, sample in enumerate(tqdm(dataset, desc=f"Predicting [{Path(model.model_path).name}]", leave=False)):
        dets = model.predict(sample.image)
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            preds_json.append(
                {
                    "image_id": img_id,
                    "category_id": int(d.class_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(d.confidence),
                }
            )
    return preds_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-yaml", type=str, default=str(PROJECT_ROOT / "models/MODELSv2.yaml"))
    ap.add_argument(
        "--models", nargs="+", default=["yolov11m-p2-aug_20250603", "rt-detr"], help="Model names to export from YAML"
    )
    ap.add_argument("--dataset-path", type=str, default=str(PROJECT_ROOT / "datasets/SkyFusion_yolo"))
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--output-annotations", type=str, required=True)
    ap.add_argument("--output-predictions-dir", type=str, required=True)
    args = ap.parse_args()

    models_yaml = Path(args.models_yaml)
    include_names = list(args.models)
    models_cfg = load_models_from_yaml(models_yaml, include_names)
    if not models_cfg:
        print("No matching models found in YAML.")
        sys.exit(1)

    # Load dataset (images loaded for inference)
    dataset = SkyFusionDataset(args.dataset_path, args.split, {"load_images": True})

    # Ensure output dirs
    out_ann = Path(args.output_annotations)
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    out_pred_dir = Path(args.output_predictions_dir)
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    # Save GT COCO once
    gt_coco = to_coco_gt(dataset)
    out_ann.write_text(json.dumps(gt_coco))
    print(f"Saved GT annotations to {out_ann}")

    # Export predictions per model
    for m in models_cfg:
        model = instantiate_model(m)
        preds = to_coco_dets(dataset, model)
        out_pred = out_pred_dir / f"{m['name']}.json"
        out_pred.write_text(json.dumps(preds))
        print(f"Saved predictions for {m['name']} to {out_pred}")


if __name__ == "__main__":
    main()
