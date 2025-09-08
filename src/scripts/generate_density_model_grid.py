#!/usr/bin/env python3
"""
Generate a portrait comparison grid for spatial-density buckets (sparse, medium, dense).

Layout: 3 columns (sparse, medium, dense) x 3 rows (Ground Truth, Model A, Model B).

For each bucket, pick one representative 640x640 image from the test split.
Row 1 overlays ground-truth boxes. Rows 2-3 show predictions from two selected
evaluation models (configurable via --models).

Output: PNG saved under output/spatial-density/visualizations/density_model_comparison_grid.png

This uses the same models and thresholds as the evaluation scripts to ensure
consistency with reported results.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple


# Add repo src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../dpm3/src

# Third-party
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Project imports
from src.odc.benchmark.datasets import SkyFusionDataset
from src.odc.benchmark.models import (
    FasterRCNNModel,
    RFDETRModel,
    UltralyticsModel,
)
from src.odc.benchmark.visualization.detection_visualizer import DetectionVisualizer


# Density bins (inclusive lower, exclusive upper bound)
DENSITY_BINS: dict[str, tuple[int, float]] = {
    "sparse": (0, 10),  # 0-9 objects
    "medium": (10, 30),  # 10-29 objects
    "dense": (30, float("inf")),  # ≥30 objects
}

# Models consistent with prior evaluations
MODELS_CONFIGS = [
    {
        "name": "yolov8m-aug-update_20250603",
        "path": "models/yolov8m-aug-update_20250603.pt",
        "params": {"conf_thr": 0.06291186979342091, "nms_iou": 0.3358295483691517},
        "type": "yolo",
    },
    {
        "name": "yolov11m-p2-aug_20250603",
        "path": "models/yolov11m-p2-aug_20250603.pt",
        "params": {"conf_thr": 0.052566007120515956, "nms_iou": 0.49317179138811856},
        "type": "yolo",
    },
    {
        "name": "rf-detr",
        "path": "models/rfdetr_best_total.pth",
        "params": {"conf_thr": 0.09616820140192325},
        "type": "rfdetr",
    },
    {
        "name": "faster-rcnn",
        "path": "models/fasterrcnn-best-epoch=18-val_map=0.31.ckpt",
        "params": {"conf_thr": 0.07957236023833904, "nms_iou": 0.621230971215935},
        "type": "faster-rcnn",
    },
    {
        "name": "rt-detr",
        "path": "models/rtdetr-aug_best.pt",
        "params": {"conf_thr": 0.2704984199324548},
        "type": "rtdetr",
    },
]


# Mapping from internal model IDs to thesis display names (for row labels)
DISPLAY_NAMES = {
    "yolov8m-aug-update_20250603": "YOLOv8m (Optimized Aug, CosLR)",
    "yolov11m-p2-aug_20250603": "YOLOv11m (P2, Optimized Aug)",
    "rf-detr": "RF-DETR",
    "faster-rcnn": "Faster R-CNN",
    "rt-detr": "RT-DETR-L",
}


def load_model(model_config):
    model_type = model_config["type"]
    params = dict(model_config.get("params", {}))
    # Parameter translation for compatibility
    if "conf_thr" in params:
        params["conf_threshold"] = params.pop("conf_thr")
    if "nms_iou" in params:
        params["iou_threshold"] = params.pop("nms_iou")

    if model_type == "faster-rcnn":
        return FasterRCNNModel(model_config["path"], params)
    elif model_type == "rfdetr":
        return RFDETRModel(model_config["path"], params)
    elif model_type in ["yolo", "rtdetr"]:
        return UltralyticsModel(model_config["path"], params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def find_one_sample_per_bin(dataset: SkyFusionDataset, rng: np.random.RandomState | None = None) -> dict[str, int]:
    """Return dataset indices for one random sample from each density bin.

    Collect all indices per bin first, then randomly choose one per bin.
    """
    if rng is None:
        rng = np.random.RandomState()

    bin_to_indices: dict[str, list[int]] = {name: [] for name in DENSITY_BINS}

    for i, sample in enumerate(dataset):
        count = len(sample.annotations)
        for bin_name, (lo, hi) in DENSITY_BINS.items():
            if lo <= count < hi:
                bin_to_indices[bin_name].append(i)
                break

    selected: dict[str, int] = {name: None for name in DENSITY_BINS}
    for name, indices in bin_to_indices.items():
        if indices:
            selected[name] = int(indices[rng.randint(len(indices))])
        else:
            selected[name] = None

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Generate portrait density comparison grid (GT, Models) x (sparse, medium, dense)"
    )
    parser.add_argument("--dataset", default="datasets/SkyFusion_yolo", help="Path to dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "test"], help="Dataset split")
    parser.add_argument(
        "--models", default="all", help="Comma-separated model names to include, or 'all' for all configured models"
    )
    parser.add_argument(
        "--out",
        default="output/spatial-density/visualizations/density_model_comparison_grid.png",
        help="Output image path",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for drawing predictions")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for selecting examples per density bin")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_path

    # Load dataset (test split)
    dataset = SkyFusionDataset(args.dataset, split=args.split, config={})

    # RNG for reproducible shuffling of examples
    rng = np.random.RandomState(args.seed) if args.seed is not None else np.random.RandomState()

    # Select examples
    example_indices = find_one_sample_per_bin(dataset, rng=rng)
    # Order rows: sparse, medium, dense
    ordered_bins = ["sparse", "medium", "dense"]
    samples = [dataset[example_indices[b]] for b in ordered_bins if example_indices[b] is not None]

    # Select models
    cfg_by_name = {cfg["name"]: cfg for cfg in MODELS_CONFIGS}
    if args.models.strip().lower() == "all":
        requested_models = [cfg["name"] for cfg in MODELS_CONFIGS]
    else:
        requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in requested_models:
            if m not in cfg_by_name:
                raise SystemExit(f"Unknown model name '{m}'. Available: {list(cfg_by_name)}")

    # Initialize requested models
    models = [(name, load_model(cfg_by_name[name])) for name in requested_models]

    # Run predictions per sample per model
    preds_by_model: dict[str, list[list]] = {name: [] for name, _ in models}
    for sample in samples:
        for name, model in models:
            preds = model.predict(sample.image)
            preds_by_model[name].append(preds)

    # Visualizer for drawing boxes
    vis = DetectionVisualizer(
        {
            "confidence_threshold": args.conf,
            "line_thickness": 3,
            "font_size": 10,
        }
    )

    # Build portrait grid: rows = [GT] + models, cols = [sparse, medium, dense]
    row_labels = ["Ground Truth"] + [DISPLAY_NAMES.get(n, n) for n in requested_models]
    col_labels = ["Sparse", "Medium", "Dense"]

    n_rows = len(row_labels)
    n_cols = len(samples)  # should be 3

    # Larger per-cell size for print; portrait aspect
    fig_w = 3.4 * n_cols
    fig_h = 3.8 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = np.array([axes])

    # Draw cells
    for c, sample in enumerate(samples):
        # Prepare image
        image = sample.image if sample.image is not None else cv2.imread(sample.image_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Top row: Ground Truth
        ax = axes[0, c]
        ax.imshow(image_rgb)
        for ann in sample.annotations:
            vis._draw_bbox_simple(ax, ann.bbox, ann.class_name, color=vis.colors.get("ground_truth", (255, 255, 0)))
        ax.set_title(col_labels[c], fontsize=16, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("Ground Truth", rotation=90, fontsize=14, fontweight="bold")

        # Next rows: each requested model
        for r, (model_name, _) in enumerate(models, start=1):
            ax = axes[r, c]
            ax.imshow(image_rgb)
            # Filter predictions by confidence
            for det in preds_by_model[model_name][c]:
                if det.confidence >= args.conf:
                    vis._draw_bbox_simple(
                        ax, det.bbox, det.class_name, color=vis.colors.get(det.class_name, (255, 0, 255))
                    )
            if c == 0:
                # Put row label at the left of the row (display name)
                label = DISPLAY_NAMES.get(model_name, model_name)
                ax.set_ylabel(label, rotation=90, fontsize=14, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved density model comparison grid (portrait) to: {grid_path}")


if __name__ == "__main__":
    main()
