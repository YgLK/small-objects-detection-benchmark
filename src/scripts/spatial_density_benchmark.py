"""
Benchmark script for spatial density analysis.

This script evaluates object detection models on different spatial density levels:
- Sparse: < 10 objects per 640×640 patch
- Medium: 10-30 objects per 640×640 patch
- Dense: > 30 objects per 640×640 patch

Outputs:
- Plots saved in the ./output/spatial-density/ dir
- Metrics for each model saved in json in the same dir
- Example of inference saved as images in the same dir for each density level
"""

import argparse
import json
import os


MODELS_CONFIGS = [
    {
        "name": "yolov8m-aug-update_20250603",
        "path": "/home/yglk/coding/dpm3-skyfusion_v3/dpm3/models/yolov8m-aug-update_20250603.pt",
        "params": {
            "conf_thr": 0.06291186979342091,
            "nms_iou": 0.3358295483691517,
        },
        "type": "yolo",
    },
    {
        "name": "yolov11m-p2-aug_20250603",
        "path": "/home/yglk/coding/dpm3-skyfusion_v3/dpm3/models/yolov11m-p2-aug_20250603.pt",
        "params": {
            "conf_thr": 0.052566007120515956,
            "nms_iou": 0.49317179138811856,
        },
        "type": "yolo",
    },
    {
        "name": "rf-detr",
        "path": "/home/yglk/coding/dpm3-skyfusion_v3/dpm3/models/rfdetr_best_total.pth",
        "params": {
            "conf_thr": 0.09616820140192325,
        },
        "type": "rfdetr",
    },
    {
        "name": "faster-rcnn",
        "path": "/home/yglk/coding/dpm3-skyfusion_v3/dpm3/models/fasterrcnn-best-epoch=18-val_map=0.31.ckpt",
        "params": {
            "conf_thr": 0.07957236023833904,
            "nms_iou": 0.621230971215935,
        },
        "type": "faster-rcnn",
    },
    {
        "name": "rt-detr",
        "path": "/home/yglk/coding/dpm3-skyfusion_v3/dpm3/models/rtdetr-aug_best.pt",
        "params": {
            "conf_thr": 0.2704984199324548,
        },
        "type": "rtdetr",
    },
]

from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.odc.benchmark.datasets import SkyFusionDataset  # Updated import path
from src.odc.benchmark.metrics.detection_metrics import DetectionMetrics
from src.odc.benchmark.models import (  # Updated import path
    FasterRCNNModel,
    RFDETRModel,
    UltralyticsModel,
)


DENSITY_BINS = {
    "sparse": (0, 10),
    "medium": (10, 30),
    "dense": (30, float("inf")),
}


def load_model(model_config):
    """Loads a model based on its configuration."""
    model_type = model_config["type"]
    # Parameter translation for compatibility
    params = model_config.get("params", {})
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


def calculate_density_bins(all_detections, all_ground_truths):
    """Calculates spatial density bins for evaluation based on object count per 640x640 patch."""
    binned_gts = {name: [[] for _ in all_ground_truths] for name in DENSITY_BINS}
    bin_counts = {name: 0 for name in DENSITY_BINS}  # Track how many images per bin

    for i, gt_list in enumerate(all_ground_truths):
        # Count objects in this image
        object_count = len(gt_list)

        # Determine which density bin this image belongs to
        for bin_name, (min_count, max_count) in DENSITY_BINS.items():
            if min_count <= object_count < max_count:
                binned_gts[bin_name][i] = gt_list
                bin_counts[bin_name] += 1
                break

    return binned_gts, bin_counts


def main():
    parser = argparse.ArgumentParser(description="Spatial Density Performance Analysis Benchmark")
    parser.add_argument(
        "--output_dir", type=str, default="./output/spatial-density", help="Directory to save the results."
    )
    parser.add_argument("--dataset_path", type=str, default="datasets/SkyFusion_yolo", help="Path to the dataset.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Running spatial density benchmark. Results will be saved in {args.output_dir}")

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    dataset = SkyFusionDataset(args.dataset_path, split="test", config={})
    print(f"   Loaded {len(dataset)} samples")

    # dataset = dataset[5] # DEBUG - Removed as it was causing AttributeError

    metrics_calculator = DetectionMetrics(class_names=dataset.get_class_names())

    all_results = {}

    for model_config in MODELS_CONFIGS:
        model_name = model_config["name"]
        print(f"\nLoading model: {model_name}...")
        model = load_model(model_config)
        print(f"   Model loaded.")

        print("\nRunning inference...")
        all_detections = []
        all_ground_truths = []
        for sample in tqdm(dataset, desc=f"Inference {model_name}"):
            preds = model.predict(sample.image)
            all_detections.append(preds)
            all_ground_truths.append(sample.annotations)

        print("\nPerforming spatial density analysis...")
        binned_gts, bin_counts = calculate_density_bins(all_detections, all_ground_truths)

        model_results = {}
        for bin_name in DENSITY_BINS:
            print(f"   - Calculating metrics for '{bin_name}' bin...")
            gts_for_bin = binned_gts[bin_name]

            # Check if there are any ground truths in this bin
            if bin_counts[bin_name] == 0:
                print(f"     Warning: No images found for bin '{bin_name}'. Skipping.")
                continue

            metrics = metrics_calculator.calculate_comprehensive_metrics(all_detections, gts_for_bin)
            metrics["image_count"] = bin_counts[bin_name]  # Add image count per bin
            model_results[bin_name] = metrics

        all_results[model_name] = model_results

        # Save results to JSON
        results_path = os.path.join(args.output_dir, f"{model_name}_spatial_density_metrics.json")
        with open(results_path, "w") as f:
            json.dump(model_results, f, indent=4)
        print(f"   Metrics for {model_name} saved to {results_path}")

    print("\nGenerating plots...")
    generate_plots(all_results, args.output_dir)

    print("\nGenerating visualizations...")
    generate_visualizations(all_results, dataset, args.output_dir)


def generate_visualizations(all_results, dataset, output_dir):
    """Generates and saves sample images with predictions for each density bin."""
    try:
        import cv2

        from src.odc.benchmark.visualizers import DetectionVisualizer
    except ImportError:
        print("Warning: OpenCV or DetectionVisualizer not found. Skipping visualization.")
        print("   Please run: pip install opencv-python-headless")
        return

    visualizer = DetectionVisualizer(class_names=dataset.get_class_names())
    images_to_visualize = find_example_images_for_density_bins(dataset)

    for bin_name, sample_idx in images_to_visualize.items():
        if sample_idx is None:
            print(f"   - No image found for '{bin_name}' bin.")
            continue

        sample = dataset[sample_idx]
        image = sample.image.copy()

        for model_name in all_results.keys():
            print(f"   - Visualizing {model_name} on '{bin_name}' example...")
            model = load_model(next(m for m in MODELS_CONFIGS if m["name"] == model_name))
            predictions = model.predict(image)

            # Draw predictions and ground truths
            vis_image = visualizer.draw_detections(image, predictions, sample.annotations)

            # Save the image
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            img_path = os.path.join(vis_dir, f"{model_name}_{bin_name}_example.png")
            cv2.imwrite(img_path, vis_image)
            print(f"     Visualization saved to {img_path}")


def find_example_images_for_density_bins(dataset):
    """Finds one sample image index for each density bin."""
    images_found = {name: None for name in DENSITY_BINS}
    bins_to_find = set(DENSITY_BINS.keys())

    for i, sample in enumerate(dataset):
        if not bins_to_find:
            break

        # Count objects in this image
        object_count = len(sample.annotations)

        for bin_name, (min_count, max_count) in DENSITY_BINS.items():
            if bin_name in bins_to_find and min_count <= object_count < max_count:
                images_found[bin_name] = i
                bins_to_find.remove(bin_name)
                break

    return images_found


def generate_plots(all_results, output_dir):
    """Generates and saves plots comparing model performance across density bins."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Warning: Matplotlib or Pandas not installed. Skipping plot generation.")
        print("   Please run: pip install matplotlib pandas")
        return

    # --- mAP@0.5 Comparison Plot ---
    map_data = {}
    for model_name, results in all_results.items():
        map_data[model_name] = {bin_name: data.get("mAP@0.5", 0) for bin_name, data in results.items()}

    df = pd.DataFrame(map_data)
    df.plot(kind="bar", figsize=(12, 7), rot=0)

    plt.title("mAP@0.5 Performance by Spatial Density")
    plt.ylabel("mAP@0.5")
    plt.xlabel("Density Bins")
    plt.legend(title="Models")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "map_50_by_density_comparison.png")
    plt.savefig(plot_path)
    print(f"   Plot saved to {plot_path}")
    plt.close()

    # --- Recall Comparison Plot ---
    recall_data = {}
    for model_name, results in all_results.items():
        recall_data[model_name] = {bin_name: data.get("mAR@0.5", 0) for bin_name, data in results.items()}

    df_recall = pd.DataFrame(recall_data)
    df_recall.plot(kind="bar", figsize=(12, 7), rot=0)

    plt.title("Recall (mAR@0.5) Performance by Spatial Density")
    plt.ylabel("Recall (mAR@0.5)")
    plt.xlabel("Density Bins")
    plt.legend(title="Models")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()

    recall_plot_path = os.path.join(output_dir, "recall_50_by_density_comparison.png")
    plt.savefig(recall_plot_path)
    print(f"   Recall plot saved to {recall_plot_path}")
    plt.close()

    print("\nSpatial density benchmark finished.")


if __name__ == "__main__":
    main()
