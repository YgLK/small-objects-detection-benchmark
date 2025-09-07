"""
Script to generate comparison plots for model performance across different object sizes.

This script reads JSON files containing size-bin metrics for different models
and generates plots comparing their performance.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics_from_json(json_path):
    """Load metrics from a JSON file."""
    with open(json_path) as f:
        return json.load(f)


def collect_model_metrics(size_bin_dir):
    """Collect metrics from all model JSON files in the size-bin directory."""
    model_metrics = {}

    # Get all JSON files in the directory
    json_files = list(Path(size_bin_dir).glob("*_size_bin_metrics.json"))

    for json_file in json_files:
        # Extract model name from filename
        model_name = json_file.name.replace("_size_bin_metrics.json", "")

        # Load metrics
        metrics = load_metrics_from_json(json_file)
        model_metrics[model_name] = metrics

    return model_metrics


def create_map_comparison_plot(model_metrics, output_path):
    """Create a comparison plot for mAP@0.5 across size bins."""
    # Prepare data for plotting
    size_bins = ["tiny", "small", "medium"]
    models = list(model_metrics.keys())

    # Create data structure for plotting
    plot_data = {}
    for model_name, metrics in model_metrics.items():
        plot_data[model_name] = [metrics.get(bin_name, {}).get("mAP@0.5", 0) for bin_name in size_bins]

    # Create DataFrame
    df = pd.DataFrame(plot_data, index=size_bins)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create bar plot
    x = np.arange(len(size_bins))
    width = 0.8 / len(models)

    for i, (model_name, values) in enumerate(plot_data.items()):
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name)

    # Formatting
    ax.set_xlabel("Size Bins")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Model Performance Comparison by Object Size")
    ax.set_xticks(x)
    ax.set_xticklabels(size_bins)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"mAP@0.5 comparison plot saved to {output_path}")


def create_recall_comparison_plot(model_metrics, output_path):
    """Create a comparison plot for recall across size bins."""
    # Prepare data for plotting
    size_bins = ["tiny", "small", "medium"]
    models = list(model_metrics.keys())

    # Create data structure for plotting
    plot_data = {}
    for model_name, metrics in model_metrics.items():
        plot_data[model_name] = [metrics.get(bin_name, {}).get("recall", 0) for bin_name in size_bins]

    # Create DataFrame
    df = pd.DataFrame(plot_data, index=size_bins)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create bar plot
    x = np.arange(len(size_bins))
    width = 0.8 / len(models)

    for i, (model_name, values) in enumerate(plot_data.items()):
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name)

    # Formatting
    ax.set_xlabel("Size Bins")
    ax.set_ylabel("Recall")
    ax.set_title("Model Recall Comparison by Object Size")
    ax.set_xticks(x)
    ax.set_xticklabels(size_bins)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Recall comparison plot saved to {output_path}")


def create_per_class_map_plot(model_metrics, output_path):
    """Create a comparison plot for per-class mAP@0.5 across size bins."""
    # Define class names
    class_names = ["aircraft", "ship", "vehicle"]
    size_bins = ["tiny", "small", "medium"]
    models = list(model_metrics.keys())

    # Create subplots
    fig, axes = plt.subplots(len(class_names), 1, figsize=(12, 3 * len(class_names)))
    if len(class_names) == 1:
        axes = [axes]

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]

        # Prepare data for this class
        plot_data = {}
        for model_name, metrics in model_metrics.items():
            plot_data[model_name] = [metrics.get(bin_name, {}).get(f"AP@0.5_{class_name}", 0) for bin_name in size_bins]

        # Create bar plot
        x = np.arange(len(size_bins))
        width = 0.8 / len(models)

        for i, (model_name, values) in enumerate(plot_data.items()):
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_name)

        # Formatting
        ax.set_ylabel("AP@0.5")
        ax.set_title(f"{class_name.capitalize()} AP@0.5 by Object Size")
        ax.set_xticks(x)
        ax.set_xticklabels(size_bins)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Add legend only to the first subplot
        if class_idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Per-class AP@0.5 comparison plot saved to {output_path}")


def main():
    """Main function to generate all comparison plots."""
    # Define paths
    size_bin_dir = "TO_PROCESS/04_size_bin_eval/size-bin"
    output_dir = "output/size_bin_comparison"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading metrics from {size_bin_dir}...")

    # Collect metrics from all models
    model_metrics = collect_model_metrics(size_bin_dir)

    if not model_metrics:
        print("Warning: No model metrics found.")
        return

    print(f"Loaded metrics for {len(model_metrics)} models.")

    # Generate plots
    print("\nGenerating plots...")

    # mAP@0.5 comparison
    map_plot_path = os.path.join(output_dir, "map_50_comparison.png")
    create_map_comparison_plot(model_metrics, map_plot_path)

    # Recall comparison
    recall_plot_path = os.path.join(output_dir, "recall_comparison.png")
    create_recall_comparison_plot(model_metrics, recall_plot_path)

    # Per-class AP@0.5 comparison
    per_class_plot_path = os.path.join(output_dir, "per_class_ap_50_comparison.png")
    create_per_class_map_plot(model_metrics, per_class_plot_path)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
