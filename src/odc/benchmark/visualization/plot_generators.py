"""Plot generators for benchmark visualization."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..reporters.base_reporter import BenchmarkResults


class PlotGenerator:
    """Generate various plots and visualizations for benchmark results."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the plot generator.

        Args:
            config: Configuration dictionary with plotting options:
                - style: Plot style ('seaborn', 'ggplot', 'classic')
                - dpi: Plot resolution (default: 300)
                - figsize: Default figure size (default: (10, 6))
                - color_palette: Color palette for plots
                - save_format: Format for saved plots ('png', 'pdf', 'svg')
        """
        self.config = config
        self.style = config.get("style", "seaborn")
        self.dpi = config.get("dpi", 300)
        self.figsize = config.get("figsize", (10, 6))
        self.save_format = config.get("save_format", "png")

        # Set up plotting style
        self._setup_style()

        # Color palette for models
        self.colors = config.get(
            "color_palette",
            [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
                "#aec7e8",
                "#ffbb78",
                "#98df8a",
                "#ff9896",
                "#c5b0d5",
                "#c49c94",
            ],
        )

    def _setup_style(self):
        """Set up matplotlib style."""
        if self.style == "seaborn":
            plt.style.use("seaborn-v0_8")
            sns.set_palette("husl")
        elif self.style == "ggplot":
            plt.style.use("ggplot")
        else:
            plt.style.use("classic")

        # Set default parameters
        plt.rcParams["figure.dpi"] = self.dpi
        plt.rcParams["savefig.dpi"] = self.dpi
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 11

    def generate_all_plots(self, results: BenchmarkResults, output_dir: str) -> dict[str, str]:
        """Generate all standard plots for the benchmark results.

        Args:
            results: BenchmarkResults object
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = {}

        # Performance comparison plots
        plot_paths["performance_comparison"] = self.plot_performance_comparison(
            results, os.path.join(output_dir, f"performance_comparison.{self.save_format}")
        )

        plot_paths["map_comparison"] = self.plot_map_comparison(
            results, os.path.join(output_dir, f"map_comparison.{self.save_format}")
        )

        plot_paths["class_wise_performance"] = self.plot_class_wise_performance(
            results, os.path.join(output_dir, f"class_wise_performance.{self.save_format}")
        )

        plot_paths["speed_accuracy_tradeoff"] = self.plot_speed_accuracy_tradeoff(
            results, os.path.join(output_dir, f"speed_accuracy_tradeoff.{self.save_format}")
        )

        plot_paths["model_complexity"] = self.plot_model_complexity(
            results, os.path.join(output_dir, f"model_complexity.{self.save_format}")
        )

        plot_paths["radar_chart"] = self.plot_radar_chart(
            results, os.path.join(output_dir, f"radar_chart.{self.save_format}")
        )

        return plot_paths

    def plot_performance_comparison(self, results: BenchmarkResults, output_path: str) -> str:
        """Generate performance comparison bar chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

        # Extract data
        model_names = [result["model_name"] for result in results.model_results]
        map_50 = [result["detection_metrics"]["mAP@0.5"] for result in results.model_results]
        map_75 = [result["detection_metrics"]["mAP@0.75"] for result in results.model_results]
        fps = [result["performance_metrics"]["fps"] for result in results.model_results]
        inference_time = [result["performance_metrics"]["inference_time_ms"] for result in results.model_results]

        # mAP@0.5 comparison
        bars1 = ax1.bar(model_names, map_50, color=self.colors[: len(model_names)])
        ax1.set_title("mAP@0.5 Comparison")
        ax1.set_ylabel("mAP@0.5")
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars1, map_50):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom"
            )

        # mAP@0.75 comparison
        bars2 = ax2.bar(model_names, map_75, color=self.colors[: len(model_names)])
        ax2.set_title("mAP@0.75 Comparison")
        ax2.set_ylabel("mAP@0.75")
        ax2.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars2, map_75):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom"
            )

        # FPS comparison
        bars3 = ax3.bar(model_names, fps, color=self.colors[: len(model_names)])
        ax3.set_title("Inference Speed (FPS)")
        ax3.set_ylabel("FPS")
        ax3.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars3, fps):
            ax3.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.1f}", ha="center", va="bottom"
            )

        # Inference time comparison
        bars4 = ax4.bar(model_names, inference_time, color=self.colors[: len(model_names)])
        ax4.set_title("Inference Time")
        ax4.set_ylabel("Time (ms)")
        ax4.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars4, inference_time):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_map_comparison(self, results: BenchmarkResults, output_path: str) -> str:
        """Generate mAP comparison across different IoU thresholds."""
        fig, ax = plt.subplots(figsize=self.figsize)

        model_names = [result["model_name"] for result in results.model_results]
        map_50 = [result["detection_metrics"]["mAP@0.5"] for result in results.model_results]
        map_75 = [result["detection_metrics"]["mAP@0.75"] for result in results.model_results]
        map_coco = [result["detection_metrics"]["mAP@[0.5:0.05:0.95]"] for result in results.model_results]

        x = np.arange(len(model_names))
        width = 0.25

        bars1 = ax.bar(x - width, map_50, width, label="mAP@0.5", color=self.colors[0])
        bars2 = ax.bar(x, map_75, width, label="mAP@0.75", color=self.colors[1])
        bars3 = ax.bar(x + width, map_coco, width, label="mAP@COCO", color=self.colors[2])

        ax.set_xlabel("Models")
        ax.set_ylabel("mAP Score")
        ax.set_title("mAP Comparison Across IoU Thresholds")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_class_wise_performance(self, results: BenchmarkResults, output_path: str) -> str:
        """Generate class-wise performance heatmap."""
        # Prepare data for heatmap
        model_names = [result["model_name"] for result in results.model_results]
        class_names = results.dataset_info["class_names"]

        # Create matrix of AP scores
        ap_matrix = []
        for result in results.model_results:
            row = []
            for class_name in class_names:
                ap_key = f"AP@0.5_{class_name}"
                row.append(result["detection_metrics"].get(ap_key, 0))
            ap_matrix.append(row)

        ap_matrix = np.array(ap_matrix)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(ap_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels([name.title() for name in class_names])
        ax.set_yticklabels(model_names)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Average Precision (AP@0.5)", rotation=-90, va="bottom")

        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(class_names)):
                text = ax.text(
                    j, i, f"{ap_matrix[i, j]:.3f}", ha="center", va="center", color="black", fontweight="bold"
                )

        ax.set_title("Class-wise Average Precision (AP@0.5)")
        ax.set_xlabel("Object Classes")
        ax.set_ylabel("Models")

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_speed_accuracy_tradeoff(self, results: BenchmarkResults, output_path: str) -> str:
        """Generate speed vs accuracy scatter plot."""
        fig, ax = plt.subplots(figsize=self.figsize)

        model_names = [result["model_name"] for result in results.model_results]
        map_50 = [result["detection_metrics"]["mAP@0.5"] for result in results.model_results]
        fps = [result["performance_metrics"]["fps"] for result in results.model_results]

        # Create scatter plot
        scatter = ax.scatter(
            fps, map_50, c=self.colors[: len(model_names)], s=100, alpha=0.7, edgecolors="black", linewidth=1
        )

        # Add model labels
        for i, name in enumerate(model_names):
            ax.annotate(name, (fps[i], map_50[i]), xytext=(5, 5), textcoords="offset points", fontsize=10, ha="left")

        ax.set_xlabel("Inference Speed (FPS)")
        ax.set_ylabel("mAP@0.5")
        ax.set_title("Speed vs Accuracy Trade-off")
        ax.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(fps, map_50, 1)
        p = np.poly1d(z)
        ax.plot(fps, p(fps), "r--", alpha=0.8, linewidth=2, label="Trend")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_model_complexity(self, results: BenchmarkResults, output_path: str) -> str:
        """Generate model complexity comparison (parameters vs performance)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        model_names = [result["model_name"] for result in results.model_results]
        map_50 = [result["detection_metrics"]["mAP@0.5"] for result in results.model_results]
        parameters = [
            result["performance_metrics"]["parameters"] / 1e6 for result in results.model_results
        ]  # Convert to millions

        # Create bubble chart (size represents model size in MB)
        model_sizes = [result["performance_metrics"]["model_size_mb"] for result in results.model_results]

        scatter = ax.scatter(
            parameters,
            map_50,
            s=[size * 10 for size in model_sizes],
            c=self.colors[: len(model_names)],
            alpha=0.6,
            edgecolors="black",
            linewidth=1,
        )

        # Add model labels
        for i, name in enumerate(model_names):
            ax.annotate(
                name, (parameters[i], map_50[i]), xytext=(5, 5), textcoords="offset points", fontsize=10, ha="left"
            )

        ax.set_xlabel("Parameters (Millions)")
        ax.set_ylabel("mAP@0.5")
        ax.set_title("Model Complexity vs Performance\n(Bubble size represents model size in MB)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_radar_chart(self, results: BenchmarkResults, output_path: str) -> str:
        """Generate radar chart comparing models across multiple metrics."""
        # Prepare data
        model_names = [result["model_name"] for result in results.model_results]

        # Normalize metrics to 0-1 scale for radar chart
        metrics = {
            "mAP@0.5": [result["detection_metrics"]["mAP@0.5"] for result in results.model_results],
            "mAP@0.75": [result["detection_metrics"]["mAP@0.75"] for result in results.model_results],
            "Speed (FPS)": [result["performance_metrics"]["fps"] for result in results.model_results],
            "Efficiency": [
                1 / result["performance_metrics"]["inference_time_ms"] * 100 for result in results.model_results
            ],
            "Memory Eff.": [
                1 / max(1, result["performance_metrics"]["memory_usage_mb"]) * 100 for result in results.model_results
            ],
        }

        # Normalize each metric to 0-1
        for metric_name, values in metrics.items():
            max_val = max(values)
            min_val = min(values)
            if max_val > min_val:
                metrics[metric_name] = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                metrics[metric_name] = [1.0] * len(values)

        # Set up radar chart
        categories = list(metrics.keys())
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Plot each model
        for i, model_name in enumerate(model_names):
            values = [metrics[cat][i] for cat in categories]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.25, color=self.colors[i % len(self.colors)])

        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.grid(True)

        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        plt.title("Model Performance Radar Chart\n(Normalized Metrics)", size=14, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_confusion_matrix(self, results: BenchmarkResults, model_name: str, output_path: str) -> str:
        """Generate confusion matrix for a specific model (placeholder for future implementation)."""
        # This would require ground truth and prediction data
        # For now, create a placeholder
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"Confusion Matrix for {model_name}\n(Implementation pending)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(f"Confusion Matrix - {model_name}")

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path
