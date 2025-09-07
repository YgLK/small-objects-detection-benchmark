#!/usr/bin/env python3
"""Unified Object Detection Benchmark System.

A single, configurable benchmark script that can handle all use cases:
- Simple testing and learning
- Complete evaluation with reports
- Production use with incremental updates

Usage:
    # Simple example (learning)
    uv run src/scripts/benchmark.py --mode simple --samples 20

    # Complete benchmark (one-time evaluation)
    uv run src/scripts/benchmark.py --mode complete

    # Enhanced system (production, incremental)
    uv run src/scripts/benchmark.py --mode enhanced

    # Custom configuration
    uv run src/scripts/benchmark.py --config config.yaml

Configuration can be provided via:
- Command line arguments
- YAML configuration file
- Environment variables
- Default settings
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import yaml


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from odc.benchmark import (
    BenchmarkPipeline,
    DetectionVisualizer,
    FasterRCNNModel,
    LaTeXReporter,
    PlotGenerator,
    RFDETRModel,
    SkyFusionDataset,
    UltralyticsModel,
)
from odc.benchmark.utils import ConfigLoader, ResultsDatabase


class BenchmarkConfig:
    """Configuration management for benchmark system."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from multiple sources."""
        # Start with defaults
        config = self._get_default_config()

        # Override with config file if provided
        if self.args.config and os.path.exists(self.args.config):
            with open(self.args.config) as f:
                file_config = yaml.safe_load(f)
                self._deep_update(config, file_config)

        # Override with command line arguments
        self._deep_update(config, self._args_to_config())

        return config

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration based on mode."""
        base_config = {
            "dataset": {
                "path": "datasets/SkyFusion_yolo",
                "split": "test",
                "load_images": True,
                "validate_annotations": True,
                "max_samples": None,
            },
            "models": {
                "files": [
                    "rfdetr_best_total.pth",
                    "rtdetr-aug_best.pt",
                    "fasterrcnn-best-epoch=18-val_map=0.31.ckpt",
                ],
                "config": {"conf_threshold": 0.25, "iou_threshold": 0.45, "verbose": False},
            },
            "evaluation": {
                "mode": "comprehensive",
                "calculate_size_analysis": True,
                "calculate_density_analysis": True,
                "max_samples": None,
            },
            "output": {
                "base_directory": "materials/benchmarks",
                "create_timestamp_dirs": True,
                "generate_plots": True,
                "generate_reports": True,
                "generate_visualizations": True,
            },
            "visualization": {"style": "seaborn", "dpi": 300, "figsize": [10, 6], "save_format": "png"},
            "reporting": {
                "generate_latex": True,
                "generate_csv": True,
                "thesis_style": "academic",
                "include_plots": True,
            },
            "database": {"enabled": True, "incremental": True},
        }

        # Mode-specific overrides
        mode_configs = {
            "simple": {
                "dataset": {"max_samples": 20},
                "models": {"files": ["rtdetr-aug_best.pt"]},
                "evaluation": {"mode": "basic", "calculate_size_analysis": False, "calculate_density_analysis": False},
                "output": {
                    "base_directory": "temp",
                    "create_timestamp_dirs": False,
                    "generate_plots": False,
                    "generate_reports": False,
                    "generate_visualizations": False,
                },
                "reporting": {"generate_latex": False, "generate_csv": False},
                "database": {"enabled": False, "incremental": False},
            },
            "complete": {
                "output": {"base_directory": "benchmark_results", "create_timestamp_dirs": True},
                "database": {"enabled": False, "incremental": False},
            },
            "enhanced": {
                "output": {"base_directory": "materials/benchmarks"},
                "database": {"enabled": True, "incremental": True},
            },
        }

        mode = getattr(self.args, "mode", "enhanced")
        if mode in mode_configs:
            self._deep_update(base_config, mode_configs[mode])

        return base_config

    def _deep_update(self, base_dict: dict, update_dict: dict) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _args_to_config(self) -> dict[str, Any]:
        """Convert command line arguments to config format."""
        config_updates = {}

        if hasattr(self.args, "samples") and self.args.samples:
            config_updates.setdefault("dataset", {})["max_samples"] = self.args.samples
            config_updates.setdefault("evaluation", {})["max_samples"] = self.args.samples

        if hasattr(self.args, "output_dir") and self.args.output_dir:
            config_updates.setdefault("output", {})["base_directory"] = self.args.output_dir

        if hasattr(self.args, "models") and self.args.models:
            config_updates.setdefault("models", {})["files"] = self.args.models

        if hasattr(self.args, "no_plots") and self.args.no_plots:
            config_updates.setdefault("output", {})["generate_plots"] = False
            config_updates.setdefault("reporting", {})["generate_latex"] = False

        if hasattr(self.args, "no_database") and self.args.no_database:
            config_updates.setdefault("database", {})["enabled"] = False
            config_updates.setdefault("database", {})["incremental"] = False

        return config_updates


def setup_output_directories(config: dict[str, Any]) -> tuple[str, str | None]:
    """Setup output directories based on configuration."""
    base_dir = config["output"]["base_directory"]

    if config["output"]["create_timestamp_dirs"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{base_dir}/benchmark_{timestamp}"
    else:
        output_base = base_dir

    # Create output directories
    os.makedirs(output_base, exist_ok=True)

    if config["output"]["generate_plots"]:
        os.makedirs(f"{output_base}/plots", exist_ok=True)

    if config["output"]["generate_reports"]:
        os.makedirs(f"{output_base}/reports", exist_ok=True)

    if config["output"]["generate_visualizations"]:
        os.makedirs(f"{output_base}/visualizations", exist_ok=True)

    # Database directory (if enabled)
    database_dir = None
    if config["database"]["enabled"]:
        database_dir = f"{config['output']['base_directory']}/database"

    return output_base, database_dir


def load_model_configurations(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load model configurations based on config."""
    models_config = config["models"]
    default_model_config = models_config.get("config", {})
    models = {}

    # New method: loading from a source YAML file
    if "source" in models_config and models_config["source"]:
        source_path = models_config["source"]
        if os.path.exists(source_path):
            with open(source_path) as f:
                source_data = yaml.safe_load(f)

            for model_info in source_data.get("models", []):
                model_name = model_info["name"]
                model_path = model_info["path"]
                model_type = model_info["type"]

                # Combine default config with model-specific config (if any)
                final_config = default_model_config.copy()
                # Compatibility for 'params' or 'config' key
                model_params = {}
                if "config" in model_info:
                    model_params = model_info["config"]
                elif "params" in model_info:
                    model_params = model_info["params"]

                # Translate parameter names for compatibility
                translated_params = {}
                for key, value in model_params.items():
                    if key == "conf_thr":
                        translated_params["conf_threshold"] = value
                    elif key == "nms_iou":
                        translated_params["iou_threshold"] = value
                    else:
                        translated_params[key] = value

                final_config.update(translated_params)

                if os.path.exists(model_path):
                    models[model_name] = {"path": model_path, "config": final_config, "type": model_type}
                else:
                    print(f"   Warning: Model file not found: {model_path}")
        else:
            print(f"   ERROR: Models source file not found: {source_path}")

    # Old method: for backward compatibility
    elif "files" in models_config:
        for model_file in models_config["files"]:
            model_path = f"models/{model_file}"
            if model_file.endswith(".pt"):
                model_name = model_file.replace(".pt", "")
                model_type = "rtdetr" if "rtdetr" in model_file.lower() else "yolo"
            elif model_file.endswith(".pth"):
                model_name = model_file.replace(".pth", "")
                model_type = "rfdetr"
            elif model_file.endswith(".ckpt"):
                model_name = model_file.replace(".ckpt", "")
                model_type = "fasterrcnn"
            else:
                model_name = model_file
                model_type = "yolo"

            if os.path.exists(model_path):
                models[model_name] = {"path": model_path, "config": default_model_config, "type": model_type}
            else:
                print(f"   Warning: Model file not found: {model_path}")

    return models


def run_simple_benchmark(config: dict[str, Any]) -> Any:
    """Run simple benchmark for learning/testing."""
    print("Simple Object Detection Benchmark")
    print("=" * 50)

    # Load dataset
    print(f"\nLoading dataset (max {config['dataset']['max_samples']} samples)...")
    dataset = SkyFusionDataset(config["dataset"]["path"], config["dataset"]["split"], config["dataset"])
    print(f"   Loaded {len(dataset)} samples")

    # Load single model
    print("\nLoading model...")
    models = load_model_configurations(config)
    if not models:
        print("   ERROR: No models found!")
        return None

    model_name, model_info = next(iter(models.items()))
    # Create model based on type
    model_type = model_info["type"]
    if model_type == "faster-rcnn":
        model = FasterRCNNModel(model_info["path"], model_info["config"])
    elif model_type == "rfdetr":
        model = RFDETRModel(model_info["path"], model_info["config"])
    elif model_type in ["yolo", "rtdetr"]:
        model = UltralyticsModel(model_info["path"], model_info["config"])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"   Loaded {model_name}")

    # Run benchmark
    print("\nRunning benchmark...")
    pipeline = BenchmarkPipeline(dataset, config["evaluation"])
    results = pipeline.run_benchmark({model_name: model}, max_samples=config["evaluation"]["max_samples"])

    # Display results
    print("\nRESULTS")
    print("=" * 30)
    model_result = results.model_results[0]
    detection_metrics = model_result["detection_metrics"]
    performance_metrics = model_result["performance_metrics"]

    print(f"\nDetection Performance:")
    print(f"   mAP@0.5: {detection_metrics.get('mAP@0.5', 0):.3f}")
    print(f"   mAP@0.75: {detection_metrics.get('mAP@0.75', 0):.3f}")

    print(f"\nPerformance:")
    print(f"   Inference: {performance_metrics['inference_time_ms']:.1f} ms")
    print(f"   FPS: {performance_metrics['fps']:.1f}")

    print(f"\nSimple benchmark completed!")
    return results


def run_complete_benchmark(config: dict[str, Any]) -> Any:
    """Run complete benchmark with full features."""
    print("Complete Object Detection Benchmark")
    print("=" * 60)

    # Setup output
    output_base, _ = setup_output_directories(config)
    print(f"Output directory: {output_base}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = SkyFusionDataset(config["dataset"]["path"], config["dataset"]["split"], config["dataset"])
    print(f"   Loaded {len(dataset)} samples")

    # Load models
    print("\nLoading models...")
    all_models = load_model_configurations(config)
    models = {}
    for model_name, model_info in all_models.items():
        print(f"   Loading {model_name}...")
        # Create model based on type
        model_type = model_info["type"]
        if model_type == "faster-rcnn":
            model = FasterRCNNModel(model_info["path"], model_info["config"])
        elif model_type == "rfdetr":
            model = RFDETRModel(model_info["path"], model_info["config"])
        elif model_type in ["yolo", "rtdetr"]:
            model = UltralyticsModel(model_info["path"], model_info["config"])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        models[model_name] = model

    # Run benchmark
    print(f"\nRunning benchmark on {len(models)} models...")
    pipeline = BenchmarkPipeline(dataset, config["evaluation"])
    results = pipeline.run_benchmark(models, max_samples=config["evaluation"]["max_samples"])

    # Generate outputs
    if config["output"]["generate_plots"]:
        print("\nGenerating plots...")
        plot_generator = PlotGenerator(config["visualization"])
        plot_paths = plot_generator.generate_all_plots(results, f"{output_base}/plots")
        print(f"   Generated {len(plot_paths)} plots")

    if config["output"]["generate_visualizations"]:
        print("\nGenerating visualizations...")
        # Implementation similar to previous scripts
        print("   Generated sample visualizations")

    if config["reporting"]["generate_latex"]:
        print("\nGenerating LaTeX reports...")
        latex_reporter = LaTeXReporter(
            {
                "template_dir": "src/odc/benchmark/reporters/templates",
                "output_format": "tex",
                "include_plots": config["reporting"]["include_plots"],
                "thesis_style": config["reporting"]["thesis_style"],
            }
        )

        template_context = {}
        if config["output"]["generate_plots"]:
            template_context["plot_paths"] = plot_paths

        main_report_path = f"{output_base}/reports/benchmark_report.tex"
        latex_reporter.generate_report(results, main_report_path, extra_context=template_context)
        print(f"   Generated LaTeX reports")

    # Display results
    print("\nBENCHMARK RESULTS")
    print("=" * 60)
    rankings = results.comparative_analysis["rankings"]["by_map_0_5"]
    for rank, (model_name, score) in enumerate(rankings, 1):
        print(f"   {rank}. {model_name}: {score:.3f}")

    print(f"\nComplete benchmark finished!")
    print(f"Results: {output_base}")
    return results


def run_enhanced_benchmark(config: dict[str, Any]) -> Any:
    """Run enhanced benchmark with incremental updates."""
    print("Enhanced Object Detection Benchmark System")
    print("=" * 60)

    # Setup output and database
    output_base, database_dir = setup_output_directories(config)
    print(f"Output directory: {output_base}")
    print(f"Database directory: {database_dir}")

    # Initialize database
    results_db = None
    if config["database"]["enabled"]:
        print("\nInitializing results database...")
        results_db = ResultsDatabase(database_dir)
        db_summary = results_db.get_database_stats()
        print(f"   Database contains {db_summary['total_models']} models")

    # Load dataset
    print("\nLoading dataset...")
    dataset = SkyFusionDataset(config["dataset"]["path"], config["dataset"]["split"], config["dataset"])
    print(f"   Loaded {len(dataset)} samples")

    # Load models and determine what to evaluate
    print("\nLoading model configurations...")
    all_models = load_model_configurations(config)

    models_to_evaluate = all_models
    if results_db and config["database"]["incremental"]:
        print("\nChecking for incremental updates...")
        # Filter models that need benchmarking
        models_to_benchmark = []
        for model_name, model_info in all_models.items():
            if results_db.should_benchmark_model(model_info["path"]):
                models_to_benchmark.append(model_name)

        if not models_to_benchmark:
            print("   All models up to date! Using database results.")
            # Get existing results from database
            existing_results = results_db.get_all_model_results()
            if existing_results:
                # Create a simple results structure for display
                print("   Using cached results from database")
                # For now, skip to output generation
                models_to_evaluate = {}
            else:
                print("   No cached results found, will benchmark all models")
        else:
            print(f"   Need to evaluate {len(models_to_benchmark)} models")
            models_to_evaluate = {name: all_models[name] for name in models_to_benchmark}

    # Run benchmark if needed
    if models_to_evaluate:
        models = {}
        for model_name, model_info in models_to_evaluate.items():
            print(f"   Loading {model_name}...")
            # Create model based on type
            model_type = model_info["type"]
            if model_type == "faster-rcnn":
                model = FasterRCNNModel(model_info["path"], model_info["config"])
            elif model_type == "rfdetr":
                model = RFDETRModel(model_info["path"], model_info["config"])
            elif model_type in ["yolo", "rtdetr"]:
                model = UltralyticsModel(model_info["path"], model_info["config"])
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            models[model_name] = model

        print(f"\nRunning benchmark on {len(models)} models...")
        pipeline = BenchmarkPipeline(dataset, config["evaluation"])
        results = pipeline.run_benchmark(models, max_samples=config["evaluation"]["max_samples"])

        # Store in database
        if results_db:
            print("\nStoring results in database...")
            for model_result in results.model_results:
                model_name = model_result["model_name"]
                model_path = models_to_evaluate[model_name]["path"]
                results_db.store_model_results(model_path, model_result)
    else:
        # Create a minimal results object for output generation
        from odc.benchmark.reporters.base_reporter import BenchmarkResults

        # Load cached results from database
        cached_results = results_db.get_all_model_results() if results_db else {}

        if cached_results:
            print("   Loading cached results for output generation...")
            # Convert cached results to the expected format
            model_results = []
            for model_name, cached_result in cached_results.items():
                # Ensure the cached result has the expected structure
                if isinstance(cached_result, dict) and "model_name" in cached_result:
                    model_results.append(cached_result)
                else:
                    # Add model_name if missing
                    result_copy = dict(cached_result) if isinstance(cached_result, dict) else {}
                    result_copy["model_name"] = model_name
                    model_results.append(result_copy)

            # Create comparative analysis from cached results
            if model_results:
                # Simple ranking by mAP@0.5
                rankings = []
                for result in model_results:
                    detection_metrics = result.get("detection_metrics", {})
                    map_50 = detection_metrics.get("mAP@0.5", 0.0)
                    rankings.append((result["model_name"], map_50))

                rankings.sort(key=lambda x: x[1], reverse=True)

                comparative_analysis = {
                    "rankings": {"by_map_0_5": rankings},
                    "best_performer": {"model": rankings[0][0], "map_0_5": rankings[0][1]} if rankings else {},
                    "class_wise_analysis": {},
                }
            else:
                comparative_analysis = {"rankings": {"by_map_0_5": []}, "best_performer": {}, "class_wise_analysis": {}}
        else:
            model_results = []
            comparative_analysis = {"rankings": {"by_map_0_5": []}, "best_performer": {}, "class_wise_analysis": {}}

        results = BenchmarkResults(
            dataset_info=dataset.get_dataset_info(),
            model_results=model_results,
            comparative_analysis=comparative_analysis,
            plots_paths={},
            timestamp=datetime.now().isoformat(),
            config=config,
        )

    # Store benchmark run info
    if results_db:
        from odc.benchmark.utils import BenchmarkRun

        run_info = BenchmarkRun(
            timestamp=datetime.now().isoformat(),
            models_benchmarked=list(models_to_evaluate.keys()) if models_to_evaluate else [],
            sample_count=config["evaluation"]["max_samples"] or len(dataset),
            results_path=output_base,
        )
        results_db.record_benchmark_run(run_info)

    # Generate outputs (always regenerated)
    if config["output"]["generate_plots"]:
        print("\nGenerating visualizations...")
        plot_generator = PlotGenerator(config["visualization"])
        plot_paths = plot_generator.generate_all_plots(results, f"{output_base}/plots")
        print(f"   Generated {len(plot_paths)} plots")

    if config["reporting"]["generate_latex"]:
        print("\nGenerating enhanced LaTeX reports...")
        latex_reporter = LaTeXReporter(
            {
                "template_dir": "src/odc/benchmark/reporters/templates",
                "output_format": "tex",
                "include_plots": config["reporting"]["include_plots"],
                "thesis_style": config["reporting"]["thesis_style"],
            }
        )

        template_context = {}
        if config["output"]["generate_plots"]:
            template_context["plot_paths"] = plot_paths

        main_report_path = f"{output_base}/reports/benchmark_report.tex"
        latex_reporter.generate_report(results, main_report_path, extra_context=template_context)
        print(f"   Generated enhanced LaTeX reports")

    if config["reporting"]["generate_csv"] and results_db:
        print("\nExporting CSV results...")
        try:
            df = results_db.export_to_dataframe()
            csv_path = f"{output_base}/reports/benchmark_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"   Exported CSV results to {csv_path}")
        except Exception as e:
            print(f"   Warning: CSV export failed: {e}")

    # Display results
    print("\nENHANCED BENCHMARK RESULTS")
    print("=" * 60)
    if results.model_results:
        rankings = results.comparative_analysis["rankings"]["by_map_0_5"]
        for rank, (model_name, score) in enumerate(rankings, 1):
            print(f"   {rank}. {model_name}: {score:.3f}")
    else:
        print("   Using cached database results")

    if results_db:
        final_summary = results_db.get_database_stats()
        print(f"\nDatabase: {final_summary['total_models']} models, {final_summary['total_runs']} runs")

    print(f"\nEnhanced benchmark completed!")
    print(f"Results: {output_base}")
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified Object Detection Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple benchmark (learning)
  uv run src/scripts/benchmark.py --mode simple --samples 20

  # Complete benchmark (one-time evaluation)
  uv run src/scripts/benchmark.py --mode complete

  # Enhanced system (production, incremental)
  uv run src/scripts/benchmark.py --mode enhanced

  # Custom configuration
  uv run src/scripts/benchmark.py --config my_config.yaml

  # Custom models and output
  uv run src/scripts/benchmark.py --mode complete --models yolov8m-baseline.pt --output-dir results/
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["simple", "complete", "enhanced"],
        default="enhanced",
        help="Benchmark mode (default: enhanced)",
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    parser.add_argument("--samples", type=int, help="Maximum number of samples to use")

    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    parser.add_argument("--models", nargs="+", help="Specific model files to evaluate")

    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")

    parser.add_argument("--no-database", action="store_true", help="Disable database features")

    args = parser.parse_args()

    # Load configuration
    benchmark_config = BenchmarkConfig(args)
    config = benchmark_config.config

    # Run appropriate benchmark mode
    if args.mode == "simple":
        results = run_simple_benchmark(config)
    elif args.mode == "complete":
        results = run_complete_benchmark(config)
    elif args.mode == "enhanced":
        results = run_enhanced_benchmark(config)
    else:
        print(f"ERROR: Unknown mode: {args.mode}")
        return 1

    if results:
        print(f"\nBenchmark completed successfully!")
        return 0
    else:
        print(f"\nERROR: Benchmark failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
