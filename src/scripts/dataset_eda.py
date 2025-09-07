#!/usr/bin/env python3
"""Dataset Exploratory Data Analysis (EDA) Script.

This script provides a unified interface for running comprehensive EDA analysis
on the SkyFusion dataset, generating visualizations, and producing detailed reports.

Usage:
    python src/scripts/dataset_eda.py [OPTIONS]

Examples:
    # Run EDA on all splits
    python src/scripts/dataset_eda.py --dataset datasets/SkyFusion_yolo --output materials/dataset_eda

    # Run EDA on specific split
    python src/scripts/dataset_eda.py --dataset datasets/SkyFusion_yolo --split train --output results/eda_train

    # Run with custom configuration
    python src/scripts/dataset_eda.py --config eda_config.yaml
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

import yaml


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from odc.dataset_eda import EDAPipeline


def setup_logging(output_dir: str, verbose: bool = False) -> None:
    """Setup logging configuration."""
    log_level = "DEBUG" if verbose else "INFO"

    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Add file logger
    log_file = os.path.join(output_dir, "eda_analysis.log")
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
    )


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file or return defaults."""
    default_config = {
        "dataset": {
            "path": "datasets/SkyFusion_yolo",
            "splits": ["train", "valid", "test"],
            "class_names": {0: "aircraft", 1: "ship", 2: "vehicle"},
        },
        "output": {
            "base_directory": "materials/dataset_eda",
            "create_timestamp_dir": True,
            "generate_combined_report": True,
        },
        "analysis": {
            "generate_distributions": True,
            "generate_heatmaps": True,
            "generate_histograms": True,
            "generate_samples": True,
            "detect_invalid_boxes": True,
            "max_samples_per_visualization": 50,
        },
        "visualization": {"style": "seaborn", "dpi": 300, "figsize": [12, 8], "color_palette": "Set2"},
    }

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            user_config = yaml.safe_load(f)

        # Merge configurations (user config overrides defaults)
        def merge_dicts(default: dict, user: dict) -> dict:
            result = default.copy()
            for key, value in user.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        return merge_dicts(default_config, user_config)

    return default_config


def setup_output_directory(base_dir: str, create_timestamp: bool = True) -> str:
    """Setup output directory structure."""
    if create_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"eda_analysis_{timestamp}")
    else:
        output_dir = base_dir

    # Create directory structure
    directories = [
        output_dir,
        os.path.join(output_dir, "reports"),
        os.path.join(output_dir, "visualizations"),
        os.path.join(output_dir, "data"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    return output_dir


def run_eda_analysis(config: dict, splits: list[str] | None = None) -> str:
    """Run comprehensive EDA analysis."""
    # Setup output directory
    output_dir = setup_output_directory(config["output"]["base_directory"], config["output"]["create_timestamp_dir"])

    # Setup logging
    setup_logging(output_dir, verbose=config.get("verbose", False))

    logger.info("Starting Dataset EDA Analysis")
    logger.info(f"Output directory: {output_dir}")

    # Determine splits to analyze
    if splits is None:
        splits = config["dataset"]["splits"]

    logger.info(f"Analyzing splits: {splits}")

    # Initialize EDA pipeline
    pipeline = EDAPipeline(
        dataset_path=config["dataset"]["path"], class_names=config["dataset"]["class_names"], output_base_dir=output_dir
    )

    # Run analysis for each split
    results = {}
    for split in splits:
        logger.info(f"Processing split: {split}")

        try:
            split_results = pipeline.run_comprehensive_analysis(split=split, config=config["analysis"])
            results[split] = split_results
            logger.success(f"Completed analysis for split: {split}")

        except Exception as e:
            logger.error(f"Error processing split {split}: {e}")
            continue

    # Generate combined report if requested
    if config["output"]["generate_combined_report"] and len(results) > 1:
        logger.info("Generating combined report...")
        try:
            combined_report_path = pipeline.generate_combined_report(results)
            logger.success(f"Combined report saved: {combined_report_path}")
        except Exception as e:
            logger.error(f"Error generating combined report: {e}")

    # Generate summary
    logger.info("EDA Analysis Summary:")
    logger.info(f"   Splits analyzed: {len(results)}")
    logger.info(f"   Output directory: {output_dir}")

    for split, split_results in results.items():
        if split_results:
            logger.info(f"   {split}: {len(split_results.get('visualizations', []))} visualizations generated")

    logger.success("EDA Analysis completed successfully!")
    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive Dataset EDA analysis on SkyFusion dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run EDA on all splits
  python src/scripts/dataset_eda.py --dataset datasets/SkyFusion_yolo

  # Run EDA on specific split
  python src/scripts/dataset_eda.py --dataset datasets/SkyFusion_yolo --split train

  # Use custom configuration
  python src/scripts/dataset_eda.py --config eda_config.yaml

  # Custom output directory
  python src/scripts/dataset_eda.py --output results/my_eda_analysis
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/SkyFusion_yolo",
        help="Path to the SkyFusion dataset directory (default: datasets/SkyFusion_yolo)",
    )

    parser.add_argument(
        "--split", type=str, choices=["train", "valid", "test"], help="Specific split to analyze (default: all splits)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="materials/dataset_eda",
        help="Output directory for results (default: materials/dataset_eda)",
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    parser.add_argument("--no-timestamp", action="store_true", help="Do not create timestamped output directory")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.dataset:
        config["dataset"]["path"] = args.dataset
    if args.output:
        config["output"]["base_directory"] = args.output
    if args.no_timestamp:
        config["output"]["create_timestamp_dir"] = False
    if args.verbose:
        config["verbose"] = True

    # Determine splits to analyze
    splits = [args.split] if args.split else None

    try:
        output_dir = run_eda_analysis(config, splits)
        print(f"\nEDA Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\nWarning: Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
