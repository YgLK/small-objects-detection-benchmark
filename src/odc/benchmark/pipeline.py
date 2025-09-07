"""Main benchmarking pipeline for object detection models."""

from datetime import datetime
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .datasets.base import GroundTruthAnnotation, ObjectDetectionDataset
from .metrics.detection_metrics import DetectionMetrics
from .metrics.performance_metrics import PerformanceEvaluator, PerformanceMetrics
from .models.base import Detection, ObjectDetectionModel
from .reporters.base_reporter import BenchmarkResults


class BenchmarkPipeline:
    """Main pipeline for benchmarking object detection models."""

    def __init__(self, dataset: ObjectDetectionDataset, config: dict[str, Any]):
        """Initialize the benchmark pipeline.

        Args:
            dataset: Dataset to use for benchmarking
            config: Configuration dictionary with benchmark settings
        """
        self.dataset = dataset
        self.config = config

        # Initialize metrics calculators
        self.detection_metrics = DetectionMetrics(dataset.get_class_names())
        self.performance_evaluator = PerformanceEvaluator()

        # Results storage
        self.results = {}

    def run_benchmark(
        self, models: dict[str, ObjectDetectionModel], max_samples: int | None = None
    ) -> BenchmarkResults:
        """Run comprehensive benchmark on all provided models.

        Args:
            models: Dictionary mapping model names to model instances
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            BenchmarkResults object with all benchmark data
        """
        print(f"Starting benchmark with {len(models)} models on {len(self.dataset)} samples...")

        # Prepare dataset samples
        samples = list(self.dataset)
        if max_samples is not None:
            samples = samples[:max_samples]

        print(f"Using {len(samples)} samples for evaluation")

        # Run benchmark for each model
        model_results = []
        all_detections = {}
        all_ground_truths = []

        # Prepare ground truth data (same for all models)
        for sample in samples:
            all_ground_truths.append(sample.annotations)

        for model_name, model in models.items():
            print(f"\n=== Benchmarking {model_name} ===")

            # Run inference and collect results
            model_result = self._benchmark_single_model(model, model_name, samples)
            model_results.append(model_result)

            # Store detections for comparative analysis
            all_detections[model_name] = model_result["detections"]

        # Perform comparative analysis
        comparative_analysis = self._perform_comparative_analysis(all_detections, all_ground_truths, models)

        # Create benchmark results
        benchmark_results = BenchmarkResults(
            dataset_info=self.dataset.get_dataset_info(),
            model_results=model_results,
            comparative_analysis=comparative_analysis,
            plots_paths={},  # Will be populated by reporters
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config,
        )

        print(f"\nBenchmark completed successfully!")
        return benchmark_results

    def _benchmark_single_model(self, model: ObjectDetectionModel, model_name: str, samples: list) -> dict[str, Any]:
        """Benchmark a single model on the dataset samples.

        Args:
            model: Model to benchmark
            model_name: Name of the model
            samples: List of dataset samples

        Returns:
            Dictionary with model benchmark results
        """
        print(f"Running inference on {len(samples)} samples...")

        # Collect predictions and ground truths
        all_detections = []
        all_ground_truths = []
        inference_times = []

        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")

            # Measure inference time
            start_time = time.perf_counter()
            detections = model.predict(sample.image)
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)  # Convert to ms

            all_detections.append(detections)
            all_ground_truths.append(sample.annotations)

        print(f"Calculating detection metrics...")

        # Calculate detection metrics
        detection_results = self.detection_metrics.calculate_comprehensive_metrics(all_detections, all_ground_truths)

        print(f"Calculating performance metrics...")

        # Calculate performance metrics
        test_images = [sample.image for sample in samples[:10]]  # Use first 10 for performance
        performance_metrics = self.performance_evaluator.measure_inference_performance(model, test_images, num_runs=5)

        # Get model metadata
        metadata = model.get_metadata()

        # Calculate additional statistics
        total_detections = sum(len(dets) for dets in all_detections)
        total_ground_truths = sum(len(gts) for gts in all_ground_truths)
        avg_detections_per_image = total_detections / len(samples) if samples else 0
        avg_ground_truths_per_image = total_ground_truths / len(samples) if samples else 0

        # Compile results
        result = {
            "model_name": model_name,
            "model_metadata": metadata,
            "detection_metrics": detection_results,
            "performance_metrics": performance_metrics.to_dict(),
            "detections": all_detections,  # Store for comparative analysis
            "statistics": {
                "total_samples": len(samples),
                "total_detections": total_detections,
                "total_ground_truths": total_ground_truths,
                "avg_detections_per_image": avg_detections_per_image,
                "avg_ground_truths_per_image": avg_ground_truths_per_image,
                "avg_inference_time_ms": np.mean(inference_times),
                "std_inference_time_ms": np.std(inference_times),
            },
        }

        # Print summary
        print(f"Results for {model_name}:")
        print(f"  mAP@0.5: {detection_results.get('mAP@0.5', 0):.3f}")
        print(f"  mAP@0.75: {detection_results.get('mAP@0.75', 0):.3f}")
        print(f"  mAP@[0.5:0.05:0.95]: {detection_results.get('mAP@[0.5:0.05:0.95]', 0):.3f}")
        print(f"  Inference time: {performance_metrics.inference_time_ms:.2f} ms")
        print(f"  FPS: {performance_metrics.fps:.1f}")

        return result

    def _perform_comparative_analysis(
        self,
        all_detections: dict[str, list[list[Detection]]],
        all_ground_truths: list[list[GroundTruthAnnotation]],
        models: dict[str, ObjectDetectionModel],
    ) -> dict[str, Any]:
        """Perform comparative analysis across all models.

        Args:
            all_detections: Dictionary mapping model names to their detections
            all_ground_truths: Ground truth annotations
            models: Dictionary of models

        Returns:
            Dictionary with comparative analysis results
        """
        print("Performing comparative analysis...")

        analysis = {}

        # Model rankings by different metrics
        model_names = list(all_detections.keys())

        # Calculate mAP@0.5 for ranking
        map_scores = {}
        for model_name in model_names:
            detections = all_detections[model_name]
            map_result = self.detection_metrics.calculate_map(detections, all_ground_truths, 0.5)
            map_scores[model_name] = map_result["mAP"]

        # Rankings
        analysis["rankings"] = {"by_map_0_5": sorted(map_scores.items(), key=lambda x: x[1], reverse=True)}

        # Best and worst performers
        best_model = max(map_scores, key=map_scores.get)
        worst_model = min(map_scores, key=map_scores.get)

        analysis["best_performer"] = {"model": best_model, "map_0_5": map_scores[best_model]}

        analysis["worst_performer"] = {"model": worst_model, "map_0_5": map_scores[worst_model]}

        # Performance comparison
        performance_comparison = {}
        for model_name, model in models.items():
            metadata = model.get_metadata()
            performance_comparison[model_name] = {
                "parameters": metadata.parameters,
                "model_size_mb": metadata.model_size_mb,
                "gflops": metadata.additional_info.get("gflops", 0.0),
            }

        analysis["performance_comparison"] = performance_comparison

        # Class-wise analysis
        class_analysis = {}
        for class_id, class_name in enumerate(self.dataset.get_class_names()):
            class_scores = {}
            for model_name in model_names:
                detections = all_detections[model_name]
                map_result = self.detection_metrics.calculate_map(detections, all_ground_truths, 0.5)
                class_scores[model_name] = map_result.get(class_name, 0.0)

            class_analysis[class_name] = {
                "best_model": max(class_scores, key=class_scores.get),
                "best_score": max(class_scores.values()),
                "worst_model": min(class_scores, key=class_scores.get),
                "worst_score": min(class_scores.values()),
                "all_scores": class_scores,
            }

        analysis["class_wise_analysis"] = class_analysis

        return analysis

    def quick_benchmark(self, models: dict[str, ObjectDetectionModel], num_samples: int = 50) -> dict[str, Any]:
        """Run a quick benchmark with limited samples for rapid evaluation.

        Args:
            models: Dictionary mapping model names to model instances
            num_samples: Number of samples to use for quick evaluation

        Returns:
            Dictionary with quick benchmark results
        """
        print(f"Running quick benchmark with {num_samples} samples...")

        # Use subset of dataset
        samples = list(self.dataset)[:num_samples]

        results = {}
        for model_name, model in models.items():
            print(f"Quick test: {model_name}")

            # Run inference
            detections = []
            ground_truths = []

            for sample in samples:
                dets = model.predict(sample.image)
                detections.append(dets)
                ground_truths.append(sample.annotations)

            # Calculate basic metrics - fix: pass the lists correctly
            map_result = self.detection_metrics.calculate_map(detections, ground_truths, 0.5)

            results[model_name] = {
                "mAP@0.5": map_result["mAP"],
                "total_detections": sum(len(dets) for dets in detections),
                "samples_tested": len(samples),
            }

            print(f"  mAP@0.5: {map_result['mAP']:.3f}")

        return results
