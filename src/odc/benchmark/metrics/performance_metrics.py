"""Performance metrics for object detection models."""

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List

import psutil

from ..models.base import ObjectDetectionModel


@dataclass
class PerformanceMetrics:
    """Performance metrics for object detection models."""

    inference_time_ms: float
    fps: float
    memory_usage_mb: float
    model_size_mb: float
    parameters: int
    gflops: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "inference_time_ms": self.inference_time_ms,
            "fps": self.fps,
            "memory_usage_mb": self.memory_usage_mb,
            "model_size_mb": self.model_size_mb,
            "parameters": self.parameters,
            "gflops": self.gflops,
        }


class PerformanceEvaluator:
    """Evaluate performance metrics for object detection models."""

    def __init__(self):
        """Initialize the performance evaluator."""
        self.process = psutil.Process(os.getpid())

    def measure_inference_performance(
        self, model: ObjectDetectionModel, test_images: list, num_runs: int = 10
    ) -> PerformanceMetrics:
        """Measure inference performance of a model.

        Args:
            model: Object detection model to evaluate
            test_images: List of test images for inference
            num_runs: Number of inference runs for averaging

        Returns:
            PerformanceMetrics object with measured performance
        """
        if not test_images:
            raise ValueError("No test images provided")

        # Warmup the model
        if not model._warmup_done:
            model.warmup()

        # Measure memory before inference
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB

        # Measure inference time
        inference_times = []

        for _ in range(num_runs):
            start_time = time.perf_counter()

            # Run inference on all test images
            for image in test_images:
                _ = model.predict(image)

            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Measure memory after inference
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before

        # Calculate average inference time per image
        avg_time_per_batch = sum(inference_times) / len(inference_times)
        avg_time_per_image = avg_time_per_batch / len(test_images)
        fps = 1000.0 / avg_time_per_image  # Convert ms to FPS

        # Get model metadata
        metadata = model.get_metadata()

        return PerformanceMetrics(
            inference_time_ms=avg_time_per_image,
            fps=fps,
            memory_usage_mb=max(0, memory_usage),  # Ensure non-negative
            model_size_mb=metadata.model_size_mb,
            parameters=metadata.parameters,
            gflops=metadata.additional_info.get("gflops", 0.0),
        )

    def measure_batch_performance(
        self, model: ObjectDetectionModel, test_images: list, batch_sizes: list[int] = None
    ) -> dict[int, PerformanceMetrics]:
        """Measure performance at different batch sizes.

        Args:
            model: Object detection model to evaluate
            test_images: List of test images for inference
            batch_sizes: List of batch sizes to test (default: [1, 4, 8, 16])

        Returns:
            Dictionary mapping batch sizes to performance metrics
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]

        results = {}

        for batch_size in batch_sizes:
            if batch_size > len(test_images):
                continue

            # Create batches
            batch_images = test_images[:batch_size]

            # Measure performance for this batch size
            performance = self.measure_inference_performance(model, batch_images, num_runs=5)
            results[batch_size] = performance

        return results

    def compare_models_performance(
        self, models: dict[str, ObjectDetectionModel], test_images: list
    ) -> dict[str, PerformanceMetrics]:
        """Compare performance across multiple models.

        Args:
            models: Dictionary mapping model names to model instances
            test_images: List of test images for inference

        Returns:
            Dictionary mapping model names to their performance metrics
        """
        results = {}

        for model_name, model in models.items():
            print(f"Measuring performance for {model_name}...")
            performance = self.measure_inference_performance(model, test_images)
            results[model_name] = performance

        return results

    def generate_performance_summary(self, performance_results: dict[str, PerformanceMetrics]) -> dict[str, Any]:
        """Generate a summary of performance results.

        Args:
            performance_results: Dictionary mapping model names to performance metrics

        Returns:
            Summary dictionary with comparative analysis
        """
        if not performance_results:
            return {}

        # Extract metrics for comparison
        inference_times = {name: perf.inference_time_ms for name, perf in performance_results.items()}
        fps_values = {name: perf.fps for name, perf in performance_results.items()}
        memory_usage = {name: perf.memory_usage_mb for name, perf in performance_results.items()}
        model_sizes = {name: perf.model_size_mb for name, perf in performance_results.items()}

        # Find best and worst performers
        fastest_model = min(inference_times, key=inference_times.get)
        slowest_model = max(inference_times, key=inference_times.get)
        highest_fps = max(fps_values, key=fps_values.get)
        lowest_memory = min(memory_usage, key=memory_usage.get)
        smallest_model = min(model_sizes, key=model_sizes.get)

        summary = {
            "fastest_model": {
                "name": fastest_model,
                "inference_time_ms": inference_times[fastest_model],
                "fps": fps_values[fastest_model],
            },
            "slowest_model": {
                "name": slowest_model,
                "inference_time_ms": inference_times[slowest_model],
                "fps": fps_values[slowest_model],
            },
            "highest_fps_model": {"name": highest_fps, "fps": fps_values[highest_fps]},
            "most_memory_efficient": {"name": lowest_memory, "memory_usage_mb": memory_usage[lowest_memory]},
            "smallest_model": {"name": smallest_model, "model_size_mb": model_sizes[smallest_model]},
            "average_inference_time_ms": sum(inference_times.values()) / len(inference_times),
            "average_fps": sum(fps_values.values()) / len(fps_values),
            "total_model_size_mb": sum(model_sizes.values()),
            "performance_rankings": {
                "by_speed": sorted(inference_times.items(), key=lambda x: x[1]),
                "by_fps": sorted(fps_values.items(), key=lambda x: x[1], reverse=True),
                "by_memory": sorted(memory_usage.items(), key=lambda x: x[1]),
                "by_size": sorted(model_sizes.items(), key=lambda x: x[1]),
            },
        }

        return summary
