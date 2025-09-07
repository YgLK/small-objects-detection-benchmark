"""Base classes for benchmark report generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class BenchmarkResults:
    """Comprehensive benchmark results data structure.

    Attributes:
        dataset_info: Information about the dataset used for benchmarking
        model_results: List of results for each model tested
        comparative_analysis: Cross-model comparison data
        plots_paths: Dictionary mapping plot names to their file paths
        timestamp: When the benchmark was run
        config: Configuration used for the benchmark
    """

    dataset_info: dict[str, Any]
    model_results: list[dict[str, Any]]
    comparative_analysis: dict[str, Any]
    plots_paths: dict[str, str]
    timestamp: str
    config: dict[str, Any]


class BaseReporter(ABC):
    """Abstract base class for benchmark report generators.

    This interface ensures all reporters provide consistent functionality
    for generating reports from benchmark results.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the reporter.

        Args:
            config: Reporter-specific configuration parameters
        """
        self.config = config

    @abstractmethod
    def generate_report(self, results: BenchmarkResults, output_path: str) -> str:
        """Generate a report from benchmark results.

        Args:
            results: BenchmarkResults object containing all benchmark data
            output_path: Path where the report should be saved

        Returns:
            Path to the generated report file
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats.

        Returns:
            List of supported file format extensions (e.g., ['tex', 'pdf'])
        """
        pass

    def validate_results(self, results: BenchmarkResults) -> bool:
        """Validate that the benchmark results are complete and valid.

        Args:
            results: BenchmarkResults object to validate

        Returns:
            True if results are valid, False otherwise
        """
        if not results.model_results:
            return False

        if not results.dataset_info:
            return False

        # Check that all model results have required fields
        required_fields = ["model_name", "detection_metrics", "performance_metrics"]
        for model_result in results.model_results:
            for field in required_fields:
                if field not in model_result:
                    return False

        return True

    def get_timestamp(self) -> str:
        """Get current timestamp for report generation.

        Returns:
            Formatted timestamp string
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
