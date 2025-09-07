"""LaTeX report generator for benchmark results."""

from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List

import jinja2

from .base_reporter import BaseReporter, BenchmarkResults


class LaTeXReporter(BaseReporter):
    """Generate LaTeX reports from benchmark results for thesis integration."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the LaTeX reporter.

        Args:
            config: Configuration dictionary with options:
                - template_dir: Directory containing LaTeX templates
                - output_format: 'tex' or 'pdf' (default: 'tex')
                - include_plots: Whether to include plot references (default: True)
                - thesis_style: Thesis formatting style (default: 'academic')
        """
        super().__init__(config)

        self.template_dir = config.get("template_dir", "src/odc/benchmark/reporters/templates")
        self.output_format = config.get("output_format", "tex")
        self.include_plots = config.get("include_plots", True)
        self.thesis_style = config.get("thesis_style", "academic")

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            block_start_string="\\BLOCK{",
            block_end_string="}",
            variable_start_string="\\VAR{",
            variable_end_string="}",
            comment_start_string="\\#{",
            comment_end_string="}",
            line_statement_prefix="%%",
            line_comment_prefix="%#",
            trim_blocks=True,
            autoescape=False,
        )

        # Add custom filters
        self._add_custom_filters()

    def _add_custom_filters(self):
        """Add custom Jinja2 filters for LaTeX formatting."""

        def format_number(value, decimals=3):
            """Format number with specified decimal places."""
            if isinstance(value, (int, float)):
                return f"{value:.{decimals}f}"
            return str(value)

        def format_percentage(value, decimals=1):
            """Format value as percentage."""
            if isinstance(value, (int, float)):
                return f"{value * 100:.{decimals}f}\\%"
            return str(value)

        def escape_latex(text):
            """Escape special LaTeX characters."""
            if not isinstance(text, str):
                text = str(text)

            # LaTeX special characters
            latex_chars = {
                "&": "\\&",
                "%": "\\%",
                "$": "\\$",
                "#": "\\#",
                "^": "\\textasciicircum{}",
                "_": "\\_",
                "{": "\\{",
                "}": "\\}",
                "~": "\\textasciitilde{}",
                "\\": "\\textbackslash{}",
            }

            for char, replacement in latex_chars.items():
                text = text.replace(char, replacement)

            return text

        def format_model_name(name):
            """Format model name for LaTeX display."""
            # Replace underscores and hyphens for better display
            formatted = name.replace("_", "\\_").replace("-", "--")
            return formatted

        def format_large_number(value):
            """Format large numbers with commas."""
            if isinstance(value, (int, float)):
                return f"{value:,}"
            return str(value)

        def enumerate_filter(iterable, start=0):
            """Enumerate filter for Jinja2."""
            return enumerate(iterable, start)

        # Register filters
        self.jinja_env.filters["format_number"] = format_number
        self.jinja_env.filters["format_percentage"] = format_percentage
        self.jinja_env.filters["escape_latex"] = escape_latex
        self.jinja_env.filters["format_model_name"] = format_model_name
        self.jinja_env.filters["format_large_number"] = format_large_number
        self.jinja_env.filters["enumerate"] = enumerate_filter

    def generate_report(self, results: BenchmarkResults, output_path: str, extra_context: dict[str, Any] = None) -> str:
        """Generate a comprehensive LaTeX report.

        Args:
            results: BenchmarkResults object containing all benchmark data
            output_path: Path where the report should be saved
            extra_context: Additional context data (e.g., plot paths, visualization paths)

        Returns:
            Path to the generated report file
        """
        if not self.validate_results(results):
            raise ValueError("Invalid benchmark results provided")

        # Prepare template data
        template_data = self._prepare_template_data(results)

        # Add extra context if provided
        if extra_context:
            # Convert plot paths to relative paths before adding to template data
            if "plot_paths" in extra_context:
                print("Converting plot paths to relative paths...")
                extra_context["plot_paths"] = self._convert_plot_paths_to_relative(
                    extra_context["plot_paths"], output_path
                )
                # Validate that the converted paths exist
                extra_context["plot_paths"] = self._validate_plot_paths(
                    extra_context["plot_paths"], os.path.dirname(output_path)
                )
                # Store for debugging in PDF compilation
                self._last_plot_paths = extra_context["plot_paths"]

            template_data.update(extra_context)

        # Load and render main template
        template = self.jinja_env.get_template("benchmark_report.tex.jinja")
        rendered_content = template.render(**template_data)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_content)

        print(f"LaTeX report generated: {output_path}")

        # Automatically compile to PDF
        pdf_path = self._compile_to_pdf(output_path)
        if pdf_path:
            print(f"PDF report compiled: {pdf_path}")
            return pdf_path
        else:
            print(f"PDF compilation failed, but .tex file is available")
            return output_path

    def _prepare_template_data(self, results: BenchmarkResults) -> dict[str, Any]:
        """Prepare data for template rendering.

        Args:
            results: BenchmarkResults object

        Returns:
            Dictionary with template data
        """
        # Basic information
        template_data = {
            "timestamp": results.timestamp,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": results.dataset_info,
            "config": results.config,
            "include_plots": self.include_plots,
            "thesis_style": self.thesis_style,
        }

        # Process model results
        model_results = []
        for model_result in results.model_results:
            processed_result = {
                "name": model_result["model_name"],
                "metadata": model_result["model_metadata"],
                "detection_metrics": model_result["detection_metrics"],
                "performance_metrics": model_result["performance_metrics"],
                "statistics": model_result["statistics"],
            }
            model_results.append(processed_result)

        template_data["model_results"] = model_results

        # Process comparative analysis
        comp_analysis = results.comparative_analysis
        template_data["comparative_analysis"] = comp_analysis

        # Create summary tables data
        template_data["summary_table"] = self._create_summary_table(model_results)
        template_data["performance_table"] = self._create_performance_table(model_results)
        template_data["class_wise_table"] = self._create_class_wise_table(
            model_results, results.dataset_info["class_names"]
        )

        # Rankings and best performers
        template_data["rankings"] = comp_analysis.get("rankings", {})
        template_data["best_performer"] = comp_analysis.get("best_performer", {})
        template_data["class_analysis"] = comp_analysis.get("class_wise_analysis", {})

        return template_data

    def _create_summary_table(self, model_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create summary table data for LaTeX rendering."""
        summary_data = []

        for result in model_results:
            metrics = result["detection_metrics"]
            perf = result["performance_metrics"]

            row = {
                "model_name": result["name"],
                "map_50": metrics.get("mAP@0.5", 0),
                "map_75": metrics.get("mAP@0.75", 0),
                "map_coco": metrics.get("mAP@[0.5:0.05:0.95]", 0),
                "inference_time": perf.get("inference_time_ms", 0),
                "fps": perf.get("fps", 0),
                "parameters": perf.get("parameters", 0),
                "model_size": perf.get("model_size_mb", 0),
            }
            summary_data.append(row)

        return summary_data

    def _create_performance_table(self, model_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create performance comparison table data."""
        performance_data = []

        for result in model_results:
            perf = result["performance_metrics"]
            metadata = result["metadata"]

            row = {
                "model_name": result["name"],
                "parameters": perf.get("parameters", 0),
                "model_size_mb": perf.get("model_size_mb", 0),
                "gflops": perf.get("gflops", 0),
                "inference_time_ms": perf.get("inference_time_ms", 0),
                "fps": perf.get("fps", 0),
                "memory_usage_mb": perf.get("memory_usage_mb", 0),
            }
            performance_data.append(row)

        return performance_data

    def _create_class_wise_table(
        self, model_results: list[dict[str, Any]], class_names: list[str]
    ) -> list[dict[str, Any]]:
        """Create class-wise performance table data."""
        class_wise_data = []

        for result in model_results:
            metrics = result["detection_metrics"]

            row = {"model_name": result["name"]}

            # Add AP for each class
            for class_name in class_names:
                ap_key = f"AP@0.5_{class_name}"
                row[f"ap_{class_name}"] = metrics.get(ap_key, 0)

            class_wise_data.append(row)

        return class_wise_data

    def generate_model_comparison_report(
        self, results: BenchmarkResults, output_path: str, extra_context: dict[str, Any] = None
    ) -> str:
        """Generate a focused model comparison report for thesis chapters.

        Args:
            results: BenchmarkResults object
            output_path: Path where the report should be saved
            extra_context: Additional context data (e.g., plot paths, visualization paths)

        Returns:
            Path to the generated report file
        """
        template_data = self._prepare_template_data(results)

        # Add extra context if provided
        if extra_context:
            template_data.update(extra_context)

        template = self.jinja_env.get_template("model_comparison.tex.jinja")
        rendered_content = template.render(**template_data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_content)

        print(f"Model comparison report generated: {output_path}")

        # Automatically compile to PDF
        pdf_path = self._compile_to_pdf(output_path)
        if pdf_path:
            print(f"PDF comparison report compiled: {pdf_path}")
            return pdf_path
        else:
            print(f"PDF compilation failed, but .tex file is available")
            return output_path

    def generate_appendix_tables(self, results: BenchmarkResults, output_path: str) -> str:
        """Generate detailed appendix tables for thesis.

        Args:
            results: BenchmarkResults object
            output_path: Output file path

        Returns:
            Path to generated appendix
        """
        template_data = self._prepare_template_data(results)

        # Load appendix template
        template = self.jinja_env.get_template("appendix_tables.tex.jinja")
        rendered_content = template.render(**template_data)

        # Write the appendix
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_content)

        print(f"Appendix tables generated: {output_path}")

        # Automatically compile to PDF
        pdf_path = self._compile_to_pdf(output_path)
        if pdf_path:
            print(f"PDF appendix compiled: {pdf_path}")
            return pdf_path
        else:
            print(f"PDF compilation failed, but .tex file is available")
            return output_path

    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats."""
        return ["tex", "pdf"]

    def _compile_to_pdf(self, tex_path: str) -> str:
        """Compile LaTeX file to PDF using pdflatex.

        Args:
            tex_path: Path to the .tex file

        Returns:
            Path to the generated PDF file, or None if compilation failed
        """
        tex_file = Path(tex_path)
        output_dir = tex_file.parent
        pdf_path = tex_file.with_suffix(".pdf")

        # Add debug information about working directory and file paths
        print(f"LaTeX compilation working directory: {output_dir}")
        print(f"LaTeX file: {tex_file.name}")

        # Check if plot files exist relative to LaTeX file
        if hasattr(self, "_last_plot_paths"):
            print("Checking plot file accessibility from LaTeX directory:")
            for plot_name, plot_path in self._last_plot_paths.items():
                full_plot_path = output_dir / plot_path
                exists = full_plot_path.exists()
                print(f"   {plot_name}: {plot_path} {'OK' if exists else 'MISSING'}")
                if not exists:
                    print(f"      Expected at: {full_plot_path}")

        # Check if pdflatex is available
        if not shutil.which("pdflatex"):
            print("pdflatex not found in PATH. Please install TeX Live or MiKTeX.")
            return None

        try:
            # Run pdflatex twice to resolve references
            for run in range(2):
                print(f"Running pdflatex (pass {run + 1}/2)...")

                # Run pdflatex with appropriate options
                cmd = [
                    "pdflatex",
                    "-interaction=nonstopmode",  # Don't stop on errors
                    "-file-line-error",  # Better error messages
                    tex_file.name,  # Use just the filename
                ]

                result = subprocess.run(
                    cmd,
                    cwd=output_dir,  # Change to output directory
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                )

                # Check return code but be more tolerant (LaTeX can return 1 but still succeed)
                if result.returncode != 0:
                    # Check if PDF was actually created despite non-zero return code
                    if pdf_path.exists() and run == 1:  # Only check on final run
                        print(f"pdflatex returned {result.returncode} but PDF was created")
                        break
                    elif run == 0:  # First run, try again
                        print(f"pdflatex pass {run + 1} returned {result.returncode}, trying again...")
                        continue
                    else:  # Final run failed
                        print(f"pdflatex failed with return code {result.returncode}")
                        if result.stderr:
                            print(f"stderr: {result.stderr[-300:]}")  # Last 300 chars
                        # Also check for common image-related errors in stdout
                        if result.stdout and (
                            "cannot find" in result.stdout.lower() or "file not found" in result.stdout.lower()
                        ):
                            print("Possible image file issues detected in LaTeX output:")
                            # Extract relevant lines
                            lines = result.stdout.split("\n")
                            for line in lines:
                                if any(
                                    keyword in line.lower()
                                    for keyword in ["cannot find", "file not found", "includegraphics"]
                                ):
                                    print(f"{line.strip()}")
                        return None

            # Check if PDF was actually created
            if pdf_path.exists():
                # Clean up auxiliary files
                self._cleanup_latex_aux_files(tex_file)
                print(f"PDF successfully created: {pdf_path}")
                return str(pdf_path)
            else:
                print("PDF file was not created despite successful pdflatex execution")
                return None

        except subprocess.TimeoutExpired:
            print("pdflatex compilation timed out after 120 seconds")
            return None
        except FileNotFoundError:
            print("pdflatex command not found")
            return None
        except Exception as e:
            print(f"Unexpected error during PDF compilation: {e}")
            return None

    def _cleanup_latex_aux_files(self, tex_file: Path):
        """Clean up auxiliary files created by pdflatex.

        Args:
            tex_file: Path to the original .tex file
        """
        aux_extensions = [".aux", ".log", ".fls", ".fdb_latexmk", ".out", ".toc", ".lof", ".lot"]

        for ext in aux_extensions:
            aux_file = tex_file.with_suffix(ext)
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors

    def _convert_plot_paths_to_relative(self, plot_paths: dict[str, str], latex_file_path: str) -> dict[str, str]:
        """Convert absolute plot paths to relative paths from LaTeX file location.

        Args:
            plot_paths: Dictionary mapping plot names to absolute file paths
            latex_file_path: Path to the LaTeX file that will reference the plots

        Returns:
            Dictionary mapping plot names to relative file paths
        """
        latex_dir = os.path.dirname(os.path.abspath(latex_file_path))
        relative_paths = {}

        print(f"LaTeX file directory: {latex_dir}")

        for plot_name, absolute_path in plot_paths.items():
            try:
                # Convert to absolute path first to handle any relative input paths
                abs_plot_path = os.path.abspath(absolute_path)
                relative_path = os.path.relpath(abs_plot_path, latex_dir)
                relative_paths[plot_name] = relative_path
                print(f"{plot_name}: {absolute_path} -> {relative_path}")
            except ValueError as e:
                # Fallback to absolute path if relative conversion fails
                print(f"Warning: Could not convert {plot_name} to relative path: {e}")
                print(f"Using absolute path as fallback: {absolute_path}")
                relative_paths[plot_name] = absolute_path

        return relative_paths

    def _validate_plot_paths(self, plot_paths: dict[str, str], latex_dir: str) -> dict[str, str]:
        """Validate that plot files exist and paths are accessible.

        Args:
            plot_paths: Dictionary mapping plot names to relative file paths
            latex_dir: Directory containing the LaTeX file

        Returns:
            Dictionary mapping plot names to validated relative file paths
        """
        validated_paths = {}

        print("Validating plot file existence...")

        for plot_name, relative_path in plot_paths.items():
            full_path = os.path.join(latex_dir, relative_path)
            if os.path.exists(full_path):
                validated_paths[plot_name] = relative_path
                print(f"{plot_name}: {relative_path}")
            else:
                print(f"Warning: Plot file not found: {full_path}")
                print(f"Skipping {plot_name} from LaTeX template")
                # Could optionally create a placeholder or skip the plot

        return validated_paths
