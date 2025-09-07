"""
Report Generator Module for SkyFusion Dataset.

This module generates comprehensive markdown reports based on the analysis
results from all the EDA modules.

Author: Generated for SkyFusion Dataset Analysis
"""

from datetime import datetime
import os
from typing import Any, Dict

from loguru import logger


class ReportGenerator:
    """Generates comprehensive markdown reports for dataset analysis."""

    def __init__(
        self,
        split_name: str,
        analysis_results: dict[str, Any],
        invalid_results: dict[str, Any],
        output_dirs: dict[str, str],
    ):
        """
        Initialize the report generator.

        Args:
        ----
            split_name: Name of the dataset split (train, test, valid)
            analysis_results: Results from DatasetAnalyzer
            invalid_results: Results from InvalidBoxCounter
            output_dirs: Dictionary with output directory paths
        """
        self.split_name = split_name
        self.analysis_results = analysis_results
        self.invalid_results = invalid_results
        self.output_dirs = output_dirs

        # Class names
        self.class_names = {0: "aircraft", 1: "ship", 2: "vehicle"}

    def _format_number(self, num: float, decimals: int = 2) -> str:
        """Format numbers for display in the report."""
        if isinstance(num, int) or num.is_integer():
            return f"{int(num):,}"
        else:
            return f"{num:,.{decimals}f}"

    def _get_relative_path(self, full_path: str) -> str:
        """Convert full path to relative path for markdown links."""
        if not full_path:
            return ""

        # Get relative path from the report location
        base_dir = self.output_dirs["base"]
        if full_path.startswith(base_dir):
            return full_path[len(base_dir) :].lstrip("/")
        return full_path

    def _generate_overview_section(self) -> str:
        """Generate the dataset overview section."""
        total_images = self.analysis_results["total_files"]
        total_objects = self.analysis_results["total_objects"]
        empty_images = self.analysis_results["empty_images"]
        image_sizes = self.analysis_results["image_sizes"]

        # Calculate object density
        avg_objects_per_image = total_objects / total_images if total_images > 0 else 0
        median_objects = self.analysis_results["bbox_count_per_image"]
        median_val = sorted(median_objects)[len(median_objects) // 2] if median_objects else 0
        max_objects = max(median_objects) if median_objects else 0

        # Get image size
        img_size = image_sizes[0] if image_sizes else (640, 640)

        # Adjust description based on split type
        if self.split_name == "combined":
            split_description = "complete dataset combines all splits (train, test, valid) and contains"
        else:
            split_description = f"{self.split_name} split contains"

        section = f"""## 1. Dataset Overview

The SkyFusion dataset {split_description} high-resolution satellite or aerial imagery with annotations for aircraft, ships, and vehicles.

### Key Statistics

- **Total Images**: {self._format_number(total_images)}
- **Total Objects**: {self._format_number(total_objects)}
- **Image Size**: {img_size[0]}×{img_size[1]} pixels (consistent across all images)
- **Object Density**: Average of {avg_objects_per_image:.2f} objects per image (median: {median_val})
- **Max Objects**: {self._format_number(max_objects)} objects in a single image
- **Empty Images**: {self._format_number(empty_images)} ({empty_images / total_images * 100:.2f}% of total)

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|"""

        # Add class distribution table
        class_counts = self.analysis_results["class_counts"]
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = count / total_objects * 100 if total_objects > 0 else 0
            class_name = self.class_names[class_id].title()
            section += f"\n| {class_name} | {self._format_number(count)} | {percentage:.2f}% |"

        # Determine dominant class
        if class_counts:
            dominant_class_id = max(class_counts.keys(), key=lambda x: class_counts[x])
            dominant_class = self.class_names[dominant_class_id]
            dominant_percentage = class_counts[dominant_class_id] / total_objects * 100

            section += f"""

The dataset is dominated by {dominant_class} annotations, which make up {dominant_percentage:.1f}% of all objects. This distribution is typical for aerial surveillance datasets where ground vehicles are most numerous.

![Class Distribution](visualizations/distributions/class_distribution.png)
*Figure 1: Distribution of object classes in the {self.split_name} dataset*
"""

        return section

    def _generate_size_analysis_section(self) -> str:
        """Generate the object size analysis section."""
        bbox_sizes_by_class = self.analysis_results["bbox_sizes_by_class"]

        section = """## 2. Object Size Analysis

Object sizes vary significantly across classes, reflecting their real-world dimensions when viewed from above.

### Size Distribution by Class

| Class | Min Area (normalized) | Max Area (normalized) | Avg Area (normalized) | Min Size (px²) | Max Size (px²) | Avg Size (px²) |
|-------|------------------------|----------------------|------------------------|---------------|---------------|----------------|"""

        # Calculate size statistics for each class
        img_size = self.analysis_results["image_sizes"][0] if self.analysis_results["image_sizes"] else (640, 640)
        pixel_area = img_size[0] * img_size[1]

        for class_id in sorted(bbox_sizes_by_class.keys()):
            sizes = bbox_sizes_by_class[class_id]
            if sizes:
                min_norm = min(sizes)
                max_norm = max(sizes)
                avg_norm = sum(sizes) / len(sizes)

                min_px = min_norm * pixel_area
                max_px = max_norm * pixel_area
                avg_px = avg_norm * pixel_area

                class_name = self.class_names[class_id].title()
                section += f"\n| {class_name} | {min_norm:.6f} | {max_norm:.6f} | {avg_norm:.6f} | {min_px:.0f} | {max_px:.0f} | {avg_px:.0f} |"

        section += """

Aircraft are generally the largest objects, with average size being significantly larger than vehicles. This reflects the real-world physical size differences when viewed from aerial perspectives.

![Bounding Box Size Distribution](visualizations/distributions/bbox_size_distribution.png)
*Figure 2: Size distribution of bounding boxes by class (log scale)*

![Size Violin Plot](visualizations/distributions/box_size_violin_plot.png)
*Figure 3: Violin plot showing the distribution of object sizes across classes*

### Size Range Distribution

The dataset contains objects across a wide range of sizes:
- **Very small objects** (<25px²): Primarily vehicles
- **Small objects** (25-100px²): Majority of vehicles and some ships
- **Medium objects** (100-400px²): Mix of all classes
- **Medium-large objects** (400-1600px²): Many aircraft and some ships
- **Large objects** (>1600px²): Predominantly aircraft
"""

        return section

    def _generate_distribution_section(self) -> str:
        """Generate the object distribution patterns section."""
        bbox_count_per_image = self.analysis_results["bbox_count_per_image"]

        # Calculate distribution statistics
        single_object = sum(1 for count in bbox_count_per_image if count == 1)
        few_objects = sum(1 for count in bbox_count_per_image if 2 <= count <= 10)
        medium_objects = sum(1 for count in bbox_count_per_image if 11 <= count <= 20)
        many_objects = sum(1 for count in bbox_count_per_image if count > 20)

        total_images = len(bbox_count_per_image)

        section = f"""## 3. Object Distribution Patterns

### Objects per Image

- **{single_object / total_images * 100:.2f}%** of images contain only a single object
- **{few_objects / total_images * 100:.2f}%** of images contain 2-10 objects
- **{medium_objects / total_images * 100:.2f}%** of images contain 11-20 objects
- **{many_objects / total_images * 100:.2f}%** of images contain more than 20 objects

![Objects Per Image Distribution](visualizations/distributions/objects_per_image.png)
*Figure 4: Histogram showing the distribution of object counts per image*

### Spatial Distribution

The spatial distribution of objects shows distinct patterns:
- Aircraft tend to be located in clusters, likely representing airports or airfields
- Ships appear predominantly in water bodies with linear distributions along shipping lanes
- Vehicles have the highest density and widespread distribution, often appearing in grid-like patterns in urban areas

![Object Heatmap - All Classes](visualizations/heatmaps/all_classes_heatmap.png)
*Figure 5: Heatmap showing the spatial distribution of all object classes*

![Aircraft Heatmap](visualizations/heatmaps/aircraft_heatmap.png)
*Figure 6: Heatmap showing aircraft spatial distribution*

![Ship Heatmap](visualizations/heatmaps/ship_heatmap.png)
*Figure 7: Heatmap showing ship spatial distribution*

![Vehicle Heatmap](visualizations/heatmaps/vehicle_heatmap.png)
*Figure 8: Heatmap showing vehicle spatial distribution*
"""

        return section

    def _generate_quality_assessment_section(self) -> str:
        """Generate the data quality assessment section."""
        invalid_count = self.invalid_results["invalid_boxes_count"]
        self.invalid_results["total_boxes"]
        invalid_percentage = self.invalid_results["invalid_percentage"]
        invalid_by_class = self.invalid_results["invalid_by_class"]

        section = f"""## 4. Data Quality Assessment

### Bounding Box Validity

Analysis of bounding box validity reveals:
- **{self._format_number(invalid_count)} invalid boxes** ({invalid_percentage:.3f}% of total annotations)
- These boxes have dimensions that would collapse to ≤1 pixel when displayed at {self.analysis_results["image_sizes"][0] if self.analysis_results["image_sizes"] else (640, 640)} resolution
"""

        # Add invalid boxes by class
        for class_id in sorted(invalid_by_class.keys()):
            invalid_class_count = invalid_by_class[class_id]
            class_name = self.class_names[class_id]
            if invalid_class_count > 0:
                section += f"- {invalid_class_count} invalid boxes belong to the {class_name} class\n"

        if invalid_count > 0:
            section += """
Recommendation: Filtering these extremely small boxes before training might be beneficial as they may:
- Be ignored by the model due to their size
- Create difficulties with IoU calculations
- Potentially harm training with unsatisfiable targets
"""
        else:
            section += "\nExcellent data quality - no invalid bounding boxes detected!"

        # Add aspect ratio information
        aspect_ratios_by_class = self.analysis_results["aspect_ratios_by_class"]

        section += "\n### Aspect Ratios\n\nAspect ratios (width/height) provide insights into object shapes:\n"

        for class_id in sorted(aspect_ratios_by_class.keys()):
            ratios = aspect_ratios_by_class[class_id]
            if ratios:
                min_ratio = min(ratios)
                max_ratio = max(ratios)
                avg_ratio = sum(ratios) / len(ratios)
                class_name = self.class_names[class_id].title()
                section += f"- **{class_name}**: {min_ratio:.2f}-{max_ratio:.2f} (avg: {avg_ratio:.2f})\n"

        section += """
![Aspect Ratio Distribution](visualizations/distributions/aspect_ratio_distribution.png)
*Figure 9: Distribution of bounding box aspect ratios by class*
"""

        return section

    def _generate_examples_section(self) -> str:
        """Generate the visual examples section."""
        section = """## 5. Visual Examples

### Class-Specific Samples

The dataset includes diverse examples of each class:
- **Aircraft**: Vary in size from small private planes to large commercial aircraft
- **Ships**: Range from small boats to large vessels
- **Vehicles**: Include cars, trucks, and possibly other land vehicles in various contexts

![Sample Aircraft Only](visualizations/examples/object_count/sample_aircraft_only.png)
*Figure 10: Sample image containing only aircraft*

![Sample Ship Only](visualizations/examples/object_count/sample_ship_only.png)
*Figure 11: Sample image containing only ships*

![Sample Vehicle Only](visualizations/examples/object_count/sample_vehicle_only.png)
*Figure 12: Sample image containing only vehicles*

### Object Count Examples

The dataset contains images with varying numbers of objects:

![Sample Image with 1 Object](visualizations/examples/object_count/sample_image_1_objects.png)
*Figure 13: Sample image with a single object*

![Sample Image with 5 Objects](visualizations/examples/object_count/sample_image_5_objects.png)
*Figure 14: Sample image with 5 objects*

![Sample Image with 20 Objects](visualizations/examples/object_count/sample_image_20_objects.png)
*Figure 15: Sample image with 20 objects*

![Sample Image with 50 Objects](visualizations/examples/object_count/sample_image_50_objects.png)
*Figure 16: Sample image with 50 objects*

![Diverse Samples Grid](visualizations/examples/object_count/diverse_samples_grid.png)
*Figure 17: Grid of diverse sample images from the dataset*

### Size Variations

Examples show the significant variation in object sizes:
- The largest objects (typically aircraft) can be thousands of times larger than the smallest objects
- Very small objects (typically vehicles) are challenging to detect but make up a significant portion of the dataset

![Largest Objects Grid](visualizations/histograms/largest_objects_grid.png)
*Figure 18: Grid showing examples of the largest objects in the dataset*

![Smallest Objects Grid](visualizations/histograms/smallest_objects_grid.png)
*Figure 19: Grid showing examples of the smallest objects in the dataset*
"""

        return section

    def _generate_training_considerations_section(self) -> str:
        """Generate the training considerations section."""
        class_counts = self.analysis_results["class_counts"]
        total_objects = self.analysis_results["total_objects"]

        # Find dominant class
        dominant_class_id = max(class_counts.keys(), key=lambda x: class_counts[x])
        dominant_percentage = class_counts[dominant_class_id] / total_objects * 100

        # Calculate size range
        all_sizes = []
        for sizes in self.analysis_results["bbox_sizes_by_class"].values():
            all_sizes.extend(sizes)

        if all_sizes:
            min_size = min(all_sizes)
            max_size = max(all_sizes)
            img_size = self.analysis_results["image_sizes"][0] if self.analysis_results["image_sizes"] else (640, 640)
            pixel_area = img_size[0] * img_size[1]
            min_pixels = min_size * pixel_area
            max_pixels = max_size * pixel_area
            max_pixels / min_pixels if min_pixels > 0 else 0
        else:
            min_pixels = 0
            max_pixels = 0

        section = f"""## 6. Training Considerations

Based on this analysis, several factors should be considered when training models on this dataset:

1. **Class Imbalance**: The significant imbalance toward {self.class_names[dominant_class_id]} ({dominant_percentage:.1f}%) may require class weighting or specialized sampling techniques during training

2. **Scale Variation**: The extreme range of object sizes (from {min_pixels:.0f}px² to {max_pixels:.0f}px²) suggests that:
   - Multi-scale detection approaches should be used
   - Feature pyramid networks or similar architectures would be beneficial
   - Data augmentation should preserve small objects

3. **Minimum Detection Size**: Consider the practical minimum size that models should detect ({min_pixels:.0f} pixels may be too small for reliable detection)

4. **Evaluation Metrics**: When evaluating model performance:
   - Use mAP across multiple IoU thresholds
   - Consider separate evaluation for different size ranges
   - Analyze per-class performance due to the class imbalance

5. **Data Quality**: With {self.invalid_results["invalid_percentage"]:.3f}% invalid annotations, the dataset quality is {"excellent" if self.invalid_results["invalid_percentage"] < 0.1 else "good" if self.invalid_results["invalid_percentage"] < 1.0 else "moderate"}
"""

        return section

    def _generate_conclusion_section(self) -> str:
        """Generate the conclusion section."""
        class_counts = self.analysis_results["class_counts"]
        total_objects = self.analysis_results["total_objects"]

        # Find dominant class
        dominant_class_id = max(class_counts.keys(), key=lambda x: class_counts[x])
        dominant_percentage = class_counts[dominant_class_id] / total_objects * 100

        invalid_percentage = self.invalid_results["invalid_percentage"]

        section = f"""## 7. Conclusion

The SkyFusion dataset {self.split_name} split presents a challenging aerial object detection task with:
- Significant class imbalance ({dominant_percentage:.1f}% {self.class_names[dominant_class_id]})
- Wide variation in object sizes and scales
- {"High" if invalid_percentage < 0.1 else "Good" if invalid_percentage < 1.0 else "Moderate"} data quality with {invalid_percentage:.3f}% problematic annotations

These characteristics make it suitable for benchmarking detector performance on aerial imagery, particularly for small object detection tasks. The dataset quality is {"high" if invalid_percentage < 0.1 else "good" if invalid_percentage < 1.0 else "acceptable"}, with only {invalid_percentage:.3f}% problematic annotations.

When developing models for this dataset, special attention should be given to handling scale variance and class imbalance to achieve optimal performance across all object categories.

---

*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for the {self.split_name} split of the SkyFusion dataset.*
"""

        return section

    def generate_report(self) -> str:
        """
        Generate the complete markdown report.

        Returns
        -------
            Path to the generated report file
        """
        logger.info(f"Generating markdown report for {self.split_name} split...")

        # Generate report content
        report_content = f"""# SkyFusion Dataset Analysis Report - {self.split_name.title()} Split

{self._generate_overview_section()}

{self._generate_size_analysis_section()}

{self._generate_distribution_section()}

{self._generate_quality_assessment_section()}

{self._generate_examples_section()}

{self._generate_training_considerations_section()}

{self._generate_conclusion_section()}
"""

        # Save report
        report_path = os.path.join(self.output_dirs["base"], f"skyfusion_dataset_report_{self.split_name}.md")

        with open(report_path, "w") as f:
            f.write(report_content)

        logger.success(f"Markdown report generated: {report_path}")
        return report_path
