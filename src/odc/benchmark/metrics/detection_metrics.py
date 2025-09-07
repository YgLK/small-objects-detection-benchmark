"""Detection metrics calculation including mAP using COCO evaluation."""

from typing import Any

import numpy as np

from ..datasets.base import GroundTruthAnnotation
from ..models.base import Detection


class DetectionMetrics:
    """Calculate detection metrics including mAP, precision, recall, and F1-score."""

    def __init__(self, class_names: list[str], iou_thresholds: list[float] | None = None):
        """Initialize the detection metrics calculator.

        Args:
            class_names: List of class names in order of their IDs
            iou_thresholds: IoU thresholds for evaluation (default: COCO standard)
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

        if iou_thresholds is None:
            # COCO standard IoU thresholds: 0.5:0.05:0.95
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05).round(2).tolist()
        else:
            self.iou_thresholds = iou_thresholds

    def calculate_iou(
        self, bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]
    ) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection area
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def calculate_map(
        self,
        all_detections: list[list[Detection]],
        all_ground_truths: list[list[GroundTruthAnnotation]],
        iou_threshold: float,
    ) -> dict[str, Any]:
        """Calculate mean Average Precision (mAP) and other stats for a given IoU threshold."""
        average_precisions = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        per_class_recall = {}

        for class_id, class_name in enumerate(self.class_names):
            class_detections = []
            for image_dets in all_detections:
                class_detections.extend([d for d in image_dets if d.class_id == class_id])

            class_ground_truths = []
            for image_gts in all_ground_truths:
                class_ground_truths.extend([gt for gt in image_gts if gt.class_id == class_id])

            num_gt = len(class_ground_truths)

            if not class_detections and not class_ground_truths:
                average_precisions[class_name] = 1.0
                per_class_recall[class_name] = 1.0
                continue

            if not class_detections:
                average_precisions[class_name] = 0.0
                per_class_recall[class_name] = 0.0
                total_fn += num_gt
                continue

            ap, tp, fp = self._calculate_ap(class_detections, class_ground_truths, iou_threshold)
            average_precisions[class_name] = ap

            class_tp = int(np.sum(tp))
            class_fp = int(np.sum(fp))
            class_fn = num_gt - class_tp

            total_tp += class_tp
            total_fp += class_fp
            total_fn += class_fn

            per_class_recall[class_name] = class_tp / num_gt if num_gt > 0 else 0.0

        mAP = np.mean(list(average_precisions.values()))
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        return {
            "mAP": mAP,
            "per_class_ap": average_precisions,
            "recall": overall_recall,
            "fp": total_fp,
            "fn": total_fn,
            "per_class_recall": per_class_recall,
        }

    def _calculate_ap(
        self, detections: list[Detection], ground_truths: list[GroundTruthAnnotation], iou_threshold: float
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Calculate Average Precision (AP) for a single class and return stats."""
        if not detections:
            return 0.0, np.array([]), np.array([])

        detections.sort(key=lambda d: d.confidence, reverse=True)

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        num_gt = len(ground_truths)

        if num_gt == 0:
            fp = np.ones(len(detections))
        else:
            gt_matched = [False] * num_gt
            for i, det in enumerate(detections):
                best_iou = 0
                best_gt_idx = -1

                for j, gt in enumerate(ground_truths):
                    iou = self._calculate_iou(det.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / (num_gt + 1e-16)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)

        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11

        return ap, tp, fp

    def calculate_comprehensive_metrics(
        self, all_detections: list[list[Detection]], all_ground_truths: list[list[GroundTruthAnnotation]]
    ) -> dict[str, Any]:
        """Calculate comprehensive detection metrics and return them with script-friendly keys."""
        results = {}

        # Use IoU=0.5 for recall, FP, FN calculations
        iou_thresh_for_stats = 0.5
        map_results_50 = self.calculate_map(all_detections, all_ground_truths, iou_thresh_for_stats)

        results["map_50"] = map_results_50.get("mAP", 0.0)
        results["recall"] = map_results_50.get("recall", 0.0)
        results["fp"] = map_results_50.get("fp", 0)
        results["fn"] = map_results_50.get("fn", 0)
        results["per_class_recall"] = map_results_50.get("per_class_recall", {})

        # Calculate mAP at IoU=0.75
        map_results_75 = self.calculate_map(all_detections, all_ground_truths, 0.75)
        results["map_75"] = map_results_75.get("mAP", 0.0)

        # Calculate mAP@[0.5:0.05:0.95] (COCO standard)
        map_values = [results["map_50"]]
        # Skip 0.5 since it's already calculated
        iou_thresholds_coco = [iou for iou in self.iou_thresholds if iou != 0.5 and iou != 0.75]
        map_values.append(results["map_75"])

        for iou_thresh in iou_thresholds_coco:
            map_result = self.calculate_map(all_detections, all_ground_truths, iou_thresh)
            map_values.append(map_result["mAP"])

        results["map_coco"] = np.mean(map_values)

        # Add backward-compatible keys
        results["mAP@0.5"] = results["map_50"]
        results["mAP@0.75"] = results["map_75"]
        results["mAP@[0.5:0.05:0.95]"] = results["map_coco"]

        # Add per-class AP for backward compatibility if needed
        if "per_class_ap" in map_results_50:
            for class_name, ap in map_results_50["per_class_ap"].items():
                results[f"AP@0.5_{class_name}"] = ap
        if "per_class_ap" in map_results_75:
            for class_name, ap in map_results_75["per_class_ap"].items():
                results[f"AP@0.75_{class_name}"] = ap

        return results

    def _calculate_iou(self, box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-16)
