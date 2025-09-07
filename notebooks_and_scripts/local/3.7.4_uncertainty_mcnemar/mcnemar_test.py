#!/usr/bin/env python3
"""
McNemar's test for object-level detection success at IoU=0.5 between two models.
Uses COCO matching to determine per-GT-object success (detected vs missed).

Example:
  uv run scripts/stats/mcnemar_test.py \
    --gt /home/yglk/coding/local/masters-thesis-local/data/annotations/test.json \
    --pred-a /home/yglk/coding/local/masters-thesis-local/data/predictions/rt-detr.json \
    --pred-b /home/yglk/coding/local/masters-thesis-local/data/predictions/yolov11m-p2-aug_20250603.json \
    --name-a rt-detr \
    --name-b yolov11m-p2-aug_20250603 \
    --iou-thr 0.5 \
    --out /home/yglk/coding/local/masters-thesis-local/outputs/stats/mcnemar.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.stats import chi2


def get_object_level_matches(coco_gt: COCO, coco_dt, iou_thr: float = 0.5) -> dict[int, bool]:
    """
    Get per-GT-object detection success using COCO matching at specified IoU threshold.
    Returns dict mapping GT annotation ID to boolean (detected or not).
    """
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.evaluate()

    # Extract matches from evaluation results
    # coco_eval.evalImgs contains per-image evaluation results
    gt_matches = {}

    for img_eval in coco_eval.evalImgs:
        if img_eval is None:
            continue

        img_id = img_eval["image_id"]
        cat_id = img_eval["category_id"]

        # Get GT annotations for this image and category
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)

        # Get matches from evaluation (dtMatches has shape [T, D] where T=IoU thresholds, D=detections)
        dt_matches = img_eval["dtMatches"][0]  # Use first (and only) IoU threshold

        # Mark GT objects as detected if they have a match
        gt_matched = set()
        for dt_match in dt_matches:
            if dt_match > 0:  # GT ID (1-indexed), 0 means no match
                gt_matched.add(dt_match)

        # Record success for each GT object
        for gt_ann in gt_anns:
            gt_id = gt_ann["id"]
            gt_matches[gt_id] = gt_id in gt_matched

    return gt_matches


def mcnemar_test(n01: int, n10: int) -> dict[str, float]:
    """
    Perform McNemar's test with continuity correction.
    n01: A=0, B=1 (B detected, A missed)
    n10: A=1, B=0 (A detected, B missed)
    """
    if (n01 + n10) == 0:
        return {"chi2": 0.0, "p_value": 1.0}

    # McNemar test statistic with continuity correction
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return {"chi2": float(chi2_stat), "p_value": float(p_value)}


def proportion_difference_ci(
    n01: int, n10: int, n_total: int, alpha: float = 0.05
) -> tuple[float, tuple[float, float]]:
    """
    Compute difference in detection proportions with 95% CI using normal approximation.
    Returns (delta_p, (ci_low, ci_high))
    """
    if n_total == 0:
        return 0.0, (0.0, 0.0)

    p_a = (n_total - n01) / n_total  # A detected / total
    p_b = (n_total - n10) / n_total  # B detected / total
    delta_p = p_a - p_b

    # Standard error for difference in proportions (paired case)
    # SE = sqrt((n01 + n10) / n_total^2)
    se = np.sqrt(n01 + n10) / n_total if n_total > 0 else 0.0

    # 95% CI using normal approximation
    z_alpha = 1.96  # 97.5th percentile of standard normal
    ci_low = delta_p - z_alpha * se
    ci_high = delta_p + z_alpha * se

    return float(delta_p), (float(ci_low), float(ci_high))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True)
    ap.add_argument("--pred-a", type=str, required=True)
    ap.add_argument("--pred-b", type=str, required=True)
    ap.add_argument("--name-a", type=str, required=True)
    ap.add_argument("--name-b", type=str, required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    # Load COCO data
    coco_gt = COCO(args.gt)
    coco_dt_a = coco_gt.loadRes(args.pred_a)
    coco_dt_b = coco_gt.loadRes(args.pred_b)

    # Get object-level matches for both models
    matches_a = get_object_level_matches(coco_gt, coco_dt_a, args.iou_thr)
    matches_b = get_object_level_matches(coco_gt, coco_dt_b, args.iou_thr)

    # Ensure we have the same GT objects
    all_gt_ids = set(matches_a.keys()) | set(matches_b.keys())

    # Count discordant pairs
    n01 = 0  # A=0, B=1 (A missed, B detected)
    n10 = 0  # A=1, B=0 (A detected, B missed)
    n11 = 0  # A=1, B=1 (both detected)
    n00 = 0  # A=0, B=0 (both missed)

    for gt_id in all_gt_ids:
        a_detected = matches_a.get(gt_id, False)
        b_detected = matches_b.get(gt_id, False)

        if not a_detected and b_detected:
            n01 += 1
        elif a_detected and not b_detected:
            n10 += 1
        elif a_detected and b_detected:
            n11 += 1
        else:
            n00 += 1

    n_total = len(all_gt_ids)

    # Perform McNemar's test
    mcnemar_result = mcnemar_test(n01, n10)

    # Compute proportion difference and CI
    delta_p, (ci_low, ci_high) = proportion_difference_ci(n01, n10, n_total)

    # Compile results
    result = {
        "model_a": args.name_a,
        "model_b": args.name_b,
        "iou_threshold": args.iou_thr,
        "n_total_objects": n_total,
        "contingency_table": {
            "n00_both_missed": n00,
            "n01_a_missed_b_detected": n01,
            "n10_a_detected_b_missed": n10,
            "n11_both_detected": n11,
        },
        "mcnemar_test": {
            "chi2_statistic": mcnemar_result["chi2"],
            "p_value": mcnemar_result["p_value"],
            "significant_at_0_05": mcnemar_result["p_value"] < 0.05,
        },
        "proportion_difference": {"delta_p_a_minus_b": delta_p, "ci95_low": ci_low, "ci95_high": ci_high},
        "detection_rates": {
            f"{args.name_a}_detection_rate": (n10 + n11) / n_total if n_total > 0 else 0.0,
            f"{args.name_b}_detection_rate": (n01 + n11) / n_total if n_total > 0 else 0.0,
        },
    }

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"McNemar's test results:")
    print(f"  Total GT objects: {n_total}")
    print(f"  Discordant pairs: n01={n01}, n10={n10}")
    print(f"  χ² = {mcnemar_result['chi2']:.3f}, p = {mcnemar_result['p_value']:.6f}")
    print(f"  Δp ({args.name_a} - {args.name_b}) = {delta_p:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()
