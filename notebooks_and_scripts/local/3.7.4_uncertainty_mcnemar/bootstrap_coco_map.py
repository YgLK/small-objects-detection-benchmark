#!/usr/bin/env python3
"""
Compute paired bootstrap confidence intervals for COCO detection metrics
(mAP@[0.5:0.95], mAP@0.5, mAP@0.75) from saved COCO-format GT and
prediction JSONs, without re-running inference.

Example:
  uv run scripts/stats/bootstrap_coco_map.py \
    --gt /home/yglk/coding/local/masters-thesis-local/data/annotations/test.json \
    --pred-dir /home/yglk/coding/local/masters-thesis-local/data/predictions \
    --models yolov11m-p2-aug_20250603 rt-detr \
    --B 1000 \
    --out-dir /home/yglk/coding/local/masters-thesis-local/outputs/stats
"""

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_subset(coco_gt: COCO, coco_dt, img_ids: list[int]) -> dict[str, float]:
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.params.imgIds = img_ids
    e.evaluate()
    e.accumulate()
    e.summarize()
    return {
        "mAP": float(e.stats[0]),
        "mAP50": float(e.stats[1]),
        "mAP75": float(e.stats[2]),
    }


essential_keys = ["mAP", "mAP50", "mAP75"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True, help="Path to COCO GT JSON file")
    ap.add_argument("--pred-dir", type=str, required=True, help="Directory with prediction JSONs named <model>.json")
    ap.add_argument("--models", nargs="+", required=True, help="Model names to include (match filenames)")
    ap.add_argument("--B", type=int, default=1000, help="Bootstrap iterations (default: 1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for summaries and arrays")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_dir = out_dir / "boot"
    boot_dir.mkdir(parents=True, exist_ok=True)

    # Load GT
    coco_gt = COCO(args.gt)
    all_img_ids = sorted(coco_gt.getImgIds())

    # Load Dets per model
    coco_dts = {}
    for name in args.models:
        p = Path(args.pred_dir) / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"Predictions missing for {name}: {p}")
        coco_dts[name] = coco_gt.loadRes(str(p))

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Storage arrays
    arrays = {name: {k: np.zeros(args.B, dtype=np.float32) for k in essential_keys} for name in args.models}

    for b in range(args.B):
        sample_ids = [rng.choice(all_img_ids) for _ in all_img_ids]
        for name in args.models:
            m = eval_subset(coco_gt, coco_dts[name], sample_ids)
            for k in essential_keys:
                arrays[name][k][b] = m[k]

    # Save arrays and summary
    summary = {}
    for name in args.models:
        summary[name] = {}
        for k in essential_keys:
            arr = arrays[name][k]
            np.save(boot_dir / f"{name}_{k}.npy", arr)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            summary[name][k] = {
                "mean": float(arr.mean()),
                "se": float(arr.std(ddof=1)),
                "ci95": [float(lo), float(hi)],
            }

    (out_dir / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {out_dir / 'bootstrap_summary.json'}")


if __name__ == "__main__":
    main()
