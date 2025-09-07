#!/usr/bin/env python3
"""
Compute bootstrap confidence interval for the difference in a metric between two models
using arrays saved by bootstrap_coco_map.py (npy files).

Example:
  uv run scripts/stats/pairwise_delta.py \
    --boot-dir /home/yglk/coding/local/masters-thesis-local/outputs/stats/boot \
    --metric mAP50 \
    --a rt-detr \
    --b yolov11m-p2-aug_20250603 \
    --out /home/yglk/coding/local/masters-thesis-local/outputs/stats/pairwise_diff.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-dir", type=str, required=True)
    ap.add_argument("--metric", type=str, default="mAP50", choices=["mAP", "mAP50", "mAP75"])
    ap.add_argument("--a", type=str, required=True, help="Model A name (prefix of npy files)")
    ap.add_argument("--b", type=str, required=True, help="Model B name (prefix of npy files)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    boot_dir = Path(args.boot_dir)
    A = np.load(boot_dir / f"{args.a}_{args.metric}.npy")
    B = np.load(boot_dir / f"{args.b}_{args.metric}.npy")
    D = A - B
    lo, hi = np.percentile(D, [2.5, 97.5])
    out = {
        "metric": args.metric,
        "A": args.a,
        "B": args.b,
        "mean_delta": float(D.mean()),
        "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0 or hi < 0),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
