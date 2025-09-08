#!/usr/bin/env python3
"""
Count images per spatial-density bucket from a COCO dataset.

Density bins (matching evaluation script):
- sparse: 0-9 objects per 640x640 image
- medium: 10-29
- dense: 30+

Usage examples:
  python src/scripts/count_density_bins.py --path datasets/SkyFusion
  python src/scripts/count_density_bins.py --path datasets/SkyFusion/SkyFusion.zip
  python src/scripts/count_density_bins.py --path datasets/SkyFusion/annotations/instances_test.json

Outputs per-split counts and a combined summary. Optionally writes CSV.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable
import io
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import zipfile


# Canonical density bins (inclusive lower, exclusive upper for upper bounds)
DENSITY_BINS: dict[str, tuple[int, float]] = {
    "sparse": (0, 10),  # 0-9
    "medium": (10, 30),  # 10-29
    "dense": (30, float("inf")),  # 30+
}


def _iter_annotation_jsons(path: Path) -> Iterable[tuple[str, dict]]:
    """Yield (name, json_obj) from:
    - A single JSON file path
    - A directory (searches for */annotations/*.json and any *.json recursively)
    - A .zip file containing annotations/*.json
    """
    if path.is_file() and path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            yield (path.name, json.load(f))
        return

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            # Prefer common COCO locations
            json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            # Prioritize paths that include "annotations"
            json_names.sort(key=lambda n: ("annotations" not in n.lower(), n))
            for name in json_names:
                with zf.open(name) as fp:
                    data = json.load(io.TextIOWrapper(fp, encoding="utf-8"))
                yield (name, data)
        return

    if path.is_dir():
        # Try typical COCO layout first
        anno_dir = path / "annotations"
        candidates: list[Path] = []
        if anno_dir.exists():
            candidates.extend(sorted(anno_dir.glob("*.json")))
        # Fallback: any JSONs under the directory
        if not candidates:
            candidates.extend(sorted(path.rglob("*.json")))
        for p in candidates:
            try:
                with p.open("r", encoding="utf-8") as f:
                    yield (str(p.relative_to(path)), json.load(f))
            except Exception as e:
                print(f"Warning: Skipping {p}: {e}")
        return

    raise FileNotFoundError(f"No JSON found at: {path}")


def _bin_for_count(count: int) -> str:
    for name, (lo, hi) in DENSITY_BINS.items():
        if lo <= count < hi:
            return name
    # Should never happen
    return "dense" if count >= 30 else "sparse"


def summarize_split(name: str, coco: dict, ignore_crowd: bool = False) -> tuple[Counter, int]:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # Count annotations per image_id
    ann_by_img: dict[int, int] = defaultdict(int)
    for ann in annotations:
        if ignore_crowd and ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann.get("image_id")
        if img_id is not None:
            ann_by_img[img_id] += 1

    # Ensure images with zero objects are counted as 0
    bin_counts: Counter = Counter()
    for img in images:
        c = ann_by_img.get(img.get("id"), 0)
        bin_counts[_bin_for_count(c)] += 1

    total_images = len(images)
    return bin_counts, total_images


def format_summary(name: str, bin_counts: Counter, total: int) -> str:
    def pct(n: int) -> str:
        return f"{(100.0 * n / total):.1f}%" if total > 0 else "n/a"

    return (
        f"Split: {name}\n"
        f"  Total images: {total}\n"
        f"  sparse (0-9):  {bin_counts['sparse']:>5}  ({pct(bin_counts['sparse'])})\n"
        f"  medium (10-29):{bin_counts['medium']:>5}  ({pct(bin_counts['medium'])})\n"
        f"  dense (≥30):   {bin_counts['dense']:>5}  ({pct(bin_counts['dense'])})\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Count images per density bin from COCO annotations")
    p.add_argument("--path", required=True, help="Path to COCO root dir, annotations JSON, or a .zip archive")
    p.add_argument("--ignore-crowd", action="store_true", help="Ignore annotations with iscrowd=1")
    p.add_argument("--csv", help="Optional path to write a CSV summary (combined only)")
    args = p.parse_args()

    path = Path(args.path)

    per_split: list[tuple[str, Counter, int]] = []
    for name, coco in _iter_annotation_jsons(path):
        # Heuristic: skip non-COCO JSONs
        if not (isinstance(coco, dict) and "images" in coco and "annotations" in coco):
            continue
        bin_counts, total = summarize_split(name, coco, ignore_crowd=args.ignore_crowd)
        per_split.append((name, bin_counts, total))
        print(format_summary(name, bin_counts, total))

    if not per_split:
        print("No valid COCO annotations found.")
        sys.exit(2)

    # Combined summary
    combined_counts: Counter = Counter()
    combined_total = 0
    for _, bc, tot in per_split:
        combined_counts.update(bc)
        combined_total += tot

    print("Combined:")
    print(format_summary("combined", combined_counts, combined_total))

    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "total", "sparse_0_9", "medium_10_29", "dense_30_plus"])
            for name, bc, tot in per_split:
                w.writerow([name, tot, bc["sparse"], bc["medium"], bc["dense"]])
            w.writerow(
                [
                    "combined",
                    combined_total,
                    combined_counts["sparse"],
                    combined_counts["medium"],
                    combined_counts["dense"],
                ]
            )
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
