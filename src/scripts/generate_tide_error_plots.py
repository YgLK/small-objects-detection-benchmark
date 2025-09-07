#!/usr/bin/env python
"""
Generate TIDE-style error distribution plots per model from tide_results.json.

Outputs:
- Per-model pie charts
- Per-model bar charts
- Per-model composite figure (pie + horizontal dAP bars + FP/FN bars)
- A combined stacked bar across models
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


CATEGORY_ORDER = ["Miss", "Loc", "Bkg", "Dupe", "Cls", "Both"]
PALETTE = sns.color_palette("Set2", n_colors=len(CATEGORY_ORDER))
CATEGORY_COLORS = {cat: PALETTE[i] for i, cat in enumerate(CATEGORY_ORDER)}
FPFN_COLORS = {"FP": sns.color_palette("deep")[0], "FN": sns.color_palette("deep")[1]}


def safe_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_model_errors(model_entry: dict) -> dict[str, float]:
    errs = model_entry.get("errors_main_dap", {})
    # Ensure all categories exist
    return {cat: float(errs.get(cat, 0.0)) for cat in CATEGORY_ORDER}


def get_special_errors(model_entry: dict) -> dict[str, float]:
    se = model_entry.get("errors_special_dap", {})
    return {
        "FalsePos": float(se.get("FalsePos", 0.0)),
        "FalseNeg": float(se.get("FalseNeg", 0.0)),
    }


def plot_pie(model_name: str, errors: dict[str, float], out_dir: str) -> str:
    labels = []
    sizes = []
    colors = []
    for cat in CATEGORY_ORDER:
        val = errors[cat]
        if val <= 0:
            continue
        labels.append(cat)
        sizes.append(val)
        colors.append(CATEGORY_COLORS[cat])

    total = sum(sizes) if sizes else 1.0
    autopct = lambda p: f"{p:.1f}%"  # noqa: E731

    plt.figure(figsize=(5.2, 5.2), dpi=150)
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=autopct,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
        textprops=dict(color="#222", fontsize=9),
    )
    plt.title(f"Error distribution (dAP) — {model_name}\nTotal dAP: {total:.2f}", fontsize=11)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_name(model_name)}__tide_error_pie.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_bar(model_name: str, errors: dict[str, float], out_dir: str) -> str:
    vals = [errors[c] for c in CATEGORY_ORDER]
    colors = [CATEGORY_COLORS[c] for c in CATEGORY_ORDER]

    plt.figure(figsize=(6.8, 3.2), dpi=150)
    bars = plt.bar(CATEGORY_ORDER, vals, color=colors, edgecolor="#222")
    for b, v in zip(bars, vals):
        if v > 0:
            plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("dAP (contribution)")
    plt.title(f"Error decomposition (dAP) — {model_name}", fontsize=11)
    plt.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_name(model_name)}__tide_error_bar.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_composite(
    model_name: str, errors_main: dict[str, float], errors_special: dict[str, float], out_dir: str
) -> str:
    """Create a TIDE-like composite: pie + horizontal bars + FP/FN bars."""
    # Prepare pie data
    pie_labels, pie_sizes, pie_colors = [], [], []
    for cat in CATEGORY_ORDER:
        val = errors_main[cat]
        if val > 0:
            pie_labels.append(cat)
            pie_sizes.append(val)
            pie_colors.append(CATEGORY_COLORS[cat])

    fig = plt.figure(figsize=(5.0, 7.2), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0])

    # Top: pie chart spanning both columns
    ax_pie = fig.add_subplot(gs[0, :])
    wedges, texts, autotexts = ax_pie.pie(
        pie_sizes,
        labels=pie_labels,
        colors=pie_colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
        textprops=dict(color="#111", fontsize=10),
    )
    total_dap = sum(pie_sizes) if pie_sizes else 0.0
    ax_pie.set_title(f"{model_name}\nTotal dAP: {total_dap:.2f}", fontsize=13, pad=10)

    # Bottom-left: horizontal bars for dAP per category
    ax_barh = fig.add_subplot(gs[1, 0])
    cats = CATEGORY_ORDER
    vals = [errors_main[c] for c in cats]
    colors = [CATEGORY_COLORS[c] for c in cats]
    ax_barh.barh(cats, vals, color=colors, edgecolor="#222")
    for y, v in enumerate(vals):
        if v > 0:
            ax_barh.text(v + max(vals) * 0.02 if max(vals) > 0 else v + 0.1, y, f"{v:.2f}", va="center", fontsize=8)
    ax_barh.set_xlabel("dAP")
    ax_barh.set_xlim(0, max(vals) * 1.35 if max(vals) > 0 else 1.0)
    ax_barh.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)

    # Bottom-right: FP/FN bars
    ax_fpfn = fig.add_subplot(gs[1, 1])
    fp = errors_special.get("FalsePos", 0.0)
    fn = errors_special.get("FalseNeg", 0.0)
    ax_fpfn.bar(["FP", "FN"], [fp, fn], color=[FPFN_COLORS["FP"], FPFN_COLORS["FN"]], edgecolor="#222")
    ax_fpfn.set_ylim(0, max(fp, fn) * 1.4 if max(fp, fn) > 0 else 1.0)
    for x, v in zip([0, 1], [fp, fn]):
        if v > 0:
            ax_fpfn.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax_fpfn.set_title("FP vs FN (dAP)", fontsize=10)
    ax_fpfn.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_name(model_name)}__tide_error_composite.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_stacked_bar(all_errors: dict[str, dict[str, float]], out_dir: str) -> str:
    # all_errors: model_name -> {cat -> value}
    models = list(all_errors.keys())
    ind = range(len(models))

    bottoms = [0.0] * len(models)
    plt.figure(figsize=(max(7.5, len(models) * 1.6), 4.2), dpi=150)

    for cat in CATEGORY_ORDER:
        vals = [all_errors[m][cat] for m in models]
        plt.bar(models, vals, bottom=bottoms, color=CATEGORY_COLORS[cat], edgecolor="#222", label=cat)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    plt.ylabel("dAP (contribution)")
    plt.title("Error decomposition by model (stacked dAP)", fontsize=12)
    plt.xticks(rotation=20, ha="right")
    plt.legend(ncol=min(6, len(CATEGORY_ORDER)), fontsize=8)
    plt.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "all_models__tide_error_stackedbar.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate TIDE error distribution plots from JSON")
    parser.add_argument("--input", required=True, help="Path to tide_results.json")
    parser.add_argument("--output", required=True, help="Directory to write plots into (will be created if missing)")
    args = parser.parse_args()

    data = load_data(args.input)
    models: dict[str, dict] = data.get("models", {})

    os.makedirs(args.output, exist_ok=True)

    all_errors: dict[str, dict[str, float]] = {}
    generated: list[str] = []

    for model_name, model_entry in models.items():
        errs = get_model_errors(model_entry)
        se = get_special_errors(model_entry)
        all_errors[model_name] = errs
        generated.append(plot_pie(model_name, errs, args.output))
        generated.append(plot_bar(model_name, errs, args.output))
        generated.append(plot_composite(model_name, errs, se, args.output))

    generated.append(plot_stacked_bar(all_errors, args.output))

    # Write an index.txt listing outputs for convenience
    index_path = os.path.join(args.output, "index.txt")
    with open(index_path, "w") as f:
        f.write("\n".join(os.path.basename(p) for p in generated))

    print(f"Wrote {len(generated)} plots to: {args.output}")


if __name__ == "__main__":
    main()
