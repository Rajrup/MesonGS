#!/usr/bin/env python3
"""
Plot RLGR ablation results from ablation_rlgr.csv.

Generates:
  1. rlgr_latency   – 1x2 grouped bar chart (encode / decode) with transfer breakdown
  2. rlgr_compressed_size – line plot of compressed size per frame per variant
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANT_ORDER = [
    "cpu", "gpu_full", "gpu_8192", "gpu_4096", "gpu_2048",
    "gpu_1024", "gpu_512", "gpu_256", "gpu_128", "gpu_64", "gpu_32",
]
VARIANT_LABELS = {
    "cpu": "CPU",
    "gpu_full": "GPU (full)",
    "gpu_8192": "GPU 8192",
    "gpu_4096": "GPU 4096",
    "gpu_2048": "GPU 2048",
    "gpu_1024": "GPU 1024",
    "gpu_512": "GPU 512",
    "gpu_256": "GPU 256",
    "gpu_128": "GPU 128",
    "gpu_64": "GPU 64",
    "gpu_32": "GPU 32",
}
VARIANT_COLORS = {
    "cpu": "#d62728",
    "gpu_full": "#1f77b4",
    "gpu_8192": "#ff7f0e",
    "gpu_4096": "#2ca02c",
    "gpu_2048": "#9467bd",
    "gpu_1024": "#8c564b",
    "gpu_512": "#e377c2",
    "gpu_256": "#7f7f7f",
    "gpu_128": "#bcbd22",
    "gpu_64": "#17becf",
    "gpu_32": "#aec7e8",
}


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k != "variant" and k != "correct" else v)
                         for k, v in row.items()})
            rows[-1]["frame_id"] = int(rows[-1]["frame_id"])
            rows[-1]["correct"] = rows[-1]["correct"] == "True"
    return rows


def plot_latency(rows, output_folder, fmt):
    """Grouped bar chart: encode and decode latency with transfer breakdown."""
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    variants = [v for v in VARIANT_ORDER if v in by_variant]
    n = len(variants)
    x = np.arange(n)

    fig, (ax_enc, ax_dec) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Encode ---
    pure_enc = [np.mean([r["pure_rlgr_encode_ms"] for r in by_variant[v]]) for v in variants]
    xfer_enc = [np.mean([r["transfer_to_cpu_ms"] for r in by_variant[v]]) for v in variants]
    colors = [VARIANT_COLORS[v] for v in variants]

    ax_enc.bar(x, pure_enc, color=colors, alpha=0.85, label="RLGR encode")
    ax_enc.bar(x, xfer_enc, bottom=pure_enc, color=colors, alpha=0.4,
               hatch="//", label="GPU→CPU transfer")
    ax_enc.set_xticks(x)
    ax_enc.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=45, ha="right")
    ax_enc.set_ylabel("Time (ms)")
    ax_enc.set_title("Encode Latency")
    ax_enc.grid(True, axis="y", alpha=0.3)
    ax_enc.legend(fontsize=9)

    for i, v in enumerate(variants):
        total = pure_enc[i] + xfer_enc[i]
        ax_enc.annotate(f"{total:.1f}", xy=(i, total), ha="center", va="bottom", fontsize=8)

    # --- Decode ---
    pure_dec = [np.mean([r["pure_rlgr_decode_ms"] for r in by_variant[v]]) for v in variants]
    xfer_dec = [np.mean([r["transfer_to_gpu_ms"] for r in by_variant[v]]) for v in variants]

    ax_dec.bar(x, pure_dec, color=colors, alpha=0.85, label="RLGR decode")
    ax_dec.bar(x, xfer_dec, bottom=pure_dec, color=colors, alpha=0.4,
               hatch="//", label="CPU→GPU transfer")
    ax_dec.set_xticks(x)
    ax_dec.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=45, ha="right")
    ax_dec.set_ylabel("Time (ms)")
    ax_dec.set_title("Decode Latency")
    ax_dec.grid(True, axis="y", alpha=0.3)
    ax_dec.legend(fontsize=9)

    for i, v in enumerate(variants):
        total = pure_dec[i] + xfer_dec[i]
        ax_dec.annotate(f"{total:.1f}", xy=(i, total), ha="center", va="bottom", fontsize=8)

    fig.suptitle("RLGR Ablation: Encode / Decode Latency", fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(output_folder, f"rlgr_latency.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_compressed_size(rows, output_folder, fmt):
    """1x2: Left = bar+whisker of compressed size, Right = overhead % vs CPU baseline."""
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    variants = [v for v in VARIANT_ORDER if v in by_variant]
    n = len(variants)
    x = np.arange(n)
    colors = [VARIANT_COLORS[v] for v in variants]

    sizes_per_variant = {
        v: np.array([r["compressed_size_bytes"] for r in by_variant[v]])
        for v in variants
    }
    means_mb = [sizes_per_variant[v].mean() / (1024 * 1024) for v in variants]
    stds_mb = [sizes_per_variant[v].std() / (1024 * 1024) for v in variants]

    baseline_key = "cpu" if "cpu" in sizes_per_variant else variants[0]
    baseline_sizes = sizes_per_variant[baseline_key]

    overhead_pcts = []
    overhead_stds = []
    for v in variants:
        per_frame_pct = (sizes_per_variant[v] - baseline_sizes) / baseline_sizes * 100
        overhead_pcts.append(per_frame_pct.mean())
        overhead_stds.append(per_frame_pct.std())

    fig, (ax_size, ax_oh) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Compressed size bar + whisker ---
    ax_size.bar(x, means_mb, yerr=stds_mb, color=colors, alpha=0.85,
                capsize=4, edgecolor="black", linewidth=0.5)
    ax_size.set_xticks(x)
    ax_size.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=45, ha="right")
    ax_size.set_ylabel("Compressed Size (MB)")
    ax_size.set_title("Attribute Compressed Size")
    ax_size.grid(True, axis="y", alpha=0.3)
    ax_size.set_ylim(bottom=0)
    for i in range(n):
        ax_size.annotate(f"{means_mb[i]:.3f}", xy=(i, means_mb[i] + stds_mb[i]),
                         ha="center", va="bottom", fontsize=8)

    # --- Right: Overhead % vs CPU baseline ---
    bar_colors_oh = []
    for pct in overhead_pcts:
        bar_colors_oh.append("#999999" if abs(pct) < 0.001 else "#e74c3c" if pct > 0 else "#2ecc71")
    ax_oh.bar(x, overhead_pcts, yerr=overhead_stds, color=bar_colors_oh, alpha=0.85,
              capsize=4, edgecolor="black", linewidth=0.5)
    ax_oh.axhline(0, color="black", linewidth=0.8)
    ax_oh.set_xticks(x)
    ax_oh.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=45, ha="right")
    ax_oh.set_ylabel(f"Overhead vs {VARIANT_LABELS[baseline_key]} (%)")
    ax_oh.set_title("Size Overhead")
    ax_oh.grid(True, axis="y", alpha=0.3)
    for i in range(n):
        va = "bottom" if overhead_pcts[i] >= 0 else "top"
        y_pos = overhead_pcts[i] + (overhead_stds[i] if overhead_pcts[i] >= 0 else -overhead_stds[i])
        ax_oh.annotate(f"{overhead_pcts[i]:+.2f}%", xy=(i, y_pos),
                       ha="center", va=va, fontsize=8)

    fig.suptitle("RLGR Ablation: Compressed Size", fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(output_folder, f"rlgr_compressed_size.{fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    p = argparse.ArgumentParser(description="Plot RLGR ablation results")
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--output_folder", type=str, default=None)
    p.add_argument("--format", type=str, choices=["pdf", "png"], default="png")
    args = p.parse_args()

    if args.output_folder is None:
        args.output_folder = os.path.join(os.path.dirname(args.input_csv), "plots")

    os.makedirs(args.output_folder, exist_ok=True)
    rows = load_csv(args.input_csv)
    print(f"Loaded {len(rows)} rows from {args.input_csv}")

    plot_latency(rows, args.output_folder, args.format)
    plot_compressed_size(rows, args.output_folder, args.format)
    print("Done!")


if __name__ == "__main__":
    main()

'''
python scripts/livogs_baseline/ablation/plot_ablation_rlgr.py --input_csv /synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_rlgr/ablation_rlgr.csv --output_folder scripts/livogs_baseline/ablation/plots --format png
python scripts/livogs_baseline/ablation/plot_ablation_rlgr.py --input_csv /synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_rlgr/ablation_rlgr.csv --output_folder scripts/livogs_baseline/ablation/plots --format png
'''