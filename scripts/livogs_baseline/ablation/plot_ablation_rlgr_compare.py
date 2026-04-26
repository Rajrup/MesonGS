#!/usr/bin/env python3
"""
Compare RLGR ablation results across multiple datasets/sequences.

Generates:
  1. rlgr_latency_compare   – grouped bar chart (encode / decode) per variant, grouped by dataset
  2. rlgr_size_compare      – grouped bar chart of compressed size + overhead per variant
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

# ---------------------------------------------------------------------------
# Datasets to compare (label → path to ablation dir containing ablation_rlgr.csv)
# ---------------------------------------------------------------------------
DATASETS = {
    "HiFi4G\nA1-Greeting": "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/ablation/livogs_rlgr",
    "N3DV\nSear Steak": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_rlgr",
    "N3DV\nFlame Salmon": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_rlgr",
    "DB\nDr Johnson": "/synology/rajrup/MesonGS/train_output/db/drjohnson/ablation/livogs_rlgr",
    "T&T\nTruck": "/synology/rajrup/MesonGS/train_output/tandt/truck/ablation/livogs_rlgr",
}

VARIANT_ORDER = [
    "cpu", "gpu_full", "gpu_8192", "gpu_4096", "gpu_2048",
    "gpu_1024", "gpu_512", "gpu_256", "gpu_128",
]
VARIANT_LABELS = {
    "cpu": "CPU",
    "gpu_full": "GPU\n(full)",
    "gpu_8192": "GPU\n8192",
    "gpu_4096": "GPU\n4096",
    "gpu_2048": "GPU\n2048",
    "gpu_1024": "GPU\n1024",
    "gpu_512": "GPU\n512",
    "gpu_256": "GPU\n256",
    "gpu_128": "GPU\n128",
}

DATASET_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
DATASET_HATCHES = ["/", "\\", "x", ".", ".."]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("variant", "correct") else v)
                         for k, v in row.items()})
            rows[-1]["frame_id"] = int(rows[-1]["frame_id"])
    return rows


def aggregate_by_variant(rows):
    """Return {variant: {metric: mean_value}} for all numeric metrics."""
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    agg = {}
    metrics = [
        "rlgr_encode_ms", "rlgr_decode_ms",
        "transfer_to_cpu_ms", "transfer_to_gpu_ms",
        "pure_rlgr_encode_ms", "pure_rlgr_decode_ms",
        "compressed_size_bytes",
    ]
    for v, vrows in by_variant.items():
        agg[v] = {}
        for m in metrics:
            vals = [r[m] for r in vrows if m in r]
            agg[v][m] = np.mean(vals) if vals else 0.0
            agg[v][f"{m}_std"] = np.std(vals) if vals else 0.0
    return agg


# ---------------------------------------------------------------------------
# Figure 1: Latency comparison (2 rows x 1 col: encode, decode)
# ---------------------------------------------------------------------------
def plot_latency_compare(all_data, dataset_labels, output_folder, fmt):
    variants = [v for v in VARIANT_ORDER if any(v in d for d in all_data.values())]
    n_variants = len(variants)
    n_datasets = len(dataset_labels)
    bar_width = 0.8 / n_datasets
    x = np.arange(n_variants)

    fig, (ax_enc, ax_dec) = plt.subplots(2, 1, figsize=(max(12, n_variants * 1.5), 9),
                                          sharex=True)

    for di, ds_label in enumerate(dataset_labels):
        agg = all_data[ds_label]
        offset = (di - n_datasets / 2 + 0.5) * bar_width
        color = DATASET_COLORS[di % len(DATASET_COLORS)]

        pure_enc = [agg.get(v, {}).get("pure_rlgr_encode_ms", 0) for v in variants]
        xfer_enc = [agg.get(v, {}).get("transfer_to_cpu_ms", 0) for v in variants]
        total_enc = [p + t for p, t in zip(pure_enc, xfer_enc)]

        ax_enc.bar(x + offset, pure_enc, bar_width, color=color, alpha=0.85,
                   label=ds_label.replace("\n", " "), edgecolor="white")
        ax_enc.bar(x + offset, xfer_enc, bar_width, bottom=pure_enc,
                   color=color, alpha=0.35, hatch="//", edgecolor="white")

        for i, tot in enumerate(total_enc):
            if tot > 0:
                ax_enc.annotate(f"{tot:.1f}", xy=(x[i] + offset, tot + 8),
                                ha="center", va="bottom", fontsize=8, rotation=90)

        pure_dec = [agg.get(v, {}).get("pure_rlgr_decode_ms", 0) for v in variants]
        xfer_dec = [agg.get(v, {}).get("transfer_to_gpu_ms", 0) for v in variants]
        total_dec = [p + t for p, t in zip(pure_dec, xfer_dec)]

        ax_dec.bar(x + offset, pure_dec, bar_width, color=color, alpha=0.85,
                   edgecolor="white")
        ax_dec.bar(x + offset, xfer_dec, bar_width, bottom=pure_dec,
                   color=color, alpha=0.35, hatch="//", edgecolor="white")

        for i, tot in enumerate(total_dec):
            if tot > 0:
                ax_dec.annotate(f"{tot:.1f}", xy=(x[i] + offset, tot + 8),
                                ha="center", va="bottom", fontsize=8, rotation=90)

    # Proxy patches for transfer legend entries
    from matplotlib.patches import Patch
    xfer_enc_patch = Patch(facecolor="gray", alpha=0.35, hatch="//",
                           edgecolor="white", label="GPU→CPU transfer")
    xfer_dec_patch = Patch(facecolor="gray", alpha=0.35, hatch="//",
                           edgecolor="white", label="CPU→GPU transfer")

    for ax, title in [(ax_enc, "Encode Latency (ms)"), (ax_dec, "Decode Latency (ms)")]:
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 550)
        ax.grid(True, axis="y", alpha=0.3)

    ax_dec.set_xticks(x)
    ax_dec.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants],
                            fontsize=9, ha="center")

    handles, labels = ax_enc.get_legend_handles_labels()
    handles.append(xfer_enc_patch)
    labels.append("GPU→CPU transfer")
    fig.legend(handles, labels, loc="upper center",
               ncol=n_datasets + 1, fontsize=10, bbox_to_anchor=(0.5, 0.94))

    ax_dec.legend([xfer_dec_patch], ["CPU→GPU transfer"],
                  loc="upper right", fontsize=9)

    fig.suptitle("RLGR Ablation: Latency Comparison Across Datasets",
                 fontsize=14, y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(output_folder, f"rlgr_latency_compare.{fmt}")
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Compressed size comparison (1x2: absolute size + overhead vs CPU)
# ---------------------------------------------------------------------------
def plot_size_compare(all_data, dataset_labels, output_folder, fmt):
    variants = [v for v in VARIANT_ORDER if any(v in d for d in all_data.values())]
    n_variants = len(variants)
    n_datasets = len(dataset_labels)
    bar_width = 0.8 / n_datasets
    x = np.arange(n_variants)

    fig, (ax_size, ax_oh) = plt.subplots(1, 2, figsize=(max(14, n_variants * 1.8), 5.5))

    for di, ds_label in enumerate(dataset_labels):
        agg = all_data[ds_label]
        offset = (di - n_datasets / 2 + 0.5) * bar_width
        color = DATASET_COLORS[di % len(DATASET_COLORS)]

        means_mb = [agg.get(v, {}).get("compressed_size_bytes", 0) / (1024 * 1024) for v in variants]
        stds_mb = [agg.get(v, {}).get("compressed_size_bytes_std", 0) / (1024 * 1024) for v in variants]

        ax_size.bar(x + offset, means_mb, bar_width, yerr=stds_mb,
                    color=color, alpha=0.85, capsize=2, edgecolor="white",
                    label=ds_label.replace("\n", " "))

        cpu_size = agg.get("cpu", {}).get("compressed_size_bytes", 1)
        overhead = [(agg.get(v, {}).get("compressed_size_bytes", 0) - cpu_size) / cpu_size * 100
                    if cpu_size > 0 else 0 for v in variants]

        bars_oh = ax_oh.bar(x + offset, overhead, bar_width, color=color, alpha=0.7,
                            edgecolor="white")

        for i, (bar, pct) in enumerate(zip(bars_oh, overhead)):
            y_top = max(bar.get_height(), 0) + 0.3
            ax_oh.annotate(f"{pct:.1f}", xy=(x[i] + offset, y_top),
                           ha="center", va="bottom", fontsize=10, rotation=90)

    ax_size.set_xticks(x)
    ax_size.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants],
                             fontsize=9, ha="center")
    ax_size.set_ylabel("Compressed Size (MB)", fontsize=11)
    ax_size.set_title("Attribute Compressed Size", fontsize=12)
    ax_size.grid(True, axis="y", alpha=0.3)
    ax_size.set_ylim(bottom=0)

    ax_oh.axhline(0, color="black", linewidth=0.8)
    ax_oh.set_xticks(x)
    ax_oh.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants],
                           fontsize=9, ha="center")
    ax_oh.set_ylabel("Compression Overhead (%)", fontsize=11)
    ax_oh.set_title("Size Overhead vs CPU Baseline", fontsize=12)
    ax_oh.set_ylim(0, 10)
    ax_oh.grid(True, axis="y", alpha=0.3)

    handles, labels = ax_size.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_datasets,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("RLGR Ablation: Compressed Size Comparison Across Datasets",
                 fontsize=14, y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(output_folder, f"rlgr_size_compare.{fmt}")
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Compare RLGR ablation results across datasets"
    )
    p.add_argument("--output_folder", type=str,
                   default=str(Path(__file__).resolve().parent / "plots"))
    p.add_argument("--format", type=str, choices=["pdf", "png"], default="png")
    args = p.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    all_data = {}
    dataset_labels = []
    for label, ablation_dir in DATASETS.items():
        csv_path = os.path.join(ablation_dir, "ablation_rlgr.csv")
        if not os.path.isfile(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping {label}")
            continue
        rows = load_csv(csv_path)
        print(f"Loaded {len(rows)} rows for {label.replace(chr(10), ' ')} from {csv_path}")
        all_data[label] = aggregate_by_variant(rows)
        dataset_labels.append(label)

    if not dataset_labels:
        print("No data found!")
        return

    print(f"\nComparing {len(dataset_labels)} datasets")
    plot_latency_compare(all_data, dataset_labels, args.output_folder, args.format)
    plot_size_compare(all_data, dataset_labels, args.output_folder, args.format)
    print("Done!")


if __name__ == "__main__":
    main()

'''
python scripts/livogs_baseline/ablation/plot_ablation_rlgr_compare.py --format png
python scripts/livogs_baseline/ablation/plot_ablation_rlgr_compare.py --format pdf
'''
