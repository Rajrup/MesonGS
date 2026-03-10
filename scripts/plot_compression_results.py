#!/usr/bin/env python3
"""Publication-quality plots for static 3DGS compression benchmarks.

Generates:
  1. PSNR vs Compressed Size (1x2, one subplot per dataset)
  2. SSIM vs Compressed Size (1x2, one subplot per dataset)
  3. Latency bar chart per dataset (1x2: encode left, decode right)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_OUTPUT_ROOT = "/synology/rajrup/MesonGS/train_output"

PIPELINES = ["livogs", "imagegs", "dracogs", "mesongs_prune", "mesongs_noprune"]
PIPELINE_LABELS = {
    "livogs": "LiVoGS",
    "imagegs": "ImageGS",
    "dracogs": "DracoGS",
    "mesongs_prune": "MesonGS (prune)",
    "mesongs_noprune": "MesonGS (no prune)",
}
PIPELINE_COLORS = {
    "livogs": "#1f77b4",
    "imagegs": "#ff7f0e",
    "dracogs": "#2ca02c",
    "mesongs_prune": "#d62728",
    "mesongs_noprune": "#8b1a1a",
}
PIPELINE_HATCHES = {
    "livogs": "/",
    "imagegs": "\\",
    "dracogs": "x",
    "mesongs_prune": ".",
    "mesongs_noprune": "..",
}
PIPELINE_MARKERS = {
    "livogs": "^",
    "imagegs": "P",
    "dracogs": "o",
    "mesongs_prune": "D",
    "mesongs_noprune": "s",
}

DATASETS = [
    {"dataset": "db", "scene": "drjohnson", "title": "Deep Blending / Dr Johnson",
     "psnr_ylim": (30, 36), "ssim_ylim": (0.9, 1.0)},
    {"dataset": "tandt", "scene": "truck", "title": "Tanks & Temples / Truck",
     "psnr_ylim": (20, 24), "ssim_ylim": (0.8, 0.9), "size_xlim": (0, 250)},
]

# ---------------------------------------------------------------------------
# Per-method configuration (determines exact output directory names)
# ---------------------------------------------------------------------------
MESONGS_PRUNE_CONFIG = {
    "drjohnson": {"config": "config3", "prune": "yes"},
    "truck": {"config": "config3", "prune": "yes"},
}
MESONGS_NO_PRUNE_CONFIG = {
    "drjohnson": {"config": "config3", "prune": "no"},
    "truck": {"config": "config3", "prune": "no"},
}
LIVOGS_CONFIG = {
    "drjohnson": {"J": 14, "qpq": 0.005, "qps": 0.001, "qpo": 0.01,
                  "qpdc": 0.01, "qpac": 0.05, "sh_color_space": "klt", "nvcomp": "ANS"},
    "truck": {"J": 15, "qpq": 0.005, "qps": 0.001, "qpo": 0.01,
              "qpdc": 0.01, "qpac": 0.05, "sh_color_space": "klt", "nvcomp": "ANS"},
}
DRACOGS_CONFIG = {
    "drjohnson": {"EG": 16, "EO": 16, "ET": 16, "ES": 16, "CL": 10},
    "truck": {"EG": 16, "EO": 16, "ET": 16, "ES": 16, "CL": 10},
}
IMAGEGS_CONFIG = {
    "drjohnson": {"config": "default_sort", "sh_cluster": 65536},
    "truck": {"config": "default_sort", "sh_cluster": 65536},
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _build_result_path(pipeline: str, scene: str) -> str | None:
    """Build the exact subdirectory name for a pipeline/scene from config dicts."""
    if pipeline == "mesongs_prune":
        c = MESONGS_PRUNE_CONFIG.get(scene)
        if not c:
            return None
        return os.path.join("mesongs", f"{c['config']}_prune_{c['prune']}")
    elif pipeline == "mesongs_noprune":
        c = MESONGS_NO_PRUNE_CONFIG.get(scene)
        if not c:
            return None
        return os.path.join("mesongs", f"{c['config']}_prune_{c['prune']}")
    elif pipeline == "livogs":
        c = LIVOGS_CONFIG.get(scene)
        if not c:
            return None
        return os.path.join("livogs",
            f"J_{c['J']}_qpq_{c['qpq']}_qps_{c['qps']}_qpo_{c['qpo']}"
            f"_qpdc_{c['qpdc']}_qpac_{c['qpac']}_{c['sh_color_space']}_nvcomp_{c['nvcomp']}")
    elif pipeline == "dracogs":
        c = DRACOGS_CONFIG.get(scene)
        if not c:
            return None
        return os.path.join("dracogs",
            f"eg_{c['EG']}_eo_{c['EO']}_et_{c['ET']}_es_{c['ES']}_cl_{c['CL']}")
    elif pipeline == "imagegs":
        c = IMAGEGS_CONFIG.get(scene)
        if not c:
            return None
        return os.path.join("imagegs", f"{c['config']}_sh_cluster_{c['sh_cluster']}")
    return None


def find_result_dir(base_compression_dir: str, pipeline: str, scene: str) -> str | None:
    """Build the exact result directory path from config and verify it exists."""
    rel = _build_result_path(pipeline, scene)
    if rel is None:
        return None
    result_dir = os.path.join(base_compression_dir, rel)
    comp_stats = os.path.join(result_dir, "compression_stats.json")
    eval_results = os.path.join(result_dir, "evaluation", "evaluation_results.json")
    if os.path.exists(comp_stats) and os.path.exists(eval_results):
        return result_dir
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results(result_dir: str) -> dict[str, Any] | None:
    """Load compression stats and evaluation results from a result directory."""
    comp_path = os.path.join(result_dir, "compression_stats.json")
    eval_path = os.path.join(result_dir, "evaluation", "evaluation_results.json")
    try:
        with open(comp_path) as f:
            comp = json.load(f)
        with open(eval_path) as f:
            evl = json.load(f)
        return {
            "compressed_size_mb": comp["compressed_size_bytes"] / (1024 * 1024),
            "uncompressed_size_mb": comp["uncompressed_size_bytes"] / (1024 * 1024),
            "encode_time_ms": comp["encode_time_ms"],
            "decode_time_ms": comp["decode_time_ms"],
            "psnr": evl["decompressed"]["psnr"],
            "ssim": evl["decompressed"]["ssim"],
            "lpips": evl["decompressed"]["lpips"],
            "gt_psnr": evl["gt"]["psnr"],
            "gt_ssim": evl["gt"]["ssim"],
            "config_dir": os.path.basename(result_dir),
        }
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARNING: Failed to load {result_dir}: {e}")
        return None


def gather_all_results() -> dict[str, dict[str, dict[str, Any]]]:
    """Gather results for all pipelines and datasets."""
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    for ds in DATASETS:
        ds_key = f"{ds['dataset']}/{ds['scene']}"
        base = os.path.join(TRAIN_OUTPUT_ROOT, ds["dataset"], ds["scene"], "compression")
        all_results[ds_key] = {}

        for pipeline in PIPELINES:
            result_dir = find_result_dir(base, pipeline, ds["scene"])
            if result_dir is None:
                print(f"  [{ds_key}] {PIPELINE_LABELS[pipeline]}: no results found")
                continue
            data = load_results(result_dir)
            if data is not None:
                all_results[ds_key][pipeline] = data
                print(f"  [{ds_key}] {PIPELINE_LABELS[pipeline]}: {data['config_dir']}  "
                      f"PSNR={data['psnr']:.2f}  Size={data['compressed_size_mb']:.2f} MB  "
                      f"Enc={data['encode_time_ms']:.0f} ms  Dec={data['decode_time_ms']:.0f} ms")
    return all_results


# ---------------------------------------------------------------------------
# Figure 1 & 2: Quality vs Compressed Size (1x2 scatter)
# ---------------------------------------------------------------------------
def plot_quality_vs_size(
    all_results: dict,
    metric: str,
    ylabel: str,
    output_path: str,
) -> None:
    """Plot quality metric vs compressed size (1x2 subplots, one per dataset)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    all_compressed: list[float] = []
    for ds in DATASETS:
        ds_key = f"{ds['dataset']}/{ds['scene']}"
        for r in all_results.get(ds_key, {}).values():
            all_compressed.append(r["compressed_size_mb"])

    x_min = min(all_compressed) * 0.8 if all_compressed else 0
    x_max = max(all_compressed) * 1.15 if all_compressed else 500

    ylim_key = f"{metric}_ylim"

    for ax_idx, ds in enumerate(DATASETS):
        ax = axes[ax_idx]
        ds_key = f"{ds['dataset']}/{ds['scene']}"
        ds_results = all_results.get(ds_key, {})

        for pipeline in PIPELINES:
            if pipeline not in ds_results:
                continue
            r = ds_results[pipeline]
            ax.scatter(
                r["compressed_size_mb"], r[metric],
                label=PIPELINE_LABELS[pipeline],
                color=PIPELINE_COLORS[pipeline],
                marker=PIPELINE_MARKERS[pipeline],
                s=120, zorder=5, edgecolors="white", linewidths=0.5,
            )
            ax.annotate(
                f'  {r[metric]:.2f}',
                (r["compressed_size_mb"], r[metric]),
                fontsize=8, color=PIPELINE_COLORS[pipeline], va="bottom",
            )

        if ds_results:
            uncomp_mb = next(iter(ds_results.values()))["uncompressed_size_mb"]
            gt_val = next(iter(ds_results.values())).get(f"gt_{metric}")
            if gt_val is not None:
                ax.axhline(y=gt_val, color="gray", linestyle="--",
                           linewidth=1, alpha=0.7,
                           label=f"GT (uncompressed, {uncomp_mb:.0f} MB)")

        ax.set_xlabel("Compressed Size (MB)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ds["title"], fontsize=13)
        ax.grid(True, alpha=0.3)
        if "size_xlim" in ds:
            ax.set_xlim(ds["size_xlim"])
        else:
            ax.set_xlim(x_min, x_max)
        if ylim_key in ds:
            ax.set_ylim(ds[ylim_key])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(PIPELINES) + 1,
               fontsize=10, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3 & 4: Latency bar charts (1x2: encode, decode)
# ---------------------------------------------------------------------------
def plot_latency(
    all_results: dict,
    ds: dict,
    output_path: str,
) -> None:
    """Plot encoding/decoding latency bar chart (1x2) for a single dataset."""
    ds_key = f"{ds['dataset']}/{ds['scene']}"
    ds_results = all_results.get(ds_key, {})

    present: list[str] = [p for p in PIPELINES if p in ds_results]
    if not present:
        print(f"  WARNING: No latency data for {ds_key}")
        return

    n_bars = len(present)
    x = np.arange(n_bars)
    bar_width = 0.6

    enc_times = [ds_results[p]["encode_time_ms"] for p in present]
    dec_times = [ds_results[p]["decode_time_ms"] for p in present]
    labels = [PIPELINE_LABELS[p] for p in present]
    colors = [PIPELINE_COLORS[p] for p in present]
    hatches = [PIPELINE_HATCHES[p] for p in present]

    fig, (ax_enc, ax_dec) = plt.subplots(1, 2, figsize=(16, 5))

    def _draw_bars(ax: plt.Axes, values: list[float], title: str) -> None:
        bars = ax.bar(
            x, values, bar_width,
            color=colors, edgecolor="white", alpha=0.85,
        )
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, which="both")

        lines = []
        for p, val in zip(present, values):
            lines.append(f"{PIPELINE_LABELS[p]:>22s}: {val:>10.1f} ms")
        ax.annotate(
            "\n".join(lines),
            xy=(0.98, 0.97), xycoords="axes fraction",
            fontsize=8, va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
        )

    _draw_bars(ax_enc, enc_times, f"Encode Latency — {ds['title']}")
    _draw_bars(ax_dec, dec_times, f"Decode Latency — {ds['title']}")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PIPELINE_COLORS[p],
                       hatch=PIPELINE_HATCHES[p], edgecolor="white", alpha=0.85)
        for p in present
    ]
    fig.legend(handles, [PIPELINE_LABELS[p] for p in present],
               loc="upper center", ncol=n_bars,
               fontsize=11, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Plot static 3DGS compression benchmark results")
    p.add_argument("--output_dir", type=str,
                   default=str(Path(__file__).resolve().parent / "plots"),
                   help="Directory to save plots")
    p.add_argument("--format", type=str, choices=["pdf", "png"], default="pdf",
                   help="Output image format (default: pdf)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fmt = args.format

    print(f"Output format: {fmt}")
    print(f"Output folder: {args.output_dir}")
    print()

    print("Gathering results...")
    all_results = gather_all_results()

    print("\nGenerating quality vs size plots...")
    plot_quality_vs_size(all_results, "psnr", "PSNR (dB)",
                         os.path.join(args.output_dir, f"psnr_vs_size.{fmt}"))
    plot_quality_vs_size(all_results, "ssim", "SSIM",
                         os.path.join(args.output_dir, f"ssim_vs_size.{fmt}"))

    print("\nGenerating latency plots...")
    for ds in DATASETS:
        tag = f"{ds['dataset']}_{ds['scene']}"
        plot_latency(all_results, ds,
                     os.path.join(args.output_dir, f"latency_{tag}.{fmt}"))

    print("\nDone!")


if __name__ == "__main__":
    main()
