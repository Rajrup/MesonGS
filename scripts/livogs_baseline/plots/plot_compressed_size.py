"""
Plot LiVoGS compressed size breakdown and point counts per frame.
Requires: benchmark_livogs.csv under input_folder/<config_name>/

Generates two plots:
  1. Size breakdown: uncompressed, position compressed, attribute compressed, total compressed
  2. Point counts: original points vs voxelized points
"""
import os
import re
import argparse
import csv
import matplotlib.pyplot as plt

from typing import Any

def load_benchmark_csv(path):
    rows = []
    _per_dim_re = re.compile(r"^(quats|scales|opacity|sh)_dim(\d+)_compressed_bytes$")
    with open(path) as f:
        r = csv.DictReader(f)
        per_dim_cols = [c for c in (r.fieldnames or []) if _per_dim_re.match(c)]
        for row in r:
            parsed: dict[str, Any] = {
                "frame": int(row["frame_id"]),
                "uncompressed_size_bytes": int(row["uncompressed_size_bytes"]),
                "position_compressed_bytes": int(row["position_compressed_bytes"]),
                "attribute_compressed_bytes": int(row["attribute_compressed_bytes"]),
                "compressed_size_bytes": int(row["compressed_size_bytes"]),
                "original_points": int(row["original_points"]),
                "voxelized_points": int(row["voxelized_points"]),
            }
            # Read all per-dim columns
            for col in per_dim_cols:
                parsed[col] = int(row.get(col, 0))
            # Compute aggregated values from per-dim columns
            parsed["quats_total"] = sum(
                parsed[col] for col in per_dim_cols
                if col.startswith("quats_dim")
            )
            parsed["scales_total"] = sum(
                parsed[col] for col in per_dim_cols
                if col.startswith("scales_dim")
            )
            parsed["opacity_total"] = sum(
                parsed[col] for col in per_dim_cols
                if col.startswith("opacity_dim")
            )
            # sh_dim0..2 are DC (3 color channels), rest are SH rest
            sh_cols = sorted(
                [c for c in per_dim_cols if c.startswith("sh_dim")],
                key=lambda c: int(_per_dim_re.match(c).group(2)),  # type: ignore[union-attr]
            )
            parsed["sh_dc_total"] = sum(parsed[c] for c in sh_cols[:3])
            parsed["sh_rest_total"] = sum(parsed[c] for c in sh_cols[3:])
            rows.append(parsed)
    return sorted(rows, key=lambda x: x["frame"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Base folder for livogs_compression")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g. Neural_3D_Video)")
    parser.add_argument("--sequence_name", type=str, required=True, help="Sequence name (e.g. cook_spinach)")
    parser.add_argument("--j", type=int, required=True, help="Octree depth J (e.g. 15)")
    parser.add_argument("--qstep", type=str, required=True, help="Quantization step (e.g. 0.0001)")
    parser.add_argument("--sh_color_space", type=str, required=True, help="Color space (e.g. klt)")
    parser.add_argument("--nvcomp", type=str, default="ANS", help="nvCOMP algorithm (e.g. ANS, None)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder for plot PNG")
    args = parser.parse_args()

    config_name = f"J_{args.j}_qstep_{args.qstep}_{args.sh_color_space}_nvcomp_{args.nvcomp}"
    config_dir = os.path.join(args.input_folder, config_name)
    csv_path = os.path.join(config_dir, "benchmark_livogs.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, config_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise SystemExit(f"Required file not found: {csv_path}")

    rows = load_benchmark_csv(csv_path)
    n = len(rows)
    frame_ids = [r["frame"] for r in rows]
    x = range(n)
    tick_every = 10

    # ---- Plot 1: Size breakdown ----
    uncompressed_mb = [r["uncompressed_size_bytes"] / 1024 / 1024 for r in rows]
    pos_mb = [r["position_compressed_bytes"] / 1024 / 1024 for r in rows]
    attr_mb = [r["attribute_compressed_bytes"] / 1024 / 1024 for r in rows]
    total_mb = [r["compressed_size_bytes"] / 1024 / 1024 for r in rows]

    avg_unc = sum(uncompressed_mb) / n
    avg_tot = sum(total_mb) / n
    avg_pos = sum(pos_mb) / n
    avg_attr = sum(attr_mb) / n

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, uncompressed_mb, "o-", label=f"Uncompressed (avg {avg_unc:.2f} MB)", color="gray", markersize=2, alpha=0.7)
    ax.plot(x, total_mb, "s-", label=f"Total compressed (avg {avg_tot:.2f} MB)", color="coral", markersize=2)
    ax.plot(x, attr_mb, "^-", label=f"Attribute compressed (avg {avg_attr:.2f} MB)", color="steelblue", markersize=2)
    ax.plot(x, pos_mb, "v-", label=f"Position compressed (avg {avg_pos:.2f} MB)", color="seagreen", markersize=2)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Size (MB)")
    ax.set_title(f"LiVoGS size breakdown per frame [{config_name}, {args.dataset_name}, {args.sequence_name}]")
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x)[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    size_path = os.path.join(out_dir, "compressed_size.png")
    fig.savefig(size_path, dpi=150)
    print(f"Saved: {size_path}")
    plt.close(fig)

    # ---- Plot 1b: Per-attribute size breakdown (stacked area) ----
    has_breakdown = any(r["quats_total"] > 0 for r in rows)
    if has_breakdown:
        pos_mb_s = [r["position_compressed_bytes"] / 1024 / 1024 for r in rows]
        quats_mb = [r["quats_total"] / 1024 / 1024 for r in rows]
        scales_mb = [r["scales_total"] / 1024 / 1024 for r in rows]
        opacity_mb = [r["opacity_total"] / 1024 / 1024 for r in rows]
        sh_dc_mb = [r["sh_dc_total"] / 1024 / 1024 for r in rows]
        sh_rest_mb = [r["sh_rest_total"] / 1024 / 1024 for r in rows]

        avg_quats = sum(quats_mb) / n
        avg_scales = sum(scales_mb) / n
        avg_opacity = sum(opacity_mb) / n
        avg_sh_dc = sum(sh_dc_mb) / n
        avg_sh_rest = sum(sh_rest_mb) / n
        avg_pos_s = sum(pos_mb_s) / n

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.stackplot(
            x,
            pos_mb_s, quats_mb, scales_mb, opacity_mb, sh_dc_mb, sh_rest_mb,
            labels=[
                f"Position (avg {avg_pos_s:.3f} MB)",
                f"Quats (avg {avg_quats:.3f} MB)",
                f"Scales (avg {avg_scales:.3f} MB)",
                f"Opacity (avg {avg_opacity:.3f} MB)",
                f"SH DC (avg {avg_sh_dc:.3f} MB)",
                f"SH Rest (avg {avg_sh_rest:.3f} MB)",
            ],
            colors=["seagreen", "steelblue", "coral", "goldenrod", "mediumpurple", "lightcoral"],
            alpha=0.85,
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("Size (MB)")
        ax.set_title(f"LiVoGS per-attribute size breakdown [{config_name}, {args.dataset_name}, {args.sequence_name}]")
        ax.set_ylim(bottom=0)
        ax.set_xticks(list(x)[::tick_every])
        ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        attr_path = os.path.join(out_dir, "compressed_size_breakdown.png")
        fig.savefig(attr_path, dpi=150)
        print(f"Saved: {attr_path}")
        plt.close(fig)

    # ---- Plot 2: Point counts ----
    orig_pts = [r["original_points"] for r in rows]
    vox_pts = [r["voxelized_points"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, orig_pts, "o-", label="Original points", color="steelblue", markersize=2)
    ax.plot(x, vox_pts, "s-", label="Voxelized points", color="coral", markersize=2)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Number of Gaussians")
    ax.set_title(f"Point counts per frame [{config_name}, {args.dataset_name}, {args.sequence_name}]")
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x)[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    avg_orig = sum(orig_pts) / n
    avg_vox = sum(vox_pts) / n
    vox_ratio = avg_vox / avg_orig * 100 if avg_orig > 0 else 0
    ax.annotate(
        f"avg original={avg_orig:.0f}, voxelized={avg_vox:.0f} ({vox_ratio:.1f}%)",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()
    pts_path = os.path.join(out_dir, "point_counts.png")
    fig.savefig(pts_path, dpi=150)
    print(f"Saved: {pts_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
