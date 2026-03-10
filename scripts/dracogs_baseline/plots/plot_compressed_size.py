"""
Plot DracoGS compressed size and point counts per frame.
Requires: benchmark_dracogs.csv under input_folder/<config_name>/

Generates two plots:
  1. Size: uncompressed vs total compressed per frame
  2. Point counts: original vs decoded
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt


def load_benchmark_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "frame": int(row["frame_id"]),
                "uncompressed_size_bytes": int(row["uncompressed_size_bytes"]),
                "compressed_size_bytes": int(row["compressed_size_bytes"]),
                "original_points": int(row["original_points"]),
                "decoded_points": int(row["decoded_points"]),
            })
    return sorted(rows, key=lambda x: x["frame"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Base folder for dracogs compression output")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--sequence_name", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True,
                        help="Config subfolder name (e.g. qp_16_qfd_16_..._cl_7)")
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    config_dir = os.path.join(args.input_folder, args.config_name)
    csv_path = os.path.join(config_dir, "benchmark_dracogs.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, args.config_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise SystemExit(f"Required file not found: {csv_path}")

    rows = load_benchmark_csv(csv_path)
    n = len(rows)
    frame_ids = [r["frame"] for r in rows]
    x = range(n)
    tick_every = max(1, n // 20)

    # ---- Plot 1: Size ----
    uncompressed_mb = [r["uncompressed_size_bytes"] / 1024 / 1024 for r in rows]
    total_mb = [r["compressed_size_bytes"] / 1024 / 1024 for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, uncompressed_mb, "o-", label="Uncompressed", color="gray", markersize=3, alpha=0.7)
    ax.plot(x, total_mb, "s-", label="DracoGS compressed", color="coral", markersize=3)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Size (MB)")
    ax.set_title(f"DracoGS size per frame [{args.config_name}, {args.dataset_name}, {args.sequence_name}]")
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x)[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    avg_unc = sum(uncompressed_mb) / n
    avg_tot = sum(total_mb) / n
    ratio = avg_tot / avg_unc * 100 if avg_unc > 0 else 0
    ax.annotate(
        f"avg uncompressed={avg_unc:.2f} MB, compressed={avg_tot:.2f} MB ({ratio:.1f}%)",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()
    size_path = os.path.join(out_dir, "compressed_size.png")
    fig.savefig(size_path, dpi=150)
    print(f"Saved: {size_path}")
    plt.close(fig)

    # ---- Plot 2: Point counts ----
    orig_pts = [r["original_points"] for r in rows]
    decoded_pts = [r["decoded_points"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, orig_pts, "o-", label="Original", color="steelblue", markersize=3)
    ax.plot(x, decoded_pts, "s-", label="Decoded", color="coral", markersize=3)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Number of Gaussians")
    ax.set_title(f"Point counts per frame [{args.config_name}, {args.dataset_name}, {args.sequence_name}]")
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x)[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    avg_orig = sum(orig_pts) / n
    avg_dec = sum(decoded_pts) / n
    dec_ratio = avg_dec / avg_orig * 100 if avg_orig > 0 else 0
    ax.annotate(
        f"avg original={avg_orig:.0f}, decoded={avg_dec:.0f} ({dec_ratio:.1f}%)",
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
