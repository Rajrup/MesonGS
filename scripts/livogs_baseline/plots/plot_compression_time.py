"""
Stacked area plot of LiVoGS encode and decode time per frame.
Requires: benchmark_livogs.csv under input_folder/<config_name>/
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
                "encode_time_ms": float(row["encode_time_ms"]),
                "decode_time_ms": float(row["decode_time_ms"]),
            })
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
    out_path = os.path.join(out_dir, "compression_time.png")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise SystemExit(f"Required file not found: {csv_path}")

    rows = load_benchmark_csv(csv_path)
    frame_ids = [r["frame"] for r in rows]
    n = len(frame_ids)
    encode_ms = [r["encode_time_ms"] for r in rows]
    decode_ms = [r["decode_time_ms"] for r in rows]

    x = list(range(n))
    tick_every = 10

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        x,
        encode_ms,
        decode_ms,
        labels=["Encode (LiVoGS)", "Decode (LiVoGS)"],
        colors=["steelblue", "coral"],
        alpha=0.8,
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"LiVoGS encode + decode time per frame [{config_name}, {args.dataset_name}, {args.sequence_name}]")
    ax.set_xticks(x[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add avg annotation
    avg_enc = sum(encode_ms) / n
    avg_dec = sum(decode_ms) / n
    ax.annotate(
        f"avg enc={avg_enc:.1f}ms, dec={avg_dec:.1f}ms, total={avg_enc + avg_dec:.1f}ms",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
