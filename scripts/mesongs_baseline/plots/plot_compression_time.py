"""
Stacked area plot of MesonGS encode and decode time per frame.
Requires: benchmark_mesongs.csv under input_folder/<config_name>/
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
                        help="Base folder for mesongs compression output")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--sequence_name", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True,
                        help="Config subfolder name (e.g. depth_14_nblock_66_cb_2048)")
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    config_dir = os.path.join(args.input_folder, args.config_name)
    csv_path = os.path.join(config_dir, "benchmark_mesongs.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, args.config_name)
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
    tick_every = max(1, n // 20)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        x,
        encode_ms,
        decode_ms,
        labels=["Encode (MesonGS)", "Decode (MesonGS)"],
        colors=["steelblue", "coral"],
        alpha=0.8,
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"MesonGS encode + decode time per frame [{args.config_name}, {args.dataset_name}, {args.sequence_name}]")
    ax.set_xticks(x[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

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
