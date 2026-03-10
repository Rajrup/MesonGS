"""
Plot GT vs MesonGS-decompressed model quality per frame (PSNR and SSIM).
Requires: evaluation_results.csv under input_folder/<config_name>/evaluation/
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt


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
    evaluation_csv = os.path.join(config_dir, "evaluation", "evaluation_results.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, args.config_name)
    out_path = os.path.join(out_dir, "quality.png")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(evaluation_csv):
        raise SystemExit(f"Required file not found: {evaluation_csv}")

    frames, gt_psnr, decomp_psnr, gt_ssim, decomp_ssim = [], [], [], [], []
    with open(evaluation_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["frame_id"] == "avg":
                continue
            frames.append(int(row["frame_id"]))
            gt_psnr.append(float(row["gt_psnr"]))
            decomp_psnr.append(float(row["decomp_psnr"]))
            gt_ssim.append(float(row["gt_ssim"]))
            decomp_ssim.append(float(row["decomp_ssim"]))

    n = len(frames)
    x = range(n)
    tick_every = max(1, n // 20)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- PSNR ---
    ax1.plot(x, gt_psnr, "o-", label="GT model", color="green", markersize=3)
    ax1.plot(x, decomp_psnr, "s-", label="MesonGS decompressed", color="coral", markersize=3)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"Quality per frame: GT vs MesonGS [{args.config_name}, {args.dataset_name}, {args.sequence_name}]")
    avg_gt = sum(gt_psnr) / n
    avg_dec = sum(decomp_psnr) / n
    ax1.axhline(y=avg_gt, color="green", linestyle="--", alpha=0.4,
                label=f"GT avg = {avg_gt:.2f}")
    ax1.axhline(y=avg_dec, color="coral", linestyle="--", alpha=0.4,
                label=f"Decomp avg = {avg_dec:.2f}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- SSIM ---
    ax2.plot(x, gt_ssim, "o-", label="GT model", color="green", markersize=3)
    ax2.plot(x, decomp_ssim, "s-", label="MesonGS decompressed", color="coral", markersize=3)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("SSIM")
    avg_gt_ssim = sum(gt_ssim) / n
    avg_dec_ssim = sum(decomp_ssim) / n
    ax2.axhline(y=avg_gt_ssim, color="green", linestyle="--", alpha=0.4,
                label=f"GT avg = {avg_gt_ssim:.4f}")
    ax2.axhline(y=avg_dec_ssim, color="coral", linestyle="--", alpha=0.4,
                label=f"Decomp avg = {avg_dec_ssim:.4f}")
    ax2.set_xticks(list(x)[::tick_every])
    ax2.set_xticklabels(frames[::tick_every], rotation=90)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
