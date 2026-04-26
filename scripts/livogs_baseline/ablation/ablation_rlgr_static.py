#!/usr/bin/env python3
"""
RLGR Ablation (static model): CPU vs GPU (multiple block sizes).

For a single static Gaussian Splat model:
  1. Load the PLY once.
  2. Run encode_livogs() to get the compressed state.
  3. Decode RLGR to recover the quantized coefficient tensor (int32, GPU).
  4. Benchmark each RLGR variant N repetitions on the SAME tensor.

Variants:
  cpu       – rlgr.Encoder per-channel loop (includes GPU→CPU transfer)
  gpu_full  – rlgr_gpu.EncoderGPU(block_size=-1)  (full channel, matches CPU bytes)
  gpu_8192  – rlgr_gpu.EncoderGPU(block_size=8192)
  gpu_4096  – rlgr_gpu.EncoderGPU(block_size=4096) (current default)
  gpu_2048  – rlgr_gpu.EncoderGPU(block_size=2048)
  gpu_1024  – rlgr_gpu.EncoderGPU(block_size=1024)
  gpu_512   – rlgr_gpu.EncoderGPU(block_size=512)
  gpu_256   – rlgr_gpu.EncoderGPU(block_size=256)
  gpu_128   – rlgr_gpu.EncoderGPU(block_size=128)
  gpu_64    – rlgr_gpu.EncoderGPU(block_size=64)
  gpu_32    – rlgr_gpu.EncoderGPU(block_size=32)
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import json
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIVOGS_BASELINE_DIR = os.path.dirname(_THIS_DIR)
_SCRIPTS_DIR = os.path.dirname(_LIVOGS_BASELINE_DIR)
_MESONGS_ROOT = os.path.dirname(_SCRIPTS_DIR)
_LIVOGS_COMPRESSION = os.path.join(_MESONGS_ROOT, "LiVoGS", "compression")
for p in (_MESONGS_ROOT, _LIVOGS_COMPRESSION):
    if p not in sys.path:
        sys.path.insert(0, p)

from compress_decompress import encode_livogs
import rlgr_gpu
import rlgr

sys.path.insert(0, _LIVOGS_BASELINE_DIR)
from compress_decompress_pipeline import load_ply, searchForMaxIteration

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS = [
    ("cpu",      None),
    ("gpu_full", -1),
    ("gpu_8192", 8192),
    ("gpu_4096", 4096),
    ("gpu_2048", 2048),
    ("gpu_1024", 1024),
    ("gpu_512",  512),
    ("gpu_256",  256),
    ("gpu_128",  128),
    ("gpu_64",   64),
    ("gpu_32",   32),
]


def benchmark_gpu_variant(coeff_int32, block_size, device_id):
    """Encode + decode with GPU RLGR; return timing and size."""
    encoder = rlgr_gpu.EncoderGPU(block_size=block_size, flagSigned=1)
    decoder = rlgr_gpu.DecoderGPU()

    torch.cuda.synchronize(device_id)
    t0 = time.perf_counter()
    compressed = encoder.rlgrEncode(coeff_int32)
    torch.cuda.synchronize(device_id)
    encode_ms = (time.perf_counter() - t0) * 1000

    compressed_bytes = int(compressed['compressed_data'].shape[0])

    torch.cuda.synchronize(device_id)
    t0 = time.perf_counter()
    decoded, _ = decoder.rlgrDecode(compressed)
    torch.cuda.synchronize(device_id)
    decode_ms = (time.perf_counter() - t0) * 1000

    return {
        "rlgr_encode_ms": encode_ms,
        "rlgr_decode_ms": decode_ms,
        "transfer_to_cpu_ms": 0.0,
        "transfer_to_gpu_ms": 0.0,
        "pure_rlgr_encode_ms": encode_ms,
        "pure_rlgr_decode_ms": decode_ms,
        "compressed_size_bytes": compressed_bytes,
    }


def benchmark_cpu_variant(coeff_int32, device_id):
    """Encode + decode with CPU RLGR; return timing and size."""
    n_symbols, n_channels = coeff_int32.shape

    encoder_cpu = rlgr.Encoder()
    decoder_cpu = rlgr.Decoder()

    torch.cuda.synchronize(device_id)
    t_xfer_start = time.perf_counter()
    np_coeff = coeff_int32.cpu().numpy()
    t_xfer_end = time.perf_counter()
    transfer_to_cpu_ms = (t_xfer_end - t_xfer_start) * 1000

    compressed_bufs = []
    t_enc_start = time.perf_counter()
    for ch in range(n_channels):
        _, compressed_data = encoder_cpu.rlgrEncode(np_coeff[:, ch], 1)
        compressed_bufs.append(compressed_data)
    t_enc_end = time.perf_counter()
    pure_encode_ms = (t_enc_end - t_enc_start) * 1000

    total_compressed = sum(len(b) for b in compressed_bufs)

    t_dec_start = time.perf_counter()
    decoded_channels = []
    for ch in range(n_channels):
        _, decoded_data = decoder_cpu.rlgrDecode(compressed_bufs[ch], n_symbols, 1)
        decoded_channels.append(decoded_data)
    t_dec_end = time.perf_counter()
    pure_decode_ms = (t_dec_end - t_dec_start) * 1000

    t_xfer_start = time.perf_counter()
    np_decoded = np.stack(decoded_channels, axis=1).astype(np.int32)
    torch.from_numpy(np_decoded).to(coeff_int32.device)
    torch.cuda.synchronize(device_id)
    t_xfer_end = time.perf_counter()
    transfer_to_gpu_ms = (t_xfer_end - t_xfer_start) * 1000

    return {
        "rlgr_encode_ms": transfer_to_cpu_ms + pure_encode_ms,
        "rlgr_decode_ms": pure_decode_ms + transfer_to_gpu_ms,
        "transfer_to_cpu_ms": transfer_to_cpu_ms,
        "transfer_to_gpu_ms": transfer_to_gpu_ms,
        "pure_rlgr_encode_ms": pure_encode_ms,
        "pure_rlgr_decode_ms": pure_decode_ms,
        "compressed_size_bytes": total_compressed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="RLGR ablation (static): CPU vs GPU block sizes")
    p.add_argument("--ply_path", type=str, default=None,
                   help="Checkpoint dir containing point_cloud/ (auto-discovers max iteration)")
    p.add_argument("--given_ply_path", type=str, default=None,
                   help="Direct path to a specific PLY file (overrides --ply_path)")
    p.add_argument("--output_folder", type=str, required=True)
    p.add_argument("--repetitions", type=int, default=50,
                   help="Number of timed repetitions per variant (default: 50)")
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--J", type=int, default=14)
    p.add_argument("--qp", type=float, default=0.0001,
                   help="Uniform quantization step (default: 0.0001)")
    p.add_argument("--qpq", type=float, default=None)
    p.add_argument("--qps", type=float, default=None)
    p.add_argument("--qpo", type=float, default=None)
    p.add_argument("--qpdc", type=float, default=None)
    p.add_argument("--qpac", type=float, default=None)
    p.add_argument("--quantize_config_json", type=str, default=None)
    p.add_argument("--sh_color_space", type=str, default="klt",
                   choices=["rgb", "yuv", "klt"])
    p.add_argument("--nvcomp_algorithm", type=str, default="ANS",
                   choices=["None", "LZ4", "Snappy", "GDeflate", "Deflate",
                            "zStandard", "Cascaded", "Bitcomp", "ANS"])
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    # --- Resolve PLY path ---
    if args.given_ply_path:
        ply_file_path = args.given_ply_path
    elif args.ply_path:
        ckpt_path = os.path.join(args.ply_path, "point_cloud")
        max_iter = searchForMaxIteration(ckpt_path)
        ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")
        print(f"Auto-discovered PLY: {ply_file_path}")
    else:
        p.error("Either --given_ply_path or --ply_path must be provided")

    nvcomp_algorithm = None if args.nvcomp_algorithm == "None" else args.nvcomp_algorithm

    if args.quantize_config_json is not None:
        with open(args.quantize_config_json) as _f:
            _qp_data = json.load(_f)
        quantize_step = _qp_data["quantize_config"]
    else:
        qs = args.qp
        quantize_step = {
            'quats':   args.qpq if args.qpq is not None else qs,
            'scales':  args.qps if args.qps is not None else qs,
            'opacity': args.qpo if args.qpo is not None else qs,
            'sh_dc':   args.qpdc if args.qpdc is not None else qs,
            'sh_rest': [args.qpac if args.qpac is not None else qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
        }

    device = args.device
    device_id = int(device.split(':')[1]) if device.startswith('cuda:') else 0

    os.makedirs(args.output_folder, exist_ok=True)

    print("=" * 70)
    print("RLGR Ablation Study (Static)")
    print("=" * 70)
    print(f"  PLY file:       {ply_file_path}")
    print(f"  Output folder:  {args.output_folder}")
    print(f"  Repetitions:    {args.repetitions}")
    print(f"  J={args.J}, sh_color_space={args.sh_color_space}")
    print(f"  nvcomp:         {nvcomp_algorithm or 'None'}")
    print(f"  Variants:       {[v[0] for v in VARIANTS]}")
    print("=" * 70)

    # --- Load PLY ---
    params, _ = load_ply(ply_file_path, device=device)
    N = params['means'].shape[0]
    print(f"\nLoaded {N} Gaussians")

    # --- Encode once to get compressed state, then recover coeff tensor ---
    print("Encoding model with LiVoGS...")
    torch.cuda.synchronize(device_id)
    cs = encode_livogs(
        params, J=args.J, device=device, device_id=device_id,
        sh_color_space=args.sh_color_space, quantize_step=quantize_step,
        rlgr_block_size=4096, nvcomp_algorithm=nvcomp_algorithm,
    )
    torch.cuda.synchronize(device_id)

    decoder_ref = rlgr_gpu.DecoderGPU()
    coeff_int32, _ = decoder_ref.rlgrDecode(cs['compressed_attributes'])
    torch.cuda.synchronize(device_id)
    n_symbols, n_channels = coeff_int32.shape
    print(f"Coefficient tensor: {n_symbols} symbols x {n_channels} channels")
    del cs, decoder_ref

    # --- Warmup ---
    print("\nWarming up all variants...")
    for vname, blk in VARIANTS:
        try:
            if blk is None:
                benchmark_cpu_variant(coeff_int32, device_id)
            else:
                benchmark_gpu_variant(coeff_int32, blk, device_id)
            print(f"  {vname}: OK")
        except Exception as e:
            print(f"  {vname}: warmup failed ({e})")
    torch.cuda.empty_cache()
    print("Warmup done.\n")

    # --- Main benchmark loop ---
    csv_path = os.path.join(args.output_folder, "ablation_rlgr.csv")
    csv_columns = [
        "frame_id", "variant", "num_symbols", "num_channels",
        "rlgr_encode_ms", "rlgr_decode_ms",
        "transfer_to_cpu_ms", "transfer_to_gpu_ms",
        "pure_rlgr_encode_ms", "pure_rlgr_decode_ms",
        "compressed_size_bytes", "correct",
    ]

    all_rows = []

    for rep in range(args.repetitions):
        for vname, blk in VARIANTS:
            if blk is None:
                result = benchmark_cpu_variant(coeff_int32, device_id)
            else:
                result = benchmark_gpu_variant(coeff_int32, blk, device_id)

            row = {
                "frame_id": rep,
                "variant": vname,
                "num_symbols": n_symbols,
                "num_channels": n_channels,
                "correct": True,
                **{k: result[k] for k in csv_columns if k in result},
            }
            all_rows.append(row)

        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  Repetition {rep + 1}/{args.repetitions}")

    # --- Write CSV (same format as dynamic ablation for plotting compatibility) ---
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_columns)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\nCSV saved: {csv_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary (mean ± std across all repetitions)")
    print("=" * 70)
    for vname, _ in VARIANTS:
        rows = [r for r in all_rows if r["variant"] == vname]
        if not rows:
            continue
        enc = np.array([r["rlgr_encode_ms"] for r in rows])
        dec = np.array([r["rlgr_decode_ms"] for r in rows])
        pure_enc = np.array([r["pure_rlgr_encode_ms"] for r in rows])
        pure_dec = np.array([r["pure_rlgr_decode_ms"] for r in rows])
        size = np.array([r["compressed_size_bytes"] for r in rows])
        print(f"  {vname:>10s}  enc={enc.mean():8.2f}±{enc.std():6.2f}ms "
              f"(pure={pure_enc.mean():8.2f}±{pure_enc.std():6.2f}ms)  "
              f"dec={dec.mean():8.2f}±{dec.std():6.2f}ms "
              f"(pure={pure_dec.mean():8.2f}±{pure_dec.std():6.2f}ms)  "
              f"size={size.mean():>12,.0f}B")
    print("=" * 70)

    # --- Save config ---
    config = {
        "ply_file": ply_file_path,
        "num_gaussians": N,
        "num_symbols": n_symbols,
        "num_channels": n_channels,
        "repetitions": args.repetitions,
        "J": args.J,
        "quantize_step": quantize_step,
        "sh_color_space": args.sh_color_space,
        "nvcomp_algorithm": nvcomp_algorithm or "None",
        "variants": [v[0] for v in VARIANTS],
    }
    config_path = os.path.join(args.output_folder, "ablation_rlgr_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
