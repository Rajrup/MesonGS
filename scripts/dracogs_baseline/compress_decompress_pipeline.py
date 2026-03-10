#!/usr/bin/env python3
"""
DracoGS Compression + Decompression for a static Gaussian Splat model.

  1. Read PLY (not timed)
  2. Encode via DracoGS (in-memory Draco bitstream, timed)
  3. Decode via DracoGS (in-memory, timed)
  4. Save decoded result as PLY (not timed)
"""

import os
import sys
import json
import time
import argparse

# --- sys.path setup: DracoGS build + compression dirs ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MESONGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DRACOGS_ROOT = os.path.join(_MESONGS_ROOT, "DracoGS")
_DRACOGS_BUILD = os.path.join(_DRACOGS_ROOT, "build", "compression")
_DRACOGS_COMP = os.path.join(_DRACOGS_ROOT, "compression")

for p in (_DRACOGS_BUILD, _DRACOGS_COMP):
    if p not in sys.path:
        sys.path.insert(0, p)

from compression_decompression import encode_dracogs, decode_dracogs

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from utils import read_gs_ply, save_gs_ply

DEFAULT_EG = 16
DEFAULT_EO = 16
DEFAULT_ET = 16
DEFAULT_ES = 16
DEFAULT_CL = 10

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DracoGS compress + decompress for a static Gaussian Splat model"
    )
    parser.add_argument("--ply_path", type=str, default=None,
                        help="Path to checkpoint dir containing point_cloud/ (auto-discovers max iteration)")
    parser.add_argument("--given_ply_path", type=str, default=None,
                        help="Direct path to a specific PLY file (overrides --ply_path)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Folder for stats JSON and decompressed PLY output")
    parser.add_argument("--output_ply_path", type=str, default=None,
                        help="Path for decompressed PLY file (default: <output_path>/decompressed/point_cloud.ply)")
    parser.add_argument("--sh_degree", type=int, default=3)

    # Draco quantization parameters (0=lossless, higher=more bits=better quality)
    parser.add_argument("--eg", type=int, default=DEFAULT_EG, help="Quantization bits for position (0-30)")
    parser.add_argument("--eo", type=int, default=DEFAULT_EO, help="Quantization bits for opacity (0-30)")
    parser.add_argument("--et", type=int, default=DEFAULT_ET, help="Quantization bits for rotation/scales (0-30)")
    parser.add_argument("--es", type=int, default=DEFAULT_ES, help="Quantization bits for SH (0-30)")
    parser.add_argument("--cl", type=int, default=DEFAULT_CL, help="Compression level (0-10)")

    args = parser.parse_args()

    # --- Resolve PLY path ---
    if args.given_ply_path:
        ply_file_path = args.given_ply_path
    elif args.ply_path:
        ckpt_path = os.path.join(args.ply_path, "point_cloud")
        max_iter = searchForMaxIteration(ckpt_path)
        ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")
        print(f"Auto-discovered PLY: {ply_file_path}")
    else:
        parser.error("Either --given_ply_path or --ply_path must be provided")

    # Draco parameters
    qp = args.eg
    qfd = args.es
    qfr1 = args.es
    qfr2 = args.es
    qfr3 = args.es
    qo = args.eo
    qs = args.et
    qr = args.et
    cl = args.cl

    os.makedirs(args.output_path, exist_ok=True)

    # --- Print configuration ---
    print("=" * 70)
    print("DracoGS Compress + Decompress Pipeline (Static)")
    print("=" * 70)
    print(f"  PLY file:           {ply_file_path}")
    print(f"  Output path:        {args.output_path}")
    print(f"  SH degree:          {args.sh_degree}")
    print(f"  Quantization:       qp={qp} qfd={qfd} qfr1={qfr1} qfr2={qfr2} qfr3={qfr3} qo={qo} qs={qs} qr={qr}")
    print(f"  Compression level:  {cl}")
    print("=" * 70)

    # --- Load PLY ---
    gs_data, uncompressed_size_bytes = read_gs_ply(ply_file_path, sh_degree=args.sh_degree)
    N_original = gs_data["positions"].shape[0]
    print(f"\nOriginal points: {N_original}")
    print(f"Uncompressed size: {uncompressed_size_bytes / 1024 / 1024:.2f} MB")

    # --- Encode (timed) ---
    t_enc_start = time.perf_counter()
    bitstream = encode_dracogs(
        gs_data,
        qp=qp, qfd=qfd,
        qfr1=qfr1, qfr2=qfr2, qfr3=qfr3,
        qo=qo, qs=qs, qr=qr,
        cl=cl,
    )
    t_enc_end = time.perf_counter()
    encode_time_ms = (t_enc_end - t_enc_start) * 1000
    compressed_size_bytes = len(bitstream)

    # --- Decode (timed) ---
    t_dec_start = time.perf_counter()
    gs_decoded = decode_dracogs(bitstream)
    t_dec_end = time.perf_counter()
    decode_time_ms = (t_dec_end - t_dec_start) * 1000
    N_decoded = gs_decoded["positions"].shape[0]

    # --- Save decoded PLY ---
    if args.output_ply_path:
        ply_out_path = args.output_ply_path
    else:
        ply_out_path = os.path.join(args.output_path, "decompressed", "point_cloud.ply")

    save_gs_ply(gs_decoded, ply_out_path)
    print(f"\nSaved decompressed PLY to: {ply_out_path}")

    # --- Statistics ---
    compression_ratio = uncompressed_size_bytes / compressed_size_bytes

    stats = {
        "eg": args.eg,
        "eo": args.eo,
        "et": args.et,
        "es": args.es,
        "cl": cl,
        "draco_params": {
            "qp": qp, "qfd": qfd,
            "qfr1": qfr1, "qfr2": qfr2, "qfr3": qfr3,
            "qo": qo, "qs": qs, "qr": qr,
        },
        "sh_degree": args.sh_degree,
        "original_points": N_original,
        "decoded_points": N_decoded,
        "uncompressed_size_bytes": uncompressed_size_bytes,
        "compressed_size_bytes": compressed_size_bytes,
        "compression_ratio": compression_ratio,
        "encode_time_ms": encode_time_ms,
        "decode_time_ms": decode_time_ms,
        "ply_input": ply_file_path,
        "ply_output": ply_out_path,
    }

    stats_path = os.path.join(args.output_path, "compression_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n" + "=" * 70)
    print("Compression Summary")
    print("=" * 70)
    print(f"  Points: {N_original}  ->  {N_decoded}")
    print(f"  Encode time:         {encode_time_ms:.2f} ms")
    print(f"  Decode time:         {decode_time_ms:.2f} ms")
    print(f"  Uncompressed size:   {uncompressed_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Compressed size:     {compressed_size_bytes / 1024 / 1024:.4f} MB")
    print(f"  Compression ratio:   {compression_ratio:.2f}x")
    print(f"  Stats JSON:          {stats_path}")
    print("=" * 70)
    print("Done.")
