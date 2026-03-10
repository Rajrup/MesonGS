#!/usr/bin/env python3
"""
ImageGS (PNG) Compression + Decompression for a static Gaussian Splat model.

Uses gsplat's PngCompression which:
  - Sorts splats via PLAS (Parallel Linear Assignment Sorting)
  - Quantizes attributes to 8-bit (16-bit for means) and saves as lossless PNG
  - Compresses SH rest coefficients via K-means clustering (torchpq)

  1. Load PLY (not timed)
  2. Compress to PNG directory (timed)
  3. Decompress from PNG directory (timed)
  4. Save decoded result as PLY (not timed)

Reference: Compact 3D Scene Representation via Self-Organizing Gaussian Grids
           (https://arxiv.org/abs/2312.13299)
Core implementation from nerfstudio's gsplat: https://github.com/nerfstudio-project/gsplat
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from plyfile import PlyData

# --- sys.path setup ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MESONGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MESONGS_ROOT not in sys.path:
    sys.path.insert(0, _MESONGS_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from compress_decompress import PngCompression


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def load_ply_to_splats(ply_path: str, sh_degree: int = 3) -> tuple[dict, int]:
    """Load a 3DGS PLY file and return splats dict for PngCompression.

    The PLY stores pre-activation values (logit opacity, log scales).
    PngCompression expects pre-activation values, so we pass them through as-is.

    Returns:
        (splats, uncompressed_size_bytes) where splats has keys:
        "means", "quats", "scales", "opacities", "sh0", "shN"
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)

    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith('f_rest_')],
        key=lambda x: int(x.split('_')[-1])
    )
    if rest_names:
        sh_rest = np.stack([vertex[name] for name in rest_names], axis=1)
    else:
        sh_rest = np.zeros((len(vertex), 0), dtype=np.float32)

    opacities = np.asarray(vertex['opacity'])
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)

    splats = {
        'means': torch.from_numpy(means.copy()).float(),
        'quats': torch.from_numpy(quats.copy()).float(),
        'scales': torch.from_numpy(scales.copy()).float(),
        'opacities': torch.from_numpy(opacities.copy()).float(),
        'sh0': torch.from_numpy(sh_dc.copy()).float().unsqueeze(-2),
        'shN': torch.from_numpy(sh_rest.copy()).float().reshape(len(vertex), -1, 3),
    }

    uncompressed_size_bytes = sum(v.numel() * v.element_size() for v in splats.values())
    return splats, uncompressed_size_bytes


def save_splats_to_ply(splats: dict, output_path: str, sh_degree: int = 3):
    """Save decompressed splats dict back to a 3DGS-compatible PLY file.

    The splats contain pre-activation values, which is what GaussianModel.load_ply() expects.
    """
    def _to_np2d(t):
        """Convert a splat tensor to a 2D (N, C) numpy array."""
        a = t.detach().cpu().float().numpy()
        if a.ndim == 1:
            return a.reshape(-1, 1)
        return a.reshape(a.shape[0], -1)

    means = _to_np2d(splats['means'])          # (N, 3)
    quats = _to_np2d(splats['quats'])           # (N, 4)
    scales = _to_np2d(splats['scales'])         # (N, 3)
    opacities = _to_np2d(splats['opacities'])   # (N, 1)
    sh0 = _to_np2d(splats['sh0'])               # (N, 3)
    shN = _to_np2d(splats['shN'])               # (N, K*3)

    N = means.shape[0]
    n_rest = shN.shape[1]

    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(3):
        attr_names.append(f'f_dc_{i}')
    for i in range(n_rest):
        attr_names.append(f'f_rest_{i}')
    attr_names.append('opacity')
    for i in range(3):
        attr_names.append(f'scale_{i}')
    for i in range(4):
        attr_names.append(f'rot_{i}')

    normals = np.zeros((N, 3), dtype=np.float32)
    data = np.concatenate([
        means, normals, sh0, shN,
        opacities,
        scales, quats,
    ], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for name in attr_names:
            f.write(f"property float {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())


def get_dir_size_bytes(path: str) -> int:
    """Get total size of all files in a directory (non-recursive)."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImageGS (PNG) compress + decompress for a static Gaussian Splat model"
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
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output from PLAS and K-means")
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

    os.makedirs(args.output_path, exist_ok=True)
    compress_dir = os.path.join(args.output_path, "compressed")
    os.makedirs(compress_dir, exist_ok=True)

    verbose = not args.quiet

    # --- Print configuration ---
    print("=" * 70)
    print("ImageGS (PNG) Compress + Decompress Pipeline (Static)")
    print("=" * 70)
    print(f"  PLY file:           {ply_file_path}")
    print(f"  Output path:        {args.output_path}")
    print(f"  Compress dir:       {compress_dir}")
    print(f"  SH degree:          {args.sh_degree}")
    print("=" * 70)

    # --- Load PLY ---
    splats, uncompressed_size_bytes = load_ply_to_splats(ply_file_path, sh_degree=args.sh_degree)
    N_original = splats['means'].shape[0]
    print(f"\nOriginal points: {N_original}")
    print(f"Uncompressed size: {uncompressed_size_bytes / 1024 / 1024:.2f} MB")

    # Move splats to CUDA for PLAS sorting and K-means
    splats = {k: v.cuda() for k, v in splats.items()}

    # --- Compress (timed) ---
    compressor = PngCompression(use_sort=True, verbose=verbose)

    torch.cuda.synchronize()
    t_enc_start = time.perf_counter()

    compressor.compress(compress_dir, splats)

    torch.cuda.synchronize()
    t_enc_end = time.perf_counter()
    encode_time_ms = (t_enc_end - t_enc_start) * 1000

    compressed_size_bytes = get_dir_size_bytes(compress_dir)

    # --- Decompress (timed) ---
    t_dec_start = time.perf_counter()

    decoded_splats = compressor.decompress(compress_dir)

    t_dec_end = time.perf_counter()
    decode_time_ms = (t_dec_end - t_dec_start) * 1000

    N_decoded = decoded_splats['means'].shape[0]

    # --- Save decoded PLY ---
    if args.output_ply_path:
        ply_out_path = args.output_ply_path
    else:
        ply_out_path = os.path.join(args.output_path, "decompressed", "point_cloud.ply")

    save_splats_to_ply(decoded_splats, ply_out_path, args.sh_degree)
    print(f"\nSaved decompressed PLY to: {ply_out_path}")

    # --- Statistics ---
    compression_ratio = uncompressed_size_bytes / compressed_size_bytes

    stats = {
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
