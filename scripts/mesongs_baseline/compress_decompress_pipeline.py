#!/usr/bin/env python3
"""
MesonGS Compression + Decompression for a static Gaussian Splat model.

  1. Load PLY into MesonGS GaussianModel via Scene class
  2. Compute importance via cal_imp() (renders from train cameras)
  3. Optional: prune low-importance Gaussians
  4. encode_mesongs(): Octree -> VQ -> RAHT -> Block Quantize -> LZ77
  5. decode_mesongs(): LZ77 -> Dequant -> iRAHT -> VQ lookup -> Octree decode
  6. Convert Euler angles -> quaternions, save as PLY

Must be run from the MesonGS directory in the mesongs conda environment.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch import nn

# --- sys.path setup: MesonGS root must be on path ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MESONGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MESONGS_ROOT not in sys.path:
    sys.path.insert(0, _MESONGS_ROOT)

from scene import GaussianModel, Scene
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from mesongs import cal_imp, prune_mask, universal_config, config3, config2, nerf_syn_small_config
from compression.compress_decompress import encode_mesongs, decode_mesongs
from compression.utils import euler_to_quaternion

# ---------------------------------------------------------------------------
# PLY utilities
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def compute_uncompressed_size(gaussians, sh_degree=3):
    """Compute uncompressed size in bytes from GaussianModel attributes (float32)."""
    N = gaussians.get_xyz.shape[0]
    n_sh_rest = 3 * ((sh_degree + 1) ** 2 - 1)  # 45 for sh_degree=3
    n_floats_per_point = 3 + 3 + n_sh_rest + 1 + 3 + 4  # xyz + f_dc + f_rest + opacity + scale + rot
    return N * n_floats_per_point * 4  # float32 = 4 bytes


def save_decoded_ply(decoded_gaussians, output_path):
    """Convert decoded model (Euler angles) to quaternions and save as PLY."""
    with torch.no_grad():
        quats = euler_to_quaternion(decoded_gaussians._euler.detach())
        decoded_gaussians._rotation = nn.Parameter(quats, requires_grad=False)
    decoded_gaussians.save_ply(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MesonGS compress + decompress for a static Gaussian Splat model"
    )

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ply_path", type=str, default=None,
                        help="Path to checkpoint dir containing point_cloud/ (auto-discovers max iteration)")
    parser.add_argument("--given_ply_path", default='', type=str,
                        help="Direct path to a specific PLY file (overrides --ply_path)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", default="./output/compress_test", type=str,
                        help="Folder for stats JSON and decompressed PLY output")
    parser.add_argument("--output_ply_path", type=str, default=None,
                        help="Path for decompressed PLY file (default: <output_path>/decompressed/point_cloud.ply)")
    parser.add_argument("--prune", action="store_true", help="Enable pruning before compression")

    args = parser.parse_args(sys.argv[1:])

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

    args.given_ply_path = ply_file_path

    # --- Extract argument groups ---
    dataset_args = lp.extract(args)
    pipe_args = pp.extract(args)
    opt_args = op.extract(args)

    # --- Apply hyper_config override ---
    if pipe_args.hyper_config == 'universal':
        used_config = universal_config
    elif pipe_args.hyper_config == 'syn_small':
        used_config = nerf_syn_small_config
    elif pipe_args.hyper_config == 'config2':
        used_config = config2
    elif pipe_args.hyper_config == 'config3':
        used_config = config3
    else:
        used_config = None

    if used_config is not None:
        print(f"Applying config: {pipe_args.hyper_config} for scene {pipe_args.scene_imp}")
        dataset_args.percent = used_config['prune'][pipe_args.scene_imp]
        dataset_args.codebook_size = used_config['cb'][pipe_args.scene_imp]
        dataset_args.depth = used_config['depth'][pipe_args.scene_imp]
        dataset_args.n_block = used_config['n_block'][pipe_args.scene_imp]

    os.makedirs(args.output_path, exist_ok=True)

    # --- Print configuration ---
    print("=" * 70)
    print("MesonGS Compress + Decompress Pipeline (Static)")
    print("=" * 70)
    print(f"  PLY file:           {ply_file_path}")
    print(f"  Dataset source:     {dataset_args.source_path}")
    print(f"  Output path:        {args.output_path}")
    print(f"  Scene:              {pipe_args.scene_imp}")
    print(f"  SH degree:          {dataset_args.sh_degree}")
    print(f"  Octree depth:       {dataset_args.depth}")
    print(f"  N block:            {dataset_args.n_block}")
    print(f"  Codebook size:      {dataset_args.codebook_size}")
    print(f"  Pruning:            {args.prune} (percent={dataset_args.percent})")
    print(f"  Num bits:           {dataset_args.num_bits}")
    print("=" * 70)

    # --- Initialize model via Scene ---
    safe_state(args.quiet)

    gaussians = GaussianModel(dataset_args.sh_degree, depth=dataset_args.depth, num_bits=dataset_args.num_bits)
    scene = Scene(dataset_args, gaussians, given_ply_path=ply_file_path)
    gaussians.training_setup(opt_args)

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    N_original = gaussians.get_xyz.shape[0]
    uncompressed_size_bytes = compute_uncompressed_size(gaussians, dataset_args.sh_degree)
    print(f"\nOriginal points: {N_original}")
    print(f"Uncompressed size: {uncompressed_size_bytes / 1024 / 1024:.2f} MB")

    # --- Importance calculation ---
    print("\nCalculating Importance...")
    torch.cuda.synchronize()
    t_enc_start = time.perf_counter()
    with torch.no_grad():
        imp = cal_imp(gaussians, scene.getTrainCameras(), pipe_args, background)

    # --- Optional pruning ---
    N_after_prune = N_original
    if args.prune:
        print(f"Pruning (percent={dataset_args.percent})...")
        pmask = prune_mask(dataset_args.percent, imp)
        imp = imp[torch.logical_not(pmask)]
        gaussians.prune_points(pmask)
        N_after_prune = gaussians.get_xyz.shape[0]
        print(f"Pruned: {N_original} -> {N_after_prune} points")
    else:
        print("Skipping pruning.")

    # --- Encode ---
    bitstreams = encode_mesongs(gaussians, dataset_args, imp)

    torch.cuda.synchronize()
    t_enc_end = time.perf_counter()
    encode_time_ms = (t_enc_end - t_enc_start) * 1000

    compressed_size_bytes = sum(len(v) for v in bitstreams.values())
    N_after_octree = gaussians.get_xyz.shape[0]

    # --- Decode ---
    torch.cuda.synchronize()
    t_dec_start = time.perf_counter()

    decoded_gaussians = decode_mesongs(bitstreams, dataset_args)

    torch.cuda.synchronize()
    t_dec_end = time.perf_counter()
    decode_time_ms = (t_dec_end - t_dec_start) * 1000

    N_decoded = decoded_gaussians.get_xyz.shape[0]

    # --- Save decoded PLY ---
    if args.output_ply_path:
        ply_out_path = args.output_ply_path
    else:
        ply_out_path = os.path.join(args.output_path, "decompressed", "point_cloud.ply")

    os.makedirs(os.path.dirname(ply_out_path), exist_ok=True)
    save_decoded_ply(decoded_gaussians, ply_out_path)
    print(f"\nSaved decompressed PLY to: {ply_out_path}")

    # --- Statistics ---
    compression_ratio = uncompressed_size_bytes / compressed_size_bytes

    stats = {
        "scene": pipe_args.scene_imp,
        "config": pipe_args.hyper_config,
        "depth": dataset_args.depth,
        "n_block": dataset_args.n_block,
        "codebook_size": dataset_args.codebook_size,
        "prune": args.prune,
        "prune_percent": dataset_args.percent,
        "num_bits": dataset_args.num_bits,
        "original_points": N_original,
        "after_prune_points": N_after_prune,
        "after_octree_points": N_after_octree,
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
    print(f"  Points: {N_original}"
          f"{'  ->  ' + str(N_after_prune) + ' pruned' if args.prune else ''}"
          f"  ->  {N_after_octree} octree  ->  {N_decoded} decoded")
    print(f"  Encode time:         {encode_time_ms:.2f} ms")
    print(f"  Decode time:         {decode_time_ms:.2f} ms")
    print(f"  Uncompressed size:   {uncompressed_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Compressed size:     {compressed_size_bytes / 1024 / 1024:.4f} MB")
    print(f"  Compression ratio:   {compression_ratio:.2f}x")
    print(f"  Stats JSON:          {stats_path}")
    print("=" * 70)
    print("Done.")
