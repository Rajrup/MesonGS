#!/usr/bin/env python3
"""
LiVoGS Compression + Decompression for a static Gaussian Splat model.

  1. Load PLY (not timed)
  2. GPU warmup: encode + decode once
  3. encode_livogs(): Morton -> Voxelize -> Merge -> Position encode -> RAHT -> Quantize -> RLGR (timed)
  4. decode_livogs(): RLGR -> Dequant -> Position decode -> RAHT prelude -> iRAHT (timed)
  5. save_to_ply(): Save reconstructed model to disk (not timed)

The compressed bytestream stays on GPU (no GPU->CPU transfer).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

# --- Setup sys.path for LiVoGS imports ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MESONGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_LIVOGS_COMPRESSION = os.path.join(_MESONGS_ROOT, "LiVoGS", "compression")
if _MESONGS_ROOT not in sys.path:
    sys.path.insert(0, _MESONGS_ROOT)
if _LIVOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _LIVOGS_COMPRESSION)

from compress_decompress import encode_livogs, decode_livogs

# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def load_ply(ply_path, device='cuda'):
    """Load a Gaussian Splatting PLY and return LiVoGS-compatible param dict on GPU.

    PLY attribute order:
        x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..44, opacity,
        scale_0..2, rot_0..3

    Opacities are converted from logit to [0,1], scales from log to positive.
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
    colors = np.concatenate([sh_dc, sh_rest], axis=1)

    opacities = np.asarray(vertex['opacity'])
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)

    params = {
        'means': torch.from_numpy(means.copy()).float().to(device),
        'quats': torch.from_numpy(quats.copy()).float().to(device),
        'scales': torch.from_numpy(scales.copy()).float().to(device),
        'opacities': torch.from_numpy(opacities.copy()).float().to(device),
        'colors': torch.from_numpy(colors.copy()).float().to(device),
    }

    uncompressed_size_bytes = sum(
        v.numel() * v.element_size() for v in params.values()
    )

    params['quats'] = F.normalize(params['quats'], p=2, dim=1)
    if params['opacities'].min() < 0 or params['opacities'].max() > 1:
        params['opacities'] = torch.sigmoid(params['opacities'])
    if params['scales'].min() < 0:
        params['scales'] = torch.exp(params['scales'])

    return params, uncompressed_size_bytes


def save_ply(params, output_path, sh_degree=3, eps=1e-6):
    """Save reconstructed params back to Gaussian Splatting PLY format.

    Converts opacities back to logit space and scales back to log space so that
    GaussianModel.load_ply() can consume them directly.
    """
    means = params['means'].detach().cpu().float().numpy()
    quats = params['quats'].detach().cpu().float().numpy()
    scales = params['scales'].detach().cpu().float().numpy()
    opacities = params['opacities'].detach().cpu().float().numpy()
    colors = params['colors'].detach().cpu().float().numpy()

    N = means.shape[0]

    opacities_c = np.clip(opacities, eps, 1.0 - eps)
    opacities_logit = np.log(opacities_c / (1.0 - opacities_c))
    scales_log = np.log(np.clip(scales, eps, None))

    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(3):
        attr_names.append(f'f_dc_{i}')
    n_rest = colors.shape[1] - 3
    for i in range(n_rest):
        attr_names.append(f'f_rest_{i}')
    attr_names.append('opacity')
    for i in range(3):
        attr_names.append(f'scale_{i}')
    for i in range(4):
        attr_names.append(f'rot_{i}')

    normals = np.zeros((N, 3), dtype=np.float32)
    data = np.concatenate([
        means, normals, colors,
        opacities_logit.reshape(-1, 1),
        scales_log, quats,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LiVoGS compress + decompress for a static Gaussian Splat model"
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

    # LiVoGS-specific parameters
    parser.add_argument("--J", type=int, default=15,
                        help="Octree depth for voxelization (default: 15)")
    parser.add_argument("--qp", type=float, default=0.0001,
                        help="Uniform quantization step for all attributes (default: 0.0001)")
    parser.add_argument("--qpq", type=float, default=None,
                        help="Override quantization step for quaternions")
    parser.add_argument("--qps", type=float, default=None,
                        help="Override quantization step for scales")
    parser.add_argument("--qpo", type=float, default=None,
                        help="Override quantization step for opacity")
    parser.add_argument("--qpdc", type=float, default=None,
                        help="Override quantization step for SH DC")
    parser.add_argument("--qpac", type=float, default=None,
                        help="Override quantization step for SH rest")
    parser.add_argument("--sh_color_space", type=str, default="rgb",
                        choices=["rgb", "yuv", "klt"],
                        help="Color space for SH coefficients (default: rgb)")
    parser.add_argument("--rlgr_block_size", type=int, default=4096,
                        help="RLGR parallel block size (default: 4096)")
    parser.add_argument("--quantize_config_json", type=str, default=None,
                        help="Path to JSON file with full quantize_config dict (overrides all --qp* args)")
    parser.add_argument("--nvcomp_algorithm", type=str, default="ANS",
                        choices=["None", "LZ4", "Snappy", "GDeflate", "Deflate",
                                 "zStandard", "Cascaded", "Bitcomp", "ANS"],
                        help="nvCOMP algorithm for octree position compression (default: ANS, 'None' to disable)")
    parser.add_argument("--device", type=str, default="cuda:0")
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

    nvcomp_algorithm = None if args.nvcomp_algorithm == "None" else args.nvcomp_algorithm

    # Build quantize_step dict
    if args.quantize_config_json is not None:
        with open(args.quantize_config_json) as _f:
            _qp_data = json.load(_f)
        quantize_step = _qp_data["quantize_config"]
        print(f"  Loaded quantize config: {args.quantize_config_json} (label: {_qp_data.get('label', '?')})")
    else:
        qs = args.qp
        quantize_step = {
            'quats': args.qpq if args.qpq is not None else qs,
            'scales': args.qps if args.qps is not None else qs,
            'opacity': args.qpo if args.qpo is not None else qs,
            'sh_dc': args.qpdc if args.qpdc is not None else qs,
            'sh_rest': [args.qpac if args.qpac is not None else qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
        }

    device = args.device
    if device.startswith('cuda:'):
        device_id = int(device.split(':')[1])
    else:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

    os.makedirs(args.output_path, exist_ok=True)

    # --- Print configuration ---
    print("=" * 70)
    print("LiVoGS Compress + Decompress Pipeline (Static)")
    print("=" * 70)
    print(f"  PLY file:           {ply_file_path}")
    print(f"  Output path:        {args.output_path}")
    print(f"  SH degree:          {args.sh_degree}")
    print(f"  Device:             {device}")
    print(f"  J (octree depth):   {args.J}")
    print(f"  Quantize steps:     quats={quantize_step['quats']}, scales={quantize_step['scales']}, "
          f"opacity={quantize_step['opacity']}, sh_dc={quantize_step['sh_dc']}, sh_rest={quantize_step['sh_rest']}")
    print(f"  SH color space:     {args.sh_color_space}")
    print(f"  RLGR block size:    {args.rlgr_block_size}")
    print(f"  nvCOMP algorithm:   {nvcomp_algorithm if nvcomp_algorithm else 'none'}")
    print("=" * 70)

    # --- Load PLY ---
    params, uncompressed_size_bytes = load_ply(ply_file_path, device=device)
    N_original = params['means'].shape[0]
    print(f"\nOriginal points: {N_original}")
    print(f"Uncompressed size: {uncompressed_size_bytes / 1024 / 1024:.2f} MB")

    # --- GPU Warmup ---
    print("\nGPU Warmup...")
    torch.cuda.synchronize(device_id)
    _warmup_cs = encode_livogs(
        params, J=args.J, device=device, device_id=device_id,
        sh_color_space=args.sh_color_space,
        quantize_step=quantize_step,
        rlgr_block_size=args.rlgr_block_size,
        nvcomp_algorithm=nvcomp_algorithm,
    )
    torch.cuda.synchronize(device_id)
    _warmup_dp = decode_livogs(_warmup_cs, device=device, device_id=device_id)
    torch.cuda.synchronize(device_id)
    del _warmup_cs, _warmup_dp
    print("GPU Warmup done.")

    # --- Encode (timed) ---
    torch.cuda.synchronize(device_id)
    t_enc_start = time.perf_counter()

    compressed_state = encode_livogs(
        params, J=args.J, device=device, device_id=device_id,
        sh_color_space=args.sh_color_space,
        quantize_step=quantize_step,
        rlgr_block_size=args.rlgr_block_size,
        nvcomp_algorithm=nvcomp_algorithm,
    )

    torch.cuda.synchronize(device_id)
    t_enc_end = time.perf_counter()
    encode_time_ms = (t_enc_end - t_enc_start) * 1000

    Nvox = compressed_state['Nvox']
    compressed_size_bytes = compressed_state['total_compressed_bytes']
    position_compressed_bytes = compressed_state['position_compressed_bytes']
    attribute_compressed_bytes = compressed_state['attribute_compressed_bytes']
    per_channel_compressed_bytes = compressed_state['per_channel_compressed_bytes']

    # --- Decode (timed) ---
    t_dec_start = time.perf_counter()

    decoded_params = decode_livogs(compressed_state, device=device, device_id=device_id)

    torch.cuda.synchronize(device_id)
    t_dec_end = time.perf_counter()
    decode_time_ms = (t_dec_end - t_dec_start) * 1000

    # --- Save decoded PLY ---
    if args.output_ply_path:
        ply_out_path = args.output_ply_path
    else:
        ply_out_path = os.path.join(args.output_path, "decompressed", "point_cloud.ply")

    save_ply(decoded_params, ply_out_path, args.sh_degree)
    print(f"\nSaved decompressed PLY to: {ply_out_path}")

    # --- Statistics ---
    compression_ratio = uncompressed_size_bytes / compressed_size_bytes

    per_ch = per_channel_compressed_bytes
    total_quats = sum(per_ch[0:4])
    total_scales = sum(per_ch[4:7])
    total_opacity = per_ch[7]
    total_sh_dc = sum(per_ch[8:11])
    total_sh_rest = sum(per_ch[11:])

    stats = {
        "J": args.J,
        "quantize_step": quantize_step,
        "sh_color_space": args.sh_color_space,
        "rlgr_block_size": args.rlgr_block_size,
        "sh_degree": args.sh_degree,
        "nvcomp_algorithm": nvcomp_algorithm,
        "original_points": N_original,
        "voxelized_points": Nvox,
        "uncompressed_size_bytes": uncompressed_size_bytes,
        "compressed_size_bytes": compressed_size_bytes,
        "position_compressed_bytes": position_compressed_bytes,
        "attribute_compressed_bytes": attribute_compressed_bytes,
        "compression_ratio": compression_ratio,
        "encode_time_ms": encode_time_ms,
        "decode_time_ms": decode_time_ms,
        "ply_input": ply_file_path,
        "ply_output": ply_out_path,
        "per_channel_compressed_bytes": {
            "quats": total_quats,
            "scales": total_scales,
            "opacity": total_opacity,
            "sh_dc": total_sh_dc,
            "sh_rest": total_sh_rest,
        },
    }

    stats_path = os.path.join(args.output_path, "compression_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n" + "=" * 70)
    print("Compression Summary")
    print("=" * 70)
    print(f"  Points: {N_original}  ->  {Nvox} voxels")
    print(f"  Encode time:         {encode_time_ms:.2f} ms")
    print(f"  Decode time:         {decode_time_ms:.2f} ms")
    print(f"  Uncompressed size:   {uncompressed_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Compressed size:     {compressed_size_bytes / 1024 / 1024:.4f} MB")
    print(f"    - position:  {position_compressed_bytes / 1024 / 1024:.4f} MB")
    print(f"    - attribute: {attribute_compressed_bytes / 1024 / 1024:.4f} MB")
    print(f"      quats:   {total_quats / 1024 / 1024:.4f} MB")
    print(f"      scales:  {total_scales / 1024 / 1024:.4f} MB")
    print(f"      opacity: {total_opacity / 1024 / 1024:.4f} MB")
    print(f"      sh_dc:   {total_sh_dc / 1024 / 1024:.4f} MB")
    print(f"      sh_rest: {total_sh_rest / 1024 / 1024:.4f} MB")
    print(f"  Compression ratio:   {compression_ratio:.2f}x")
    print(f"  Stats JSON:          {stats_path}")
    print("=" * 70)
    print("Done.")
