#!/usr/bin/env python3
"""
Benchmark nvCOMP algorithms for octree position compression in LiVoGS.

Tests all nvCOMP algorithms (+ None baseline) on a few frames, verifying
bitwise correctness of decoded positions and measuring compression ratio
and encode/decode time overhead.
"""

import os
import sys
import csv
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_LIVOGS_COMPRESSION = os.path.join(_VIDEOGS_ROOT, "LiVoGS", "compression")
if _VIDEOGS_ROOT not in sys.path:
    sys.path.insert(0, _VIDEOGS_ROOT)
if _LIVOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _LIVOGS_COMPRESSION)

from compress_decompress import encode_livogs, decode_livogs

NVCOMP_ALGORITHMS = [
    None, "LZ4", "Snappy", "GDeflate", "Deflate",
    "zStandard", "Cascaded", "Bitcomp", "ANS",
]


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def load_videogs_ply(ply_path, device='cuda'):
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

    params['quats'] = F.normalize(params['quats'], p=2, dim=1)
    if params['opacities'].min() < 0 or params['opacities'].max() > 1:
        params['opacities'] = torch.sigmoid(params['opacities'])
    if params['scales'].min() < 0:
        params['scales'] = torch.exp(params['scales'])

    return params


def build_quantize_step(qstep, sh_degree=3):
    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    return {
        'quats': qstep, 'scales': qstep, 'opacity': qstep,
        'sh_dc': qstep, 'sh_rest': [qstep] * n_rest,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark nvCOMP algorithms for octree positions")
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to checkpoint dir (contains frame folders 0, 1, ...)")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=5)
    parser.add_argument("--J", type=int, default=15, help="Octree depth")
    parser.add_argument("--qstep", type=float, default=0.0001, help="Quantization step")
    parser.add_argument("--sh_color_space", type=str, default="klt",
                        choices=["rgb", "yuv", "klt"])
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: nvcomp_benchmark.csv in CWD)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    device_id = int(device.split(':')[1]) if device.startswith('cuda:') else 0
    quantize_step = build_quantize_step(args.qstep)

    frames = list(range(args.frame_start, args.frame_end))
    ply_paths = []
    for frame in frames:
        ckpt_path = os.path.join(args.ply_path, str(frame), "point_cloud")
        if not os.path.exists(ckpt_path):
            print(f"Warning: frame {frame} not found at {ckpt_path}, skipping")
            continue
        max_iter = searchForMaxIteration(ckpt_path)
        ply_paths.append((frame, os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")))

    if not ply_paths:
        print("No frames found!")
        return

    print(f"Testing {len(ply_paths)} frames with {len(NVCOMP_ALGORITHMS)} algorithm configs")
    print(f"  J={args.J}, qstep={args.qstep}, sh_color_space={args.sh_color_space}")
    print()

    # Warmup: run encode+decode for every algorithm on the first frame
    print("Warming up GPU for all algorithms...")
    warm_params = load_videogs_ply(ply_paths[0][1], device=device)
    for warm_algo in NVCOMP_ALGORITHMS:
        warm_name = warm_algo if warm_algo is not None else "None"
        try:
            torch.cuda.synchronize(device_id)
            cs = encode_livogs(warm_params, J=args.J, device=device, device_id=device_id,
                               sh_color_space=args.sh_color_space, quantize_step=quantize_step,
                               nvcomp_algorithm=warm_algo)
            _ = decode_livogs(cs, device=device, device_id=device_id)
            torch.cuda.synchronize(device_id)
            del cs
            print(f"  {warm_name}: OK")
        except Exception as e:
            print(f"  {warm_name}: warmup failed ({e})")
        torch.cuda.empty_cache()
    del warm_params
    torch.cuda.empty_cache()
    print("Warmup done.\n")

    # Pre-compute baseline (None) results for correctness comparison
    print("Pre-computing baseline (None) decoded positions...")
    baseline_means = {}
    for frame, ply_file in ply_paths:
        params = load_videogs_ply(ply_file, device=device)
        torch.cuda.synchronize(device_id)
        cs = encode_livogs(params, J=args.J, device=device, device_id=device_id,
                           sh_color_space=args.sh_color_space, quantize_step=quantize_step,
                           nvcomp_algorithm=None)
        torch.cuda.synchronize(device_id)
        dp = decode_livogs(cs, device=device, device_id=device_id)
        torch.cuda.synchronize(device_id)
        baseline_means[frame] = dp['means'].clone()
        del params, cs, dp
        torch.cuda.empty_cache()
    print(f"Baseline done for {len(baseline_means)} frames.\n")

    results = []

    for algo in NVCOMP_ALGORITHMS:
        algo_name = algo if algo is not None else "None"
        print(f"{'='*60}")
        print(f"Algorithm: {algo_name}")
        print(f"{'='*60}")

        algo_ok = True
        for frame, ply_file in ply_paths:
            params = load_videogs_ply(ply_file, device=device)
            torch.cuda.synchronize(device_id)

            t0 = time.perf_counter()
            try:
                compressed_state = encode_livogs(
                    params, J=args.J, device=device, device_id=device_id,
                    sh_color_space=args.sh_color_space, quantize_step=quantize_step,
                    nvcomp_algorithm=algo,
                )
                torch.cuda.synchronize(device_id)
            except Exception as e:
                print(f"  Frame {frame}: ENCODE FAILED: {e}")
                algo_ok = False
                del params
                torch.cuda.empty_cache()
                break
            encode_time_ms = (time.perf_counter() - t0) * 1000

            pos_raw_bytes = compressed_state['position_serialized_bytes']
            pos_compressed_bytes = compressed_state['position_compressed_bytes']

            t0 = time.perf_counter()
            try:
                decoded_params = decode_livogs(compressed_state, device=device, device_id=device_id)
                torch.cuda.synchronize(device_id)
            except Exception as e:
                print(f"  Frame {frame}: DECODE FAILED: {e}")
                algo_ok = False
                del params, compressed_state
                torch.cuda.empty_cache()
                break
            decode_time_ms = (time.perf_counter() - t0) * 1000

            positions_match = torch.equal(decoded_params['means'], baseline_means[frame])

            row = {
                "algorithm": algo_name,
                "frame": frame,
                "position_raw_bytes": pos_raw_bytes,
                "position_compressed_bytes": pos_compressed_bytes,
                "compression_ratio": pos_raw_bytes / pos_compressed_bytes if pos_compressed_bytes > 0 else 0,
                "encode_time_ms": encode_time_ms,
                "decode_time_ms": decode_time_ms,
                "positions_correct": positions_match,
            }
            results.append(row)

            status = "OK" if positions_match else "MISMATCH"
            print(f"  Frame {frame}: raw={pos_raw_bytes:,}B  comp={pos_compressed_bytes:,}B  "
                  f"ratio={row['compression_ratio']:.3f}x  "
                  f"enc={encode_time_ms:.2f}ms  dec={decode_time_ms:.2f}ms  [{status}]")

            del params, compressed_state, decoded_params
            torch.cuda.empty_cache()

            if not positions_match:
                print(f"  >>> Algorithm {algo_name} produced incorrect results, skipping remaining frames")
                algo_ok = False
                break

        if not algo_ok and not results or results[-1].get("positions_correct", True) is False:
            pass
        print()

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Algorithm':<12} {'Avg Ratio':>10} {'Avg Enc ms':>12} {'Avg Dec ms':>12} {'Correct':>8}")
    print("-" * 60)

    from collections import defaultdict
    by_algo = defaultdict(list)
    for r in results:
        by_algo[r["algorithm"]].append(r)

    for algo_name in [a if a is not None else "None" for a in NVCOMP_ALGORITHMS]:
        rows = by_algo.get(algo_name, [])
        if not rows:
            print(f"{algo_name:<12} {'FAILED':>10}")
            continue
        avg_ratio = np.mean([r["compression_ratio"] for r in rows])
        avg_enc = np.mean([r["encode_time_ms"] for r in rows])
        avg_dec = np.mean([r["decode_time_ms"] for r in rows])
        all_correct = all(r["positions_correct"] for r in rows)
        print(f"{algo_name:<12} {avg_ratio:>10.3f}x {avg_enc:>11.2f} {avg_dec:>11.2f} {'YES' if all_correct else 'NO':>8}")

    # --- Save CSV ---
    csv_path = args.output_csv or "nvcomp_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algorithm", "frame", "position_raw_bytes", "position_compressed_bytes",
            "compression_ratio", "encode_time_ms", "decode_time_ms", "positions_correct",
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()

'''
python scripts/livogs_baseline/benchmark_nvcomp_octree.py \
  --ply_path /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/checkpoint \
  --frame_start 0 --frame_end 5 \
  --J 15 --qstep 0.0001 --sh_color_space klt \
  --output_csv nvcomp_benchmark.csv \
  --device cuda:0
'''