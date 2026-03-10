#!/usr/bin/env python3
"""
Evaluate MesonGS decompression quality against a GT static Gaussian Splat model.

Loads the GT PLY and the decompressed PLY, renders test cameras using the
gaussian renderer, and computes PSNR / SSIM / LPIPS.

Output: summary JSON, CSV, and optionally saved rendered images.

Must be run from the MesonGS directory in the mesongs conda environment.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
import torchvision

# --- sys.path setup ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MESONGS_ROOT = os.path.dirname(_THIS_DIR)
if _MESONGS_ROOT not in sys.path:
    sys.path.insert(0, _MESONGS_ROOT)

from scene import GaussianModel, Scene
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def evaluate_quality(scene, gaussians, pipe, background, desc="Evaluating",
                     render_save_dir=None, gt_save_dir=None):
    """Evaluates quality of the current model state on test cameras.
    Optionally saves rendered images to render_save_dir and GT images to gt_save_dir."""
    cams = scene.getTestCameras()
    ssims = []
    lpipss = []
    psnrs = []

    if render_save_dir:
        os.makedirs(render_save_dir, exist_ok=True)
    if gt_save_dir:
        os.makedirs(gt_save_dir, exist_ok=True)

    for idx, viewpoint in enumerate(tqdm(cams, desc=desc)):
        res = render(viewpoint, gaussians, pipe, background, clamp_color=True)
        image = res["render"]
        gt_image = viewpoint.original_image[0:3, :, :].to("cuda")

        psnrs.append(psnr(image.unsqueeze(0), gt_image).unsqueeze(0))
        ssims.append(ssim(image, gt_image))
        lpipss.append(lpips(image, gt_image, net_type='vgg'))

        if render_save_dir:
            torchvision.utils.save_image(image, os.path.join(render_save_dir, f'{idx:05d}.png'))
        if gt_save_dir:
            torchvision.utils.save_image(gt_image, os.path.join(gt_save_dir, f'{idx:05d}.png'))

    psnr_val = torch.tensor(psnrs).mean().item()
    ssim_val = torch.tensor(ssims).mean().item()
    lpips_val = torch.tensor(lpipss).mean().item()

    return psnr_val, ssim_val, lpips_val


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MesonGS decompression quality against GT model"
    )

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--gt_ply_path", type=str, default=None,
                        help="Direct path to GT PLY file (overrides auto-discovery from --ply_path)")
    parser.add_argument("--ply_path", type=str, default=None,
                        help="Path to checkpoint dir containing point_cloud/ (auto-discovers max iteration)")
    parser.add_argument("--decompressed_ply_path", type=str, required=True,
                        help="Path to the decompressed PLY file")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Folder for evaluation results (JSON, CSV, optional renders)")
    parser.add_argument("--save_renders", action="store_true",
                        help="Save GT images, GT model renders, and decompressed model renders")

    args = parser.parse_args(sys.argv[1:])

    dataset_args = lp.extract(args)
    pipe_args = pp.extract(args)

    # --- Resolve GT PLY path ---
    if args.gt_ply_path:
        gt_ply_file = args.gt_ply_path
    elif args.ply_path:
        ckpt_path = os.path.join(args.ply_path, "point_cloud")
        max_iter = searchForMaxIteration(ckpt_path)
        gt_ply_file = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")
        print(f"Auto-discovered GT PLY: {gt_ply_file}")
    else:
        parser.error("Either --gt_ply_path or --ply_path must be provided")

    decomp_ply_file = args.decompressed_ply_path
    if not os.path.exists(decomp_ply_file):
        print(f"Error: Decompressed PLY not found: {decomp_ply_file}")
        sys.exit(1)

    os.makedirs(args.output_path, exist_ok=True)

    safe_state(args.quiet)

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_base = os.path.join(args.output_path, 'render') if args.save_renders else None

    # --- Print configuration ---
    print("=" * 70)
    print("MesonGS Decompression Quality Evaluation (Static)")
    print("=" * 70)
    print(f"  GT PLY:             {gt_ply_file}")
    print(f"  Decompressed PLY:   {decomp_ply_file}")
    print(f"  Dataset source:     {dataset_args.source_path}")
    print(f"  Output path:        {args.output_path}")
    print(f"  Save renders:       {args.save_renders}")
    print("=" * 70)

    # --- 1. Load GT model via Scene (provides test cameras) ---
    print("\nLoading GT model...")
    gt_gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gt_gaussians, given_ply_path=gt_ply_file)
    n_test_cams = len(scene.getTestCameras())
    print(f"Test cameras: {n_test_cams}")

    print("\nEvaluating GT Quality...")
    with torch.no_grad():
        gt_psnr, gt_ssim, gt_lpips = evaluate_quality(
            scene, gt_gaussians, pipe_args, background, desc="GT Eval",
            render_save_dir=os.path.join(render_base, 'gt_model') if render_base else None,
            gt_save_dir=os.path.join(render_base, 'gt') if render_base else None)
    print(f"GT Quality - PSNR: {gt_psnr:.4f}, SSIM: {gt_ssim:.4f}, LPIPS: {gt_lpips:.4f}")

    # --- 2. Load decompressed model ---
    print("\nLoading Decompressed model...")
    decomp_gaussians = GaussianModel(dataset_args.sh_degree)
    decomp_gaussians.load_ply(decomp_ply_file)
    scene.gaussians = decomp_gaussians

    print("Evaluating Decompressed Quality...")
    with torch.no_grad():
        decomp_psnr, decomp_ssim, decomp_lpips = evaluate_quality(
            scene, decomp_gaussians, pipe_args, background, desc="Decomp Eval",
            render_save_dir=os.path.join(render_base, 'decomp_model') if render_base else None)
    print(f"Decomp Quality - PSNR: {decomp_psnr:.4f}, SSIM: {decomp_ssim:.4f}, LPIPS: {decomp_lpips:.4f}")

    # --- 3. Summary ---
    psnr_drop = gt_psnr - decomp_psnr
    ssim_drop = gt_ssim - decomp_ssim
    lpips_drop = decomp_lpips - gt_lpips

    print("\n" + "=" * 70)
    print(f"Evaluation Summary ({n_test_cams} test cameras)")
    print("=" * 70)
    print(f"  GT Model       -> PSNR: {gt_psnr:.4f}, SSIM: {gt_ssim:.4f}, LPIPS: {gt_lpips:.4f}")
    print(f"  Decomp Model   -> PSNR: {decomp_psnr:.4f}, SSIM: {decomp_ssim:.4f}, LPIPS: {decomp_lpips:.4f}")
    print(f"  Quality Drop   -> PSNR: {psnr_drop:.4f}, SSIM: {ssim_drop:.6f}, LPIPS: {lpips_drop:.4f}")
    print("=" * 70)

    # --- 4. Save results ---
    results = {
        "config": {
            "gt_ply_path": gt_ply_file,
            "decompressed_ply_path": decomp_ply_file,
            "dataset_source": dataset_args.source_path,
            "num_test_cameras": n_test_cams,
        },
        "gt": {
            "psnr": gt_psnr,
            "ssim": gt_ssim,
            "lpips": gt_lpips,
        },
        "decompressed": {
            "psnr": decomp_psnr,
            "ssim": decomp_ssim,
            "lpips": decomp_lpips,
        },
        "drop": {
            "psnr": psnr_drop,
            "ssim": ssim_drop,
            "lpips": lpips_drop,
        },
    }

    json_path = os.path.join(args.output_path, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  JSON saved to: {json_path}")

    print("Done.")
