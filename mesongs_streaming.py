import os
import sys
import argparse
import torch
from tqdm import tqdm

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

from scene import GaussianModel, Scene
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import torchvision
from mesongs import cal_imp, prune_mask, universal_config, config3, config2, nerf_syn_small_config

# --- Setup sys.path for MesonGS imports ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MESONGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _MESONGS_ROOT not in sys.path:
    sys.path.append(_MESONGS_ROOT)

from compression.compress_decompress import encode_mesongs, decode_mesongs

def get_combined_args(parser, target_cfg):
    cmdl_args = parser.parse_args()
    if target_cfg:
        # Override defaults with config file
        with open(target_cfg, 'r') as f:
            # This part assumes a specific config format, but for now we'll just use command line args
            # matching the shell script
            pass
    return cmdl_args

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

def initialize_model(args, dataset_args, pipe_args, opt_args):
    """Reads original model, evaluates GT, prunes, evaluates pruned."""
    print("\n=== Initializing Model ===")

    safe_state(args.quiet)

    gaussians = GaussianModel(dataset_args.sh_degree, depth=dataset_args.depth, num_bits=dataset_args.num_bits)
    scene = Scene(dataset_args, gaussians, given_ply_path=args.given_ply_path)
    gaussians.training_setup(opt_args)

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_base = os.path.join(args.output_path, 'render') if args.save_renders else None

    # 1. Evaluate GT Quality
    print("Evaluating GT Quality...")
    with torch.no_grad():
        gt_psnr, gt_ssim, gt_lpips = evaluate_quality(
            scene, gaussians, pipe_args, background, desc="GT Eval",
            render_save_dir=os.path.join(render_base, 'gt_model') if render_base else None,
            gt_save_dir=os.path.join(render_base, 'gt') if render_base else None)
    print(f"GT Quality - PSNR: {gt_psnr:.4f}, SSIM: {gt_ssim:.4f}, LPIPS: {gt_lpips:.4f}")

    # 2. Importance Calculation
    print("Calculating Importance...")
    with torch.no_grad():
        imp = cal_imp(gaussians, scene.getTrainCameras(), pipe_args, background)

    # 3. Pruning (optional)
    if args.prune:
        print(f"Pruning (Percent: {dataset_args.percent})...")
        pmask = prune_mask(dataset_args.percent, imp)
        kept_imp = imp[torch.logical_not(pmask)]

        n_before = gaussians.get_xyz.shape[0]
        gaussians.prune_points(pmask)
        n_after = gaussians.get_xyz.shape[0]
        print(f"Pruned: {n_before} -> {n_after} points")

        # 4. Evaluate Pruned Quality
        print("Evaluating Pruned Quality...")
        with torch.no_grad():
            pruned_psnr, pruned_ssim, pruned_lpips = evaluate_quality(
                scene, gaussians, pipe_args, background, desc="Pruned Eval",
                render_save_dir=os.path.join(render_base, 'pruned_model') if render_base else None)
        print(f"Pruned Quality - PSNR: {pruned_psnr:.4f}, SSIM: {pruned_ssim:.4f}, LPIPS: {pruned_lpips:.4f}")
        print(f"Drop from GT - PSNR: {gt_psnr - pruned_psnr:.4f}")

        if args.save_pruned:
            pruned_ply_path = os.path.join(args.output_path, 'pruned.ply')
            gaussians.save_ply(pruned_ply_path)
            print(f"Saved pruned PLY to {pruned_ply_path}")
    else:
        print("Skipping pruning.")
        kept_imp = imp

    return scene, gaussians, dataset_args, kept_imp

def main():
    parser = argparse.ArgumentParser(description="MesonGS Streaming Simulation")
    
    # Standard MesonGS arguments
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--given_ply_path", default='', type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", default="./output/streaming_test", type=str)
    parser.add_argument("--save_renders", action="store_true", help="Save rendered images for GT, pruned, and decompressed models")
    parser.add_argument("--save_pruned", action="store_true", help="Save pruned model as PLY")
    parser.add_argument("--save_to_disk", action="store_true", help="Write .npz and .zip bitstreams to disk")
    parser.add_argument("--prune", action="store_true", help="Enable universal pruning before compression")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Setup configs based on arguments (logic from mesongs.py)
    dataset_args = lp.extract(args)
    pipe_args = pp.extract(args)
    opt_args = op.extract(args)
    
    # Apply Hyper Config
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

    # 1. Initialize
    scene, gaussians, dataset_args, imp = initialize_model(args, dataset_args, pipe_args, opt_args)
    
    # 2. Encode
    bitstreams = encode_mesongs(gaussians, dataset_args, imp,
                                output_dir=args.output_path, save_to_disk=args.save_to_disk)

    # 3. Decode
    decoded_gaussians = decode_mesongs(bitstreams, dataset_args)
    
    # 4. Evaluate Decompressed
    # We need to update the scene to use the decoded gaussians
    scene.gaussians = decoded_gaussians
    
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    render_base = os.path.join(args.output_path, 'render') if args.save_renders else None

    print("\nEvaluating Decompressed Quality...")
    with torch.no_grad():
        dec_psnr, dec_ssim, dec_lpips = evaluate_quality(
            scene, decoded_gaussians, pipe_args, background, desc="Decoded Eval",
            render_save_dir=os.path.join(render_base, 'decomp_model') if render_base else None)
    print(f"Decompressed Quality - PSNR: {dec_psnr:.4f}, SSIM: {dec_ssim:.4f}, LPIPS: {dec_lpips:.4f}")

if __name__ == "__main__":
    main()
