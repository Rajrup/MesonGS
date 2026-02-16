import io
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import csv
import uuid
from argparse import Namespace

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

from scene import GaussianModel, Scene
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, ft_render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import torchvision
from mesongs import cal_imp, prune_mask, universal_config, config3, config2, nerf_syn_small_config
from torch import nn
from scene.gaussian_model import ToEulerAngles_FT, decode_oct
from raht_torch import copyAsort, transform_batched_torch, itransform_batched_torch, inv_haar3D_param
from utils.quant_utils import split_length
from scene.gaussian_model import quantize_tensor, dequantize_tensor, torch_vanilla_quant_ave, torch_vanilla_dequant_ave

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

    # 3. Pruning
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

    # 5. Optionally save pruned PLY
    if args.save_pruned:
        pruned_ply_path = os.path.join(args.output_path, 'pruned.ply')
        gaussians.save_ply(pruned_ply_path)
        print(f"Saved pruned PLY to {pruned_ply_path}")

    return scene, gaussians, dataset_args, kept_imp

def _raht_forward(gaussians):
    """Quaternion-to-Euler conversion + RAHT forward transform.
    Returns (dc_np, ac_tensor): DC coefficient as numpy, AC coefficients as CUDA tensor."""
    r = gaussians.get_ori_rotation
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    eulers = ToEulerAngles_FT(q)
    rf = torch.concat([gaussians.get_origin_opacity.detach(), eulers.detach(),
                       gaussians.get_features_dc.detach().contiguous().squeeze()], axis=-1)
    C = rf[gaussians.reorder]
    iW1 = gaussians.res['iW1']
    iW2 = gaussians.res['iW2']
    iLeft_idx = gaussians.res['iLeft_idx']
    iRight_idx = gaussians.res['iRight_idx']

    for d in range(gaussians.depth * 3):
        C[iLeft_idx[d]], C[iRight_idx[d]] = transform_batched_torch(
            iW1[d], iW2[d], C[iLeft_idx[d]], C[iRight_idx[d]])

    return C[0].cpu().numpy(), C[1:]

def _calibrate_and_quantize(data, qas, qa_cnt, n_block):
    """Calibrate block quantizers then quantize a multi-channel tensor.
    Returns (quantized_np, trans_list, updated_qa_cnt)."""
    split = split_length(data.shape[0], n_block)
    # Calibrate: VanillaQuan.forward() sets scale/zero_point per block
    for ch in range(data.shape[-1]):
        start = 0
        for j, length in enumerate(split):
            qas[qa_cnt + ch * n_block + j](data[start : start + length, ch])
            start += length
    # Quantize using calibrated stats
    quantized = []
    trans = []
    for i in range(data.shape[-1]):
        t1, trans1 = torch_vanilla_quant_ave(data[:, i], split,
                                              qas[qa_cnt : qa_cnt + n_block])
        quantized.append(t1)
        trans.extend(trans1)
        qa_cnt += n_block
    return np.concatenate(quantized, axis=-1), trans, qa_cnt

def encode_mesongs(gaussians: GaussianModel, dataset_args: ModelParams, imp: torch.Tensor, output_dir: str):
    """Encodes the pruned gaussians into bitstreams."""
    print("\n=== Encoding MesonGS ===")
    os.makedirs(output_dir, exist_ok=True)
    bin_dir = os.path.join(output_dir, 'bins')
    os.makedirs(bin_dir, exist_ok=True)

    # 1. Octree Coding
    print("Octree Coding...")
    gaussians.octree_coding(imp, dataset_args.oct_merge, raht=dataset_args.raht)
    print(f"Points after Octree: {gaussians.get_xyz.shape[0]}")

    # 2. Init Block Quantizers
    print("Initializing Block Quantizers...")
    if dataset_args.per_block_quant:
        gaussians.init_qas(dataset_args.n_block)

    # 3. Vector Quantization
    print("Vector Quantizing Features...")
    gaussians.vq_fe(imp, dataset_args.codebook_size, dataset_args.batch_size, dataset_args.steps)

    # 4. Compression
    print("Compressing...")
    trans_array = [gaussians.depth, gaussians.n_block]
    n_block = gaussians.n_block

    with torch.no_grad():
        # VQ data
        ntk = gaussians._feature_indices.detach().contiguous().cpu().int().numpy()
        cb = gaussians._features_rest.detach().contiguous().cpu().numpy()

        # RAHT forward (single pass)
        cf, ac_data = _raht_forward(gaussians)

        # Calibrate + Quantize AC coefficients (7 channels)
        qa_cnt = 0
        qci, trans_ac, qa_cnt = _calibrate_and_quantize(ac_data, gaussians.qas, qa_cnt, n_block)
        trans_array.extend(trans_ac)

        # Calibrate + Quantize Scales (3 channels)
        scaling = gaussians.get_ori_scaling.detach()
        scaling_q, trans_sc, qa_cnt = _calibrate_and_quantize(scaling, gaussians.qas, qa_cnt, n_block)
        trans_array.extend(trans_sc)

        trans_array = np.array(trans_array)

    # 5. Entropy coding (compress to in-memory bitstreams)
    def _compress_npz(**arrays):
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    bitstreams = {
        'oct':  _compress_npz(points=gaussians.oct, params=gaussians.oct_param),
        'ntk':  _compress_npz(ntk=ntk),
        'um':   _compress_npz(umap=cb),
        'orgb': _compress_npz(f=cf, i=qci.astype(np.uint8)),
        'ct':   _compress_npz(i=scaling_q.astype(np.uint8)),
        't':    _compress_npz(t=trans_array),
    }

    # 7. Write bitstreams to disk
    for name, data in bitstreams.items():
        with open(os.path.join(bin_dir, f'{name}.npz'), 'wb') as f:
            f.write(data)

    # 9. Pack and report zip size
    bin_zip_path = os.path.join(output_dir, 'bins.zip')
    os.system(f'zip -j {bin_zip_path} {bin_dir}/* > /dev/null')

    # Report in-memory buffer sizes
    total_buf_size = 0
    for name, data in bitstreams.items():
        size = len(data)
        total_buf_size += size
        print(f"  {name}.npz size: {size / 1024:.2f} KB")
    print(f"Total .npz size: {total_buf_size / 1024 / 1024:.4f} MB")

    total_size = os.path.getsize(bin_zip_path)
    print(f"Total zip size:  {total_size / 1024 / 1024:.4f} MB")

    return bin_zip_path

def decode_mesongs(bitstream_path, dataset_args):
    """Decodes the bitstream and reconstructs the Gaussian Model."""
    print("\n=== Decoding MesonGS ===")

    gaussians = GaussianModel(dataset_args.sh_degree, depth=dataset_args.depth, num_bits=dataset_args.num_bits)
    bin_dir = os.path.join(os.path.dirname(bitstream_path), 'bins')

    # 1. Read raw bytes from disk
    raw = {}
    for name in ['t', 'oct', 'ntk', 'um', 'orgb', 'ct']:
        with open(os.path.join(bin_dir, f'{name}.npz'), 'rb') as f:
            raw[name] = f.read()

    # 2. Decompress (npz inflate from memory)
    trans_array = np.load(io.BytesIO(raw['t']))["t"]
    oct_vals    = np.load(io.BytesIO(raw['oct']))
    ntk         = np.load(io.BytesIO(raw['ntk']))["ntk"]
    cb          = np.load(io.BytesIO(raw['um']))["umap"]
    oef_vals    = np.load(io.BytesIO(raw['orgb']))
    ct_vals     = np.load(io.BytesIO(raw['ct']))

    # 3. Decode
    with torch.no_grad():
        # --- Metadata ---
        depth = int(trans_array[0])
        n_block = int(trans_array[1])
        gaussians.depth = depth
        gaussians.n_block = n_block

        # --- Octree decode -> xyz ---
        octree = oct_vals["points"]
        oct_param = oct_vals["params"]
        gaussians.og_number_points = octree.shape[0]
        dxyz, V = decode_oct(oct_param, octree, depth)
        gaussians._xyz = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
        n_points = dxyz.shape[0]

        # --- VQ lookup -> features_rest ---
        cb_tensor = torch.tensor(cb)
        features_rest = torch.zeros([ntk.shape[0], cb_tensor.shape[1]])
        for i in range(ntk.shape[0]):
            features_rest[i] = cb_tensor[int(ntk[i])]
        gaussians.n_sh = (gaussians.max_sh_degree + 1) ** 2
        features_rest = features_rest.to("cuda").contiguous().reshape(-1, gaussians.n_sh - 1, 3)
        gaussians._features_rest = nn.Parameter(features_rest, requires_grad=False)

        # --- Parse quantized RAHT coefficients and scales ---
        orgb_f    = torch.tensor(oef_vals["f"], dtype=torch.float, device="cuda")
        q_orgb_i  = torch.tensor(oef_vals["i"].astype(np.float32), dtype=torch.float, device="cuda").reshape(7, -1).contiguous().transpose(0, 1)
        q_scale_i = torch.tensor(ct_vals["i"], dtype=torch.float, device="cuda").reshape(3, -1).contiguous().transpose(0, 1)

        # --- Dequantize AC coefficients (7 channels) ---
        qa_cnt = 2
        rf_len = q_orgb_i.shape[0]
        assert rf_len + 1 == n_points
        split = split_length(rf_len, n_block)
        rf_orgb = []
        for i in range(7):
            rf_i = torch_vanilla_dequant_ave(q_orgb_i[:, i], split, trans_array[qa_cnt:qa_cnt + 2 * n_block])
            rf_orgb.append(rf_i.reshape(-1, 1))
            qa_cnt += 2 * n_block
        rf_orgb = torch.concat(rf_orgb, dim=-1)

        # --- Dequantize Scales (3 channels) ---
        scale_len = q_scale_i.shape[0]
        assert scale_len == n_points
        scale_split = split_length(scale_len, n_block)
        de_scale = []
        for i in range(3):
            scale_i = torch_vanilla_dequant_ave(q_scale_i[:, i], scale_split, trans_array[qa_cnt:qa_cnt + 2 * n_block])
            de_scale.append(scale_i.reshape(-1, 1))
            qa_cnt += 2 * n_block
        de_scale = torch.concat(de_scale, axis=-1).to("cuda")
        gaussians._scaling = nn.Parameter(de_scale.requires_grad_(False))

        # --- Inverse RAHT ---
        C = torch.concat([orgb_f.reshape(1, -1), rf_orgb], 0)
        w, val, reorder = copyAsort(V)
        gaussians.reorder = reorder

        res_inv = inv_haar3D_param(V, depth)
        pos          = res_inv['pos']
        iW1          = res_inv['iW1']
        iW2          = res_inv['iW2']
        iS           = res_inv['iS']
        iLeft_idx    = res_inv['iLeft_idx']
        iRight_idx   = res_inv['iRight_idx']
        iLeft_idx_CT  = res_inv['iLeft_idx_CT']
        iRight_idx_CT = res_inv['iRight_idx_CT']
        iTrans_idx    = res_inv['iTrans_idx']
        iTrans_idx_CT = res_inv['iTrans_idx_CT']

        CT_yuv_q_temp = C[pos.astype(int)]
        raht_features = torch.zeros(C.shape).cuda()
        OC = torch.zeros(C.shape).cuda()

        for i in range(depth * 3):
            OC[iTrans_idx[i]] = CT_yuv_q_temp[iTrans_idx_CT[i]]
            OC[iLeft_idx[i]], OC[iRight_idx[i]] = itransform_batched_torch(
                iW1[i], iW2[i],
                CT_yuv_q_temp[iLeft_idx_CT[i]],
                CT_yuv_q_temp[iRight_idx_CT[i]])
            CT_yuv_q_temp[:iS[i]] = OC[:iS[i]]

        raht_features[reorder] = OC

        # --- Assign decoded attributes ---
        gaussians._opacity = nn.Parameter(raht_features[:, :1].detach(), requires_grad=False)
        gaussians._euler = nn.Parameter(raht_features[:, 1:4].nan_to_num_(0).detach(), requires_grad=False)
        gaussians._features_dc = nn.Parameter(raht_features[:, 4:].unsqueeze(1).detach(), requires_grad=False)
        gaussians.active_sh_degree = gaussians.max_sh_degree

    print(f"Reconstructed {gaussians.get_xyz.shape[0]} points")
    return gaussians

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
    bitstream_path = encode_mesongs(gaussians, dataset_args, imp, args.output_path)
    
    # 3. Decode
    decoded_gaussians = decode_mesongs(bitstream_path, dataset_args)
    
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
