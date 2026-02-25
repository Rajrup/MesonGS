import torch
import io
import os
import numpy as np

from scene import GaussianModel
from arguments import ModelParams
from torch import nn
from scene.gaussian_model import ToEulerAngles_FT, decode_oct
from raht_torch import copyAsort, transform_batched_torch, itransform_batched_torch, inv_haar3D_param
from utils.quant_utils import split_length
from scene.gaussian_model import torch_vanilla_quant_ave, torch_vanilla_dequant_ave

def _raht_forward(gaussians: GaussianModel):
    """Quaternion-to-Euler conversion + RAHT forward transform.
    Returns (dc_np, ac_tensor): DC coefficient as numpy, AC coefficients as CUDA tensor."""
    r = gaussians.get_ori_rotation
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    eulers = ToEulerAngles_FT(q)
    rf = torch.concat([gaussians.get_origin_opacity.detach(), eulers.detach(), gaussians.get_features_dc.detach().contiguous().squeeze()], axis=-1)
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

def encode_mesongs(gaussians: GaussianModel, dataset_args: ModelParams, imp: torch.Tensor,
                    output_dir: str = "", save_to_disk: bool = False):
    """Encodes the pruned gaussians into in-memory bitstreams.
    Returns a dict of {name: bytes} for each compressed npz component.
    If save_to_disk is True, also writes .npz files and a .zip to output_dir."""
    print("\n=== Encoding MesonGS ===")

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

    # Report sizes
    total_buf_size = 0
    for name, data in bitstreams.items():
        size = len(data)
        total_buf_size += size
        print(f"  {name}.npz size: {size / 1024:.2f} KB")
    print(f"Total bitstream size: {total_buf_size / 1024 / 1024:.4f} MB")

    # 6. Optionally write to disk
    if save_to_disk and output_dir != "":
        bin_dir = os.path.join(output_dir, 'bins')
        os.makedirs(bin_dir, exist_ok=True)
        for name, data in bitstreams.items():
            with open(os.path.join(bin_dir, f'{name}.npz'), 'wb') as f:
                f.write(data)

        bin_zip_path = os.path.join(output_dir, 'bins.zip')
        os.system(f'zip -j {bin_zip_path} {bin_dir}/* > /dev/null')
        print(f"Total zip size:  {os.path.getsize(bin_zip_path) / 1024 / 1024:.4f} MB")

    return bitstreams

def decode_mesongs(bitstreams, dataset_args):
    """Decodes in-memory bitstreams and reconstructs the Gaussian Model.
    bitstreams: dict of {name: bytes} as returned by encode_mesongs."""
    print("\n=== Decoding MesonGS ===")

    gaussians = GaussianModel(dataset_args.sh_degree, depth=dataset_args.depth, num_bits=dataset_args.num_bits)

    # 1. Entropy decoding (npz inflate from memory)
    trans_array = np.load(io.BytesIO(bitstreams['t']))["t"]
    oct_vals    = np.load(io.BytesIO(bitstreams['oct']))
    ntk         = np.load(io.BytesIO(bitstreams['ntk']))["ntk"]
    cb          = np.load(io.BytesIO(bitstreams['um']))["umap"]
    oef_vals    = np.load(io.BytesIO(bitstreams['orgb']))
    ct_vals     = np.load(io.BytesIO(bitstreams['ct']))

    # 2. Decode
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
