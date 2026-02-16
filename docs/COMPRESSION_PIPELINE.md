# MesonGS Compression Pipeline – Code-Level Guide

This document maps **every step of MesonGS compression** to the codebase. Use it to benchmark encoding/decoding latency and compression gains against your training-free scheme.

**Input assumption:** Gaussians are trained with the **official 3D Gaussian Splatting** implementation and saved as a **PLY** (e.g. `point_cloud/iteration_30000/point_cloud.ply`). MesonGS loads this PLY and runs a **post-training codec** (no training-from-images in the codec; optional finetuning after compression).

---

## 1. High-level encoding flow

Entry: `mesongs.py` → `training()` (called from `main` with `--given_ply_path <ply>`).

Order of operations:

| Step | What | Where |
|------|------|--------|
| 1 | Load PLY (official 3D-GS format) | `Scene` + `GaussianModel.load_ply()` |
| 2 | Compute importance scores (view-dep + view-indep) | `cal_imp()` |
| 3 | Prune low-importance Gaussians | `prune_mask()` + `gaussians.prune_points()` |
| 4 | Octree + voxelize + merge; optionally prepare RAHT | `gaussians.octree_coding()` |
| 5 | (If per_block_quant) Allocate block quantizers | `gaussians.init_qas()` |
| 6 | Vector-quantize 1+D SH coefficients | `gaussians.vq_fe()` |
| 7 | Optional: evaluate test set (also **calibrates** qas if per_block_quant) | `evaluate_test()` |
| 8 | **Encode to disk** (Euler, RAHT, block quant, pack) | `scene.save_ft()` → `save_npz()` or `save_full_npz()` |

**Decoding:** `GaussianModel.load_npz(exp_dir)` reads the same `bins/` layout and inverts RAHT, dequant, VQ lookup, and restores `_xyz`, `_opacity`, `_euler`, `_features_dc`, `_features_rest`, `_scaling`.

---

## 2. Step-by-step (encoding)

### 2.1 Load PLY

- **File:** `scene/gaussian_model.py`
- **Method:** `load_ply(path, ...)`
- **Reads:** `x,y,z`, `opacity`, `f_dc_0/1/2`, `f_rest_*`, `scale_*`, `rot_*` (quaternion).
- **Stores:** `_xyz`, `_opacity`, `_features_dc`, `_features_rest`, `_scaling`, `_rotation` (all `nn.Parameter` on CUDA).

Scene is built with a `given_ply_path` so no training-from-images is done before compression when you only want to run the codec.

---

### 2.2 Importance (view-dependent × view-independent)

- **File:** `mesongs.py`
- **Function:** `cal_imp(gaussians, views, pipeline, background)`

**View-dependent:** Renders with **`meson_count=True`**. The rasterizer returns per-Gaussian importance `imp` (contribution to the image, alpha-blended). Summed over all training views:

```python
full_opa_imp += render_results["imp"]   # over train cameras
```

So **Id** (paper) = sum over pixels of α_i ∏_{j<i}(1−α_j) per view, then summed over views.

**View-independent:** Volume **V** = product of scales; normalized by 90th percentile of sorted volumes, clipped, then **Ii = (V_norm)^β**. β is scene-dependent (`beta_list` in `cal_imp`).

**Final:** `imp = v_list * full_opa_imp` (element-wise). So **Ig = Id × Ii** as in the paper.

- **Rasterizer:** `gaussian_renderer/__init__.py`, `render(..., meson_count=True)`. The C++ rasterizer returns `imp`; see `rasterizer(..., opacities=..., ...)` and return dict `"imp": imp`.

**Latency:** One full render per training view; no backward. Dominant cost for importance is **N_train_views × render_time**.

---

### 2.3 Pruning

- **File:** `mesongs.py`
- **Function:** `prune_mask(percent, imp)`
  - Sorts `imp` ascending.
  - Threshold index = `int(percent * (num_points - 1))`; value at that index is the cutoff.
  - Mask = `(imp <= value_nth_percentile)` → these points are **dropped** (mask True = prune).
- **Application:** `gaussians.prune_points(pmask)` (in `scene/gaussian_model.py`).
  - `prune_points` keeps points where `~mask` (valid_points_mask) and updates all parameter tensors and optimizer state.

**Config:** `dataset.percent` comes from `hyper_config` (e.g. `universal_config['prune'][scene_imp]`). Typical values ~0.06–0.5 (prune 6%–50% of points). Paper uses an **importance threshold**; here it’s implemented as a **percentile** of sorted importance.

---

### 2.4 Octree + voxelization + merge (geometry + attribute reorder)

- **File:** `scene/gaussian_model.py`
- **Method:** `octree_coding(self, imp, merge_type, raht=False)`

**2.4.1 Concatenate attributes (pre-merge):**

- Builds one matrix: `[opacity, features_dc (flatten), features_rest (flatten), scaling, rotation]` → shape `(N, 1+3+45+3+4)` for default SH.
- Calls **`create_octree_overall(xyz, features, imp, depth, oct_merge)`** (same file, uses `octreecodes`).

**2.4.2 `octreecodes` (and `create_octree_overall`):**

- **Geometry:** AABB of points; each axis divided into `2^depth` bins via `d1halfing_fast` (linspace). Each point gets a linearized voxel index `ki = otcodex*2^(2d) + otcodey*2^d + otcodez`.
- Points are **sorted by `ki`** (Morton-like order).
- **Deduplication:** `np.unique(ki, return_index=True)`; for each unique voxel, points in that voxel are merged:
  - **merge_type `'mean'`:** mean of positions (and of attributes).
  - **merge_type `'imp'`:** importance-weighted average of positions/attributes.
- Output: unique `ki`, bounding box `(minx,maxx,...)`, and **merged feature matrix** (one row per voxel).

**2.4.3 After octree:**

- **Decode voxels to 3D positions:** `decode_oct(paramarr, oct, depth)` → continuous coordinates from voxel indices (cell centers).
- Model is **replaced** with merged points: `_xyz` from decoded positions; opacity, features_dc, features_rest, scaling, rotation from merged features (slices `features[:, :1]`, `1:4`, `4:4+3*(n_sh-1)`, `49:52`, `52:56]` — indices assume fixed layout).
- If **raht=True:** Build RAHT data structures: `copyAsort(V)` (Morton sort), `haar3D_param(depth, w, val)`, `inv_haar3D_param(V, depth)`. Stored in `self.res`, `self.reorder`, etc., for use in **save** and **load**.

**Important:** RAHT is **not** applied to **scales** in the 8-bit path in this codebase (scales are quantized directly in `save_npz`; see “Important attributes” below). So “RAHT on opacity, Euler, 0-D SH” matches the paper; scales are quantized per-block without RAHT at 8-bit.

**Latency:** CPU numpy (octree, sort, merge). For large scenes, voxelization and merge can be noticeable.

---

### 2.5 Block quantizers (for 8-bit RAHT path)

- **File:** `scene/gaussian_model.py`
- **Method:** `init_qas(n_block)`
  - Allocates `10 * n_block` **VanillaQuan(bit=8)** modules (CUDA).
  - Layout: 7 channels (opacity + Euler×3 + 0-D SH×3) × `n_block` blocks for “rf” (opacity/euler/dc); 3 scale channels × `n_block` for scales → total 10×n_block.

**Calibration:** `VanillaQuan` (in `utils/quant_utils.py`) has no separate calibration phase. In **forward**, it calls `self.update(x)` which sets `min_val`/`max_val` from the input and computes `scale`, `zero_point`. So the **first time** each quantizer sees data (e.g. during the first **ft_render** in `evaluate_test`, or when saving with pre-computed stats), scale/zero_point are set. For **save_npz** with `per_block_quant`, it uses `self.qas[...].scale` and `.zero_point`; if no forward has been run, those buffers can be empty. In practice, **evaluate_test** runs before **save_ft**, so the first test render calibrates all block quantizers. For a **benchmark that skips evaluation**, you’d need either one dummy forward pass over the RAHT coefficients and scales to calibrate qas, or a dedicated calibration function.

---

### 2.6 Vector quantization (1+D SH)

- **File:** `vq.py`
- **Function:** `vq_features(features, importance, codebook_size, vq_chunk, steps, decay, ...)`
  - **features:** `_features_rest.detach().flatten(-2)` (all 1+D SH coefficients).
  - **importance:** passed from caller (same `imp` after pruning, still aligned to current point count after octree).
  - **Algorithm:** Custom **VectorQuantize** with weighted k-means-like updates: `weightedDistance`, scatter for codebook/importance updates, EMA. Runs **`steps`** iterations (default 1000), each with a random batch of size **vq_chunk** (default 2^16).
- **In model:** `gaussians.vq_fe(imp, codebook_size, batch_size, steps)`:
  - Replaces `_features_rest` with the **codebook** (shared) and stores **`_feature_indices`** (one index per point).

**Latency:** `steps × (vq_chunk forward + update)`; can be a large part of encoding time (e.g. 1000 steps).

---

### 2.7 Encode to disk (save_npz / save_full_npz)

- **Entry:** `scene/__init__.py` → `save_ft()` → `gaussians.save_npz(...)` or `save_full_npz(...)`.
- **Reported size:** `save_ft` returns `os.path.getsize(..., "bins.zip")` (zip of the `bins/` directory).

**2.7.1 Rotation → Euler (lossless)**

- In **save_npz** (and save_full_npz): quaternion is normalized, then **ToEulerAngles_FT(q)** (`scene/gaussian_model.py` or renderer). Formula: roll, pitch, yaw from q (w,x,y,z). Stored as 3 floats per point instead of 4.

**2.7.2 “Important” attributes (opacity, Euler, 0-D SH) → RAHT + block quant**

- Build **rf** = `[opacity (logit), euler×3, features_dc×3]` → shape (N, 7).
- Reorder by RAHT order: `C = rf[self.reorder]`.
- **RAHT forward:** For `d in range(depth*3)`, apply `transform_batched_torch(w1, w2, C[left_idx], C[right_idx])` (from `raht_torch`). Result: **C[0]** = DC (one vector per channel); **C[1:]** = AC coefficients.
- **DC:** Stored in float: `cf = C[0].cpu().numpy()` → saved in `orgb.npz` as `f`.
- **AC:** Block quantization per channel:
  - **per_block_quant:** `split = split_length(lc1, n_block)`; for each of the 7 channels, `torch_vanilla_quant_ave(C[1:, i], split, self.qas[qa_cnt : qa_cnt + n_block])` → quantized integers + per-block scale/zero_point appended to **trans_array**.
- **Metadata:** `trans_array` = `[depth, n_block, scale_0, zp_0, ...]` for all blocks and channels. Saved in `t.npz` as `t`.

**2.7.3 Scales (no RAHT at 8-bit)**

- **per_block_quant:** Same idea: `split_scale = split_length(n_points, n_block)`; each of the 3 scale channels is quantized with `torch_vanilla_quant_ave` and corresponding `self.qas`; scales and zero_points go into **trans_array**.
- Quantized scales saved in `ct.npz` as `i` (uint8).

**2.7.4 1+D SH (VQ)**

- **ntk:** `_feature_indices` (int) → `ntk.npz`.
- **umap:** codebook `_features_rest` (float) → `um.npz`.

**2.7.5 Geometry**

- **Octree:** `self.oct`, `self.oct_param` (from octree_coding) → `oct.npz` (`points`, `params`).

**2.7.6 Packing**

- All files under `exp_dir/bins/` are then zipped: `zip -j {bin_zip_path} {bin_dir}/*`. **Reported compressed size** = size of `bins.zip`. So the “entropy coding” here is **np.savez_compressed** (NumPy’s deflate) plus a zip of the whole directory.

**Decode (load_npz):** Reads octree → decode positions and V; loads ntk, umap → rebuild `_features_rest` from indices + codebook; loads orgb (DC + quantized AC), ct (quantized scales), t (metadata); **inverse RAHT** on DC+dequant AC using `inv_haar3D_param` and `itransform_batched_torch`; then assigns `_opacity`, `_euler`, `_features_dc`, `_scaling`. Rendering uses **Euler** (build_rotation_from_euler) when `_euler` is set.

---

## 3. Where to measure for benchmarking

### Encoding latency (training-free, no finetune)

1. **Load PLY** – `load_ply` until return.
2. **Importance** – `cal_imp`: time over all train views (N_views × render).
3. **Prune** – `prune_mask` + `prune_points`: cheap.
4. **Octree + merge** – `octree_coding`: CPU voxelization + merge.
5. **init_qas** – allocation only.
6. **VQ** – `vq_fe`: 1000 steps (or your config).
7. **Calibration of qas** – Either run `evaluate_test` (one full test render pass, which runs ft_render and calibrates qas), or a single forward over RAHT coefficients + scales to populate qas (if you add such a path).
8. **Save** – `save_npz`: RAHT forward, block quant, numpy save, zip. No backward.

So for a **fair encoding benchmark** you should either:
- Include one evaluation pass (so qas are calibrated and you get test metrics), or
- Add a minimal “calibration” pass that runs the quantizers once on current attributes and then call `save_npz`, and time from load to end of `save_npz`.

### Decoding latency

- **load_npz(exp_dir):** Load all npz, decode octree, dequant, inverse RAHT, VQ lookup. Time from start of `load_npz` until all parameters are on GPU and ready for render.

### Compression gain

- **Uncompressed:** Size of the original PLY (or in-memory size of full-precision Gaussians).
- **Compressed:** `bins.zip` size returned by `save_ft` (same as `os.path.getsize(..., "bins.zip")`).
- **Ratio:** uncompressed / compressed. Paper reports sizes after “standard zip”; here the artifact is already a zip of bins.

---

## 4. Config summary (for reproducibility)

- **Prune:** `dataset.percent` (from `hyper_config`, e.g. universal_config `prune[scene]`).
- **Octree depth:** `dataset.depth` (e.g. 14–20 for 360, 12 for Synthetic).
- **Octree merge:** `dataset.oct_merge` (`"mean"` or `"imp"`).
- **RAHT:** `dataset.raht` (True for paper setting).
- **Block quant:** `dataset.per_block_quant` (True); **n_block:** `dataset.n_block` (scene-specific in configs).
- **VQ:** `dataset.codebook_size`, `dataset.batch_size` (vq_chunk), `dataset.steps` (e.g. 1000).
- **Bits:** `dataset.num_bits` (8).
- **Save format:** `pipe.save_ft_type` → `'full_npz'` (single npz) or default (multiple npz in bins, then zip).

---

## 5. File layout (compressed)

Under `point_cloud/iteration_<iter>/pc_npz/bins/`:

- **oct.npz:** `points` (voxel codes), `params` (AABB).
- **ntk.npz:** `ntk` (VQ indices for 1+D SH).
- **um.npz:** `umap` (codebook).
- **orgb.npz:** `f` (DC of opacity+Euler+0-D SH), `i` (quantized AC, uint8).
- **ct.npz:** `i` (quantized scales, uint8).
- **t.npz:** `t` (metadata: depth, n_block, then per-block scale/zero_point for 7 + 3 channels).

Then `bins.zip` = zip of the above. **Reported size** = `bins.zip`.

---

## 6. Summary table (code locations)

| Step | File | Function / method |
|------|------|--------------------|
| Load PLY | scene/gaussian_model.py | load_ply |
| Importance | mesongs.py | cal_imp |
| Importance (render) | gaussian_renderer/__init__.py | render(..., meson_count=True) |
| Prune mask | mesongs.py | prune_mask |
| Prune apply | scene/gaussian_model.py | prune_points |
| Octree + voxelize + merge | scene/gaussian_model.py | octree_coding, octreecodes, create_octree_overall |
| RAHT params | raht_torch | copyAsort, haar3D_param, inv_haar3D_param |
| Block quant init | scene/gaussian_model.py | init_qas |
| Block quant (forward) | utils/quant_utils.py | VanillaQuan.forward, update |
| VQ 1+D SH | vq.py | vq_features; scene/gaussian_model.py vq_fe |
| Quat→Euler | scene/gaussian_model.py | ToEulerAngles_FT |
| RAHT forward | scene/gaussian_model.py (save_npz) | transform_batched_torch (raht_torch) |
| Block quant (save) | scene/gaussian_model.py | torch_vanilla_quant_ave, trans_array |
| Write + zip | scene/gaussian_model.py | save_npz, np.savez_compressed, os.system zip |
| Decode | scene/gaussian_model.py | load_npz, decode_oct, itransform_batched_torch, torch_vanilla_dequant_ave |

Use this to instrument encoding/decoding and to compare with your training-free method on the same PLY inputs and the same quality/size trade-offs (e.g. same prune ratio and bit depth).
