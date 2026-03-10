#!/bin/bash

# Evaluate ImageGS (PNG) compression pipeline for a static Gaussian Splat model.
#
# Step 1: Compress + Decompress (compress_decompress_pipeline.py)
# Step 2: Evaluate quality of GT vs Decompressed (evaluate_decompress.py)
#
# Usage: evaluate_imagegs_compression.sh [OPTIONS]
#   --dataset          Dataset name           (default: db)
#   --scene            Scene name             (default: drjohnson)

DATASET="db"
SCENE="drjohnson"
SH_DEGREE=3
N_CLUSTERS=65536

# DATASET="tandt"
# SCENE="truck"
# SH_DEGREE=3
# N_CLUSTERS=65536

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2";      shift 2 ;;
        --scene)        SCENE="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

DATAPATH="/synology/rajrup/MesonGS/data/${DATASET}/${SCENE}"
PLY_CKPT="/synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}"
OUTPUT_BASE="/synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}/compression/imagegs/default_sort_sh_cluster_${N_CLUSTERS}"
DECOMP_PLY="${OUTPUT_BASE}/decompressed/point_cloud.ply"

MESONGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "${OUTPUT_BASE}"

### 1. ImageGS Compress + Decompress
echo "======================================================================"
echo "Step 1: ImageGS (PNG) Compress + Decompress"
echo "======================================================================"
echo "  Dataset:      ${DATAPATH}"
echo "  PLY ckpt:     ${PLY_CKPT}"
echo "  Output:       ${OUTPUT_BASE}"
echo "  Scene:        ${SCENE}"
echo "======================================================================"

cd "${MESONGS_ROOT}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate mesongs

python scripts/imagegs_baseline/compress_decompress_pipeline.py \
    --ply_path "${PLY_CKPT}" \
    --output_path "${OUTPUT_BASE}" \
    --sh_degree ${SH_DEGREE}

### 2. Evaluate Decompression Quality (PSNR/SSIM/LPIPS vs GT)
echo ""
echo "======================================================================"
echo "Step 2: Evaluate Decompression Quality"
echo "======================================================================"

python scripts/evaluate_decompress.py \
    -s "${DATAPATH}" \
    --ply_path "${PLY_CKPT}" \
    --decompressed_ply_path "${DECOMP_PLY}" \
    --convert_SHs_python \
    --use_indexed \
    --eval \
    --output_path "${OUTPUT_BASE}/evaluation" \
    --save_renders

echo ""
echo "======================================================================"
echo "Done! Results in: ${OUTPUT_BASE}"
echo "======================================================================"
