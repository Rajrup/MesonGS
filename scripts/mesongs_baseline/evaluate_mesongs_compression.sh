#!/bin/bash

# Evaluate MesonGS compression pipeline for a static Gaussian Splat model.
#
# Step 1: Compress + Decompress (compress_decompress_pipeline.py)
# Step 2: Evaluate quality of GT vs Decompressed (evaluate_decompress.py)
#
# Usage: evaluate_mesongs_compression.sh [OPTIONS]
#   --dataset        Dataset name       (default: db)
#   --scene          Scene name         (default: drjohnson)
#   --config         Hyper config       (default: config3)
#   --prune          Enable pruning     (flag)

DATASET="db"
SCENE="drjohnson"
CONFIG="config3"

# DATASET="tandt"
# SCENE="truck"
# CONFIG="config3"

# PRUNE_FLAG=""
PRUNE_FLAG="--prune"
PRUNE_TAG="no"

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   DATASET="$2";  shift 2 ;;
        --scene)     SCENE="$2";    shift 2 ;;
        --config)    CONFIG="$2";   shift 2 ;;
        --prune)     PRUNE_FLAG="--prune"; PRUNE_TAG="yes"; shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

DATAPATH="/synology/rajrup/MesonGS/data/${DATASET}/${SCENE}"
PLY_CKPT="/synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}"
OUTPUT_BASE="/synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}/compression/mesongs/${CONFIG}_prune_${PRUNE_TAG}"
DECOMP_PLY="${OUTPUT_BASE}/decompressed/point_cloud.ply"

MESONGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LSEG=0
CB=0
DEPTH=0

mkdir -p "${OUTPUT_BASE}"

### 1. MesonGS Compress + Decompress
echo "======================================================================"
echo "Step 1: MesonGS Compress + Decompress"
echo "======================================================================"
echo "  Dataset:      ${DATAPATH}"
echo "  PLY ckpt:     ${PLY_CKPT}"
echo "  Output:       ${OUTPUT_BASE}"
echo "  Scene:        ${SCENE}"
echo "  Config:       ${CONFIG}"
echo "======================================================================"

cd "${MESONGS_ROOT}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate mesongs

python scripts/mesongs_baseline/compress_decompress_pipeline.py \
    -s "${DATAPATH}" \
    --ply_path "${PLY_CKPT}" \
    --num_bits 8 \
    --convert_SHs_python \
    --percent 0 \
    --codebook_size ${CB} \
    --steps 1000 \
    --scene_imp "${SCENE}" \
    --depth ${DEPTH} \
    --raht \
    --clamp_color \
    --per_block_quant \
    --lseg ${LSEG} \
    --use_indexed \
    --debug \
    --hyper_config "${CONFIG}" \
    --eval \
    --output_path "${OUTPUT_BASE}" \
    ${PRUNE_FLAG}

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
