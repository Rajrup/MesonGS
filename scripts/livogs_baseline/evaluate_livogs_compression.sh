#!/bin/bash

# Evaluate LiVoGS compression pipeline for a static Gaussian Splat model.
#
# Step 1: Compress + Decompress (compress_decompress_pipeline.py)
# Step 2: Evaluate quality of GT vs Decompressed (evaluate_decompress.py)
#
# Usage: evaluate_livogs_compression.sh [OPTIONS]
#   --dataset          Dataset name           (default: db)
#   --scene            Scene name             (default: drjohnson)
#   --j                Octree depth           (default: 15)
#   --qp               Quantization step      (default: 0.0001)
#   --sh_color_space   Color space            (default: klt)
#   --nvcomp           nvCOMP algorithm       (default: ANS, 'None' to disable)

DATASET="db"
SCENE="drjohnson"
SH_DEGREE=3

# LiVoGS compression parameters
J=14        # At octree depth > 18, ANS fails. There is some ANS limitation.
qpq=0.005
qps=0.001
qpo=0.01
qpdc=0.01
qpac=0.05

# DATASET="tandt"
# SCENE="truck"
# SH_DEGREE=3

# # LiVoGS compression parameters
# J=15        # At octree depth > 18, ANS fails. There is some ANS limitation.
# qpq=0.005
# qps=0.001
# qpo=0.01
# qpdc=0.01
# qpac=0.05

SH_COLOR_SPACE="klt"
RLGR_BLOCK_SIZE=4096
NVCOMP_ALGORITHM="ANS"

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)        DATASET="$2";          shift 2 ;;
        --scene)          SCENE="$2";            shift 2 ;;
        --j)              J="$2";                shift 2 ;;
        --qpq)            qpq="$2";              shift 2 ;;
        --qps)            qps="$2";              shift 2 ;;
        --qpo)            qpo="$2";              shift 2 ;;
        --qpdc)           qpdc="$2";             shift 2 ;;
        --qpac)           qpac="$2";             shift 2 ;;
        --sh_color_space) SH_COLOR_SPACE="$2";   shift 2 ;;
        --nvcomp)         NVCOMP_ALGORITHM="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

DATAPATH="/synology/rajrup/MesonGS/data/${DATASET}/${SCENE}"
PLY_CKPT="/synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}"
OUTPUT_BASE="/synology/rajrup/MesonGS/train_output/${DATASET}/${SCENE}/compression/livogs/J_${J}_qpq_${qpq}_qps_${qps}_qpo_${qpo}_qpdc_${qpdc}_qpac_${qpac}_${SH_COLOR_SPACE}_nvcomp_${NVCOMP_ALGORITHM}"
DECOMP_PLY="${OUTPUT_BASE}/decompressed/point_cloud.ply"

MESONGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "${OUTPUT_BASE}"

### 1. LiVoGS Compress + Decompress
echo "======================================================================"
echo "Step 1: LiVoGS Compress + Decompress"
echo "======================================================================"
echo "  Dataset:      ${DATAPATH}"
echo "  PLY ckpt:     ${PLY_CKPT}"
echo "  Output:       ${OUTPUT_BASE}"
echo "  Scene:        ${SCENE}"
echo "  J:            ${J}"
echo "  Quantize:     qpq=${qpq} qps=${qps} qpo=${qpo} qpdc=${qpdc} qpac=${qpac}"
echo "  Color space:  ${SH_COLOR_SPACE}"
echo "  nvCOMP:       ${NVCOMP_ALGORITHM}"
echo "======================================================================"

cd "${MESONGS_ROOT}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate mesongs

python scripts/livogs_baseline/compress_decompress_pipeline.py \
    --ply_path "${PLY_CKPT}" \
    --output_path "${OUTPUT_BASE}" \
    --sh_degree ${SH_DEGREE} \
    --J ${J} \
    --qpq ${qpq} \
    --qps ${qps} \
    --qpo ${qpo} \
    --qpdc ${qpdc} \
    --qpac ${qpac} \
    --sh_color_space ${SH_COLOR_SPACE} \
    --rlgr_block_size ${RLGR_BLOCK_SIZE} \
    --nvcomp_algorithm ${NVCOMP_ALGORITHM}

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
