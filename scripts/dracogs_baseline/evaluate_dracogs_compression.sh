#!/bin/bash

# Evaluate DracoGS compression pipeline for VideoGS-trained models
#
# Usage: evaluate_dracogs_compression.sh [OPTIONS]
#   --dataset_name     Dataset name           (default: HiFi4G_Dataset)
#   --sequence_name    Sequence name          (default: 4K_Actor1_Greeting)
#   --resolution       Resolution scale       (default: 2)
#   --frame_start      Start frame            (default: 0)
#   --frame_end        End frame              (default: 200)
#   --interval         Frame interval         (default: 10)
#   --qp               Position quantization  (default: 16)
#   --qfd              SH DC quantization     (default: 16)
#   --qfr1             SH band1 quantization  (default: 16)
#   --qfr2             SH band2 quantization  (default: 16)
#   --qfr3             SH band3 quantization  (default: 16)
#   --qo               Opacity quantization   (default: 16)
#   --qs               Scale quantization     (default: 16)
#   --qr               Rotation quantization  (default: 16)
#   --cl               Compression level      (default: 7)

DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"
RESOLUTION=2

START_FRAME=0
END_FRAME=200
INTERVAL=10
SH_DEGREE=3

# LTS quantization parameters
EG=16
EO=16
ET=16
ES=16
CL=10

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name)    DATASET_NAME="$2";    shift 2 ;;
        --sequence_name)   SEQUENCE_NAME="$2";   shift 2 ;;
        --resolution)      RESOLUTION="$2";      shift 2 ;;
        --frame_start)     START_FRAME="$2";     shift 2 ;;
        --frame_end)       END_FRAME="$2";       shift 2 ;;
        --interval)        INTERVAL="$2";        shift 2 ;;
        --eg)              EG="$2";              shift 2 ;;
        --eo)              EO="$2";              shift 2 ;;
        --et)              ET="$2";              shift 2 ;;
        --es)              ES="$2";              shift 2 ;;
        --cl)              CL="$2";              shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

data_path="/synology/rajrup/VideoGS"
dataset_path="${data_path}/${DATASET_NAME}_processed/${SEQUENCE_NAME}"
gt_model_path="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/checkpoint"

# Build output folder name from quantization parameters
output_tag="eg_${EG}_eo_${EO}_et_${ET}_es_${ES}_cl_${CL}"
output_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/compression/dracogs/${output_tag}"

VIDEOGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

### 1. DracoGS Compress + Decompress (in videogs conda env)
echo "======================================================================"
echo "Step 1: DracoGS Compress + Decompress"
echo "======================================================================"
echo "  Dataset:      ${dataset_path}"
echo "  GT model:     ${gt_model_path}"
echo "  Output:       ${output_folder}"
echo "  Scene:        ${SEQUENCE_NAME}"
echo "  Quant:        eg=${EG} eo=${EO} et=${ET} es=${ES} cl=${CL}"
echo "======================================================================"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate videogs
cd "${VIDEOGS_ROOT}"

python scripts/dracogs_baseline/compress_decompress_pipeline.py \
    --ply_path "${gt_model_path}" \
    --output_folder "${output_folder}" \
    --output_ply_folder "${output_folder}/decompressed_ply" \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL} \
    --sh_degree ${SH_DEGREE} \
    --scene_name "${SEQUENCE_NAME}" \
    --eg ${EG} --eo ${EO} --et ${ET} --es ${ES} \
    --cl ${CL}

### 2. Evaluate Decompression Quality (PSNR/SSIM vs GT)
echo ""
echo "======================================================================"
echo "Step 2: Evaluate Decompression Quality"
echo "======================================================================"

python scripts/evaluate_decompress.py \
    --gt_ply_path "${gt_model_path}" \
    --decompressed_ply_path "${output_folder}/decompressed_ply" \
    --dataset_path "${dataset_path}" \
    --output_render_path "${output_folder}/evaluation" \
    --save_renders \
    --sh_degree ${SH_DEGREE} \
    --resolution ${RESOLUTION} \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL}

echo ""
echo "======================================================================"
echo "Done! Results in: ${output_folder}"
echo "======================================================================"
