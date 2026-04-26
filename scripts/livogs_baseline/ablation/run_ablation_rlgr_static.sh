#!/bin/bash
# Run RLGR ablation study on static Gaussian Splat models (drjohnson & truck).
#
# Usage:
#   bash scripts/livogs_baseline/ablation/run_ablation_rlgr_static.sh [OPTIONS]
#     --repetitions   Number of benchmark repetitions per variant (default: 50)
#     --format        Plot output format: pdf or png (default: pdf)

set -eo pipefail

REPETITIONS=50
FMT="pdf"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repetitions)  REPETITIONS="$2"; shift 2 ;;
        --format)       FMT="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

MESONGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${MESONGS_ROOT}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate mesongs

TRAIN_OUTPUT="/synology/rajrup/MesonGS/train_output"

# ======================================================================
# Dataset 1: Deep Blending / drjohnson
# ======================================================================
DATASET="db"
SCENE="drjohnson"
J=14
qpq=0.005; qps=0.001; qpo=0.01; qpdc=0.01; qpac=0.05

PLY_CKPT="${TRAIN_OUTPUT}/${DATASET}/${SCENE}"
OUTPUT_FOLDER="${PLY_CKPT}/ablation/livogs_rlgr"

echo "======================================================================"
echo "RLGR Ablation (Static): ${DATASET}/${SCENE}"
echo "  PLY ckpt:     ${PLY_CKPT}"
echo "  Output:       ${OUTPUT_FOLDER}"
echo "  J=${J}  qpq=${qpq} qps=${qps} qpo=${qpo} qpdc=${qpdc} qpac=${qpac}"
echo "  Repetitions:  ${REPETITIONS}"
echo "======================================================================"

python scripts/livogs_baseline/ablation/ablation_rlgr_static.py \
    --ply_path "${PLY_CKPT}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --repetitions ${REPETITIONS} \
    --J ${J} \
    --qpq ${qpq} --qps ${qps} --qpo ${qpo} --qpdc ${qpdc} --qpac ${qpac} \
    --sh_color_space klt \
    --nvcomp_algorithm ANS

echo ""
echo "Generating plots for ${DATASET}/${SCENE}..."
python scripts/livogs_baseline/ablation/plot_ablation_rlgr.py \
    --input_csv "${OUTPUT_FOLDER}/ablation_rlgr.csv" \
    --output_folder "${OUTPUT_FOLDER}/plots" \
    --format ${FMT}

# ======================================================================
# Dataset 2: Tanks & Temples / truck
# ======================================================================
DATASET="tandt"
SCENE="truck"
J=15
qpq=0.005; qps=0.001; qpo=0.01; qpdc=0.01; qpac=0.05

PLY_CKPT="${TRAIN_OUTPUT}/${DATASET}/${SCENE}"
OUTPUT_FOLDER="${PLY_CKPT}/ablation/livogs_rlgr"

echo ""
echo "======================================================================"
echo "RLGR Ablation (Static): ${DATASET}/${SCENE}"
echo "  PLY ckpt:     ${PLY_CKPT}"
echo "  Output:       ${OUTPUT_FOLDER}"
echo "  J=${J}  qpq=${qpq} qps=${qps} qpo=${qpo} qpdc=${qpdc} qpac=${qpac}"
echo "  Repetitions:  ${REPETITIONS}"
echo "======================================================================"

python scripts/livogs_baseline/ablation/ablation_rlgr_static.py \
    --ply_path "${PLY_CKPT}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --repetitions ${REPETITIONS} \
    --J ${J} \
    --qpq ${qpq} --qps ${qps} --qpo ${qpo} --qpdc ${qpdc} --qpac ${qpac} \
    --sh_color_space klt \
    --nvcomp_algorithm ANS

echo ""
echo "Generating plots for ${DATASET}/${SCENE}..."
python scripts/livogs_baseline/ablation/plot_ablation_rlgr.py \
    --input_csv "${OUTPUT_FOLDER}/ablation_rlgr.csv" \
    --output_folder "${OUTPUT_FOLDER}/plots" \
    --format ${FMT}

# ======================================================================
# Cross-dataset comparison plot
# ======================================================================
echo ""
echo "======================================================================"
echo "Generating cross-dataset comparison plots..."
echo "======================================================================"
python scripts/livogs_baseline/ablation/plot_ablation_rlgr_compare.py \
    --output_folder scripts/livogs_baseline/ablation/plots \
    --format ${FMT}

echo ""
echo "======================================================================"
echo "Done! All ablation results saved."
echo "======================================================================"
