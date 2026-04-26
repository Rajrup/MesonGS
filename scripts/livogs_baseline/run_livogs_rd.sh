#!/bin/bash
# Run LiVoGS RD (rate-distortion) sweep for static Gaussian Splat models.
#
# For each configuration row in a parameter CSV, runs:
#   1. LiVoGS compress + decompress  (compress_decompress_pipeline.py)
#   2. Quality evaluation             (evaluate_decompress.py)
# Then collects all results into a single output CSV.
#
# Usage:
#   bash scripts/livogs_baseline/run_livogs_rd.sh

set -eo pipefail

MESONGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${MESONGS_ROOT}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate mesongs

TRAIN_OUTPUT="/synology/rajrup/MesonGS/train_output"
DATA_ROOT="/synology/rajrup/MesonGS/data"
SH_DEGREE=3
RLGR_BLOCK_SIZE=4096
NVCOMP_ALGORITHM="ANS"

# ---------------------------------------------------------------------------
# Dataset definitions: (dataset, scene, config_csv_path)
# ---------------------------------------------------------------------------
declare -a DATASETS=(
    "db|drjohnson|scripts/livogs_baseline/rd_convex_hull_db_drjohnson.csv"
    "tandt|truck|scripts/livogs_baseline/rd_convex_hull_tandt_truck.csv"
)

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET SCENE CONFIG_CSV <<< "$entry"

    DATAPATH="${DATA_ROOT}/${DATASET}/${SCENE}"
    PLY_CKPT="${TRAIN_OUTPUT}/${DATASET}/${SCENE}"
    RD_OUTPUT_DIR="${TRAIN_OUTPUT}/${DATASET}/${SCENE}/compression/livogs_rd"
    mkdir -p "${RD_OUTPUT_DIR}"

    echo ""
    echo "######################################################################"
    echo "# RD Sweep: ${DATASET}/${SCENE}"
    echo "# Config CSV: ${CONFIG_CSV}"
    echo "# Output:     ${RD_OUTPUT_DIR}"
    echo "######################################################################"

    # Read CSV header (skip it), then iterate rows
    ROW_IDX=0
    TOTAL_ROWS=$(tail -n +2 "${CONFIG_CSV}" | grep -c '^' || true)

    while IFS=',' read -r config_dir J qpq qps qpo qpdc qpac sh_color_space; do
        # Strip trailing \r from Windows line endings
        sh_color_space="${sh_color_space%$'\r'}"
        ROW_IDX=$((ROW_IDX + 1))

        OUTPUT_BASE="${TRAIN_OUTPUT}/${DATASET}/${SCENE}/compression/livogs/${config_dir}"
        DECOMP_PLY="${OUTPUT_BASE}/decompressed/point_cloud.ply"

        echo ""
        echo "======================================================================"
        echo "[${ROW_IDX}/${TOTAL_ROWS}] ${config_dir}"
        echo "  J=${J} qpq=${qpq} qps=${qps} qpo=${qpo} qpdc=${qpdc} qpac=${qpac} cs=${sh_color_space}"
        echo "======================================================================"

        # Skip if both result JSONs already exist
        COMP_JSON="${OUTPUT_BASE}/compression_stats.json"
        EVAL_JSON="${OUTPUT_BASE}/evaluation/evaluation_results.json"
        if [[ -f "${COMP_JSON}" && -f "${EVAL_JSON}" ]]; then
            echo "  -> Results already exist, skipping."
            continue
        fi

        # Step 1: Compress + Decompress
        echo "  Step 1: Compress + Decompress"
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
            --sh_color_space ${sh_color_space} \
            --rlgr_block_size ${RLGR_BLOCK_SIZE} \
            --nvcomp_algorithm ${NVCOMP_ALGORITHM}

        # Step 2: Evaluate
        echo "  Step 2: Evaluate quality"
        python scripts/evaluate_decompress.py \
            -s "${DATAPATH}" \
            --ply_path "${PLY_CKPT}" \
            --decompressed_ply_path "${DECOMP_PLY}" \
            --convert_SHs_python \
            --use_indexed \
            --eval \
            --output_path "${OUTPUT_BASE}/evaluation"

    done < <(tail -n +2 "${CONFIG_CSV}")

    # ------------------------------------------------------------------
    # Collect results into a single CSV
    # ------------------------------------------------------------------
    echo ""
    echo "======================================================================"
    echo "Collecting results for ${DATASET}/${SCENE} ..."
    echo "======================================================================"

    python - "${CONFIG_CSV}" "${TRAIN_OUTPUT}/${DATASET}/${SCENE}/compression/livogs" "${RD_OUTPUT_DIR}" <<'PYEOF'
import sys, os, csv, json

config_csv = sys.argv[1]
livogs_base = sys.argv[2]
rd_output_dir = sys.argv[3]

with open(config_csv) as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

out_fields = fieldnames + [
    "psnr", "ssim", "lpips",
    "compressed_size_bytes", "uncompressed_size_bytes",
    "compressed_size_mb", "uncompressed_size_mb",
    "encode_time_ms", "decode_time_ms",
]

out_rows = []
for row in rows:
    config_dir = row["config_dir"]
    comp_json = os.path.join(livogs_base, config_dir, "compression_stats.json")
    eval_json = os.path.join(livogs_base, config_dir, "evaluation", "evaluation_results.json")

    out_row = dict(row)

    if os.path.isfile(comp_json) and os.path.isfile(eval_json):
        with open(comp_json) as f:
            comp = json.load(f)
        with open(eval_json) as f:
            evl = json.load(f)

        out_row["psnr"] = evl["decompressed"]["psnr"]
        out_row["ssim"] = evl["decompressed"]["ssim"]
        out_row["lpips"] = evl["decompressed"]["lpips"]
        out_row["compressed_size_bytes"] = comp["compressed_size_bytes"]
        out_row["uncompressed_size_bytes"] = comp["uncompressed_size_bytes"]
        out_row["compressed_size_mb"] = f"{comp['compressed_size_bytes'] / (1024*1024):.4f}"
        out_row["uncompressed_size_mb"] = f"{comp['uncompressed_size_bytes'] / (1024*1024):.4f}"
        out_row["encode_time_ms"] = f"{comp['encode_time_ms']:.2f}"
        out_row["decode_time_ms"] = f"{comp['decode_time_ms']:.2f}"
        print(f"  {config_dir}: PSNR={out_row['psnr']:.2f}  SSIM={out_row['ssim']:.4f}  "
              f"Size={float(out_row['compressed_size_mb']):.2f} MB")
    else:
        for k in out_fields:
            if k not in out_row:
                out_row[k] = ""
        print(f"  {config_dir}: MISSING results")

    out_rows.append(out_row)

os.makedirs(rd_output_dir, exist_ok=True)
out_csv = os.path.join(rd_output_dir, "rd_results.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=out_fields)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\n  Saved: {out_csv} ({len(out_rows)} rows)")
PYEOF

done

echo ""
echo "######################################################################"
echo "# All RD sweeps complete."
echo "######################################################################"
