# LiVoGS Baseline — RD Experiment Pipeline

Three scripts for running rate-distortion experiments, collecting results, and plotting RD curves.

## Scripts

### 1. `run_rd_pipeline.py` — Run experiments

Generates QP config JSONs and evaluates compression across all configured (sequence, frame, depth, QP) combinations.

```bash
python scripts/livogs_baseline/run_rd_pipeline.py
```

**Key configuration (edit in-file):**

| Variable | Description |
|---|---|
| `SEQUENCES` | List of `{dataset_name, sequence_name, qp_dir_name}` dicts |
| `FRAME_IDS` | Frame indices to evaluate |
| `EXPERIMENT_DEPTHS` | Octree depths (J values) |
| `EXPERIMENT_SH_QPS` | SH quantization parameters |
| `EXPERIMENT_BETA_VALUES` | Beta values for rate control |
| `EXPERIMENT_QP_QUATS/SCALES/OPACITY` | Attribute QP sweep values |
| `STAGE2_GPUS` | GPU IDs for parallel evaluation |
| `STAGE2_WORKERS_PER_GPU` | Concurrent workers per GPU |
| `SKIP_SAVED_EXPERIMENTS` | Skip already-completed experiments |

**Output structure:**
```
{DATA_PATH}/train_output/{dataset}/{sequence}/compression/livogs_rd/
  frame_{id}/
    J_{depth}/
      {label}/
        qp_config.json
        benchmark_livogs.csv
        livogs_config.json
        evaluation/
          evaluation_results.json
```

### 2. `collect_rd_results.py` — Collect results into CSV

Walks experiment directory trees and produces a single CSV with one row per experiment.

```bash
python scripts/livogs_baseline/collect_rd_results.py
```

**Key configuration (edit in-file):**

| Variable | Description |
|---|---|
| `RD_OUTPUT_ROOTS` | List of `{path, name?, frame_ids?}` dicts pointing to `livogs_rd/` directories |
| `OUTPUT_CSV` | Output path (default: `{first_root}/collected_rd_results.csv`) |
| `MAX_WORKERS` | Thread pool size for parallel I/O (default: 32) |

**CSV columns:**

`sequence_name`, `frame_id`, `depth`, `qp_sh`, `beta`, `qp_opacity`, `qp_scales`, `qp_quats`, `gt_psnr`, `gt_ssim`, `decomp_psnr`, `decomp_ssim`, `psnr_drop`, `ssim_drop`, `size_bytes`, `compressed_mb`, `label`

**Cross-project support:** Works with both VideoGS (`qp_sh` in QP configs) and Queen (`qp_sh`) directory layouts.

### 3. `plot_rd_results.py` — Collect + plot RD curves

Collects results (via `collect_rd_results.py`), then generates RD-curve PNGs.

```bash
python scripts/livogs_baseline/plot_rd_results.py
```

**Key configuration (edit in-file):**

| Variable | Description |
|---|---|
| `RD_OUTPUT_ROOTS` | Same format as `collect_rd_results.py` |
| `PLOT_GROUPS` | Inline plot specifications |
| `PLOT_CONFIG_JSONS` | Paths to JSON plot config files |
| `PLOT_OUTPUT_DIR` | Output directory for PNGs (default: `{first_root}/plots/`) |
| `COLLECTED_CSV` | CSV path (default: `{first_root}/collected_rd_results.csv`) |

**Plot config format** (JSON or inline):

```json
[
  {
    "psnr_range": [30, 40],
    "plots": [
      {
        "curve_var": "qp_opacity",
        "fixed": {"depth": 12, "beta": 0.0, "qp_quats": 0.0001, "qp_scales": 0.0001}
      }
    ]
  }
]
```

- `curve_var` — the knob to sweep (becomes separate RD curves). Valid: `depth`, `qp_sh`, `beta`, `qp_quats`, `qp_scales`, `qp_opacity`
- `fixed` — exact-match filters applied before plotting
- `psnr_range` — optional `[min, max]` for y-axis

See `plot_configs/default.json` for a working example.

## Typical workflow

```bash
# 1. Run experiments (hours/days, GPU-intensive)
python scripts/livogs_baseline/run_rd_pipeline.py

# 2. Plot results (fast, CPU-only)
python scripts/livogs_baseline/plot_rd_results.py

# Or just collect CSV without plotting
python scripts/livogs_baseline/collect_rd_results.py
```

## Directory layout

```
scripts/livogs_baseline/
├── run_rd_pipeline.py        # Experiment runner
├── collect_rd_results.py     # Result collector → CSV
├── plot_rd_results.py        # Collector + plotter
├── plot_configs/
│   └── default.json          # Default plot specifications
└── rd_pipeline/              # Shared internals
    ├── config.py             # Paths, constants, knob names
    ├── qp.py                 # QP config generation
    ├── codec.py              # Compression/decompression
    ├── plot.py               # RD curve plotting
    └── worker.py             # GPU worker subprocess
```

## Configuration style

All scripts use **edit-and-run** configuration — global variables at the top of each file, no CLI arguments. Edit the `Global configuration` section, then run.
