# AQUA20 Reviewer Response Experiments

This directory contains new reviewer-response experiments that are intentionally separate from the original AQUA20 campaign.

## Purpose

The reviewer-response branch is intended to support targeted follow-up analysis and controlled replication checks without altering the original scientific evidence from the frozen F01–F09 campaign.

## Frozen original campaign

- F01–F09 remain frozen scientific records.
- The original AQUA20 train/validation/test split remains unchanged.
- No reviewer experiment should overwrite or rewrite original campaign outputs.

## Reviewer experiments

- R01 = controlled original-distribution ConvNeXt-Small CE vs Gentle Class-Balanced Focal multi-seed comparison.
- R02 = repeated-seed HR384 continuation.

R01 seed layout on Azure:

- `R01/ce/seed_42`
- `R01/ce/seed_123`
- `R01/ce/seed_2026`
- `R01/focal/seed_123`
- `R01/focal/seed_2026`

Seed 42 for focal is a frozen reuse of the original F03 checkpoint and is not retrained.

## Reviewer analyses

Reviewer analyses include:

- calibration metrics;
- runtime and inference benchmarking;
- class-count verification;
- external class-mapping checks;
- duplicate auditing for AQUA20 vs external data.

## Local CPU workflow

The local Windows workstation is intended for CPU-friendly reviewer work, including:

- result aggregation;
- statistics and mean/std calculations;
- calibration from saved probability files;
- mapping verification;
- duplicate audit review;
- reviewer tables and figures.

## Azure GPU workflow

The Azure Linux NVIDIA GPU VM is intended for GPU-side execution, including:

- R01 training;
- R02 HR384 continuation;
- model inference requiring GPU;
- runtime benchmarking.

Recommended Azure install command:

```bash
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu118 -r requirements/gpu.txt
pip install -r requirements/base.txt
```

Launch R01 sequentially inside `tmux` so SSH disconnects do not kill the run:

```bash
tmux new -s aqua20-r01
bash scripts/reviewer/run_r01_sequential.sh
```

The `seed_42` focal run reuses the frozen F03 checkpoint rather than retraining.

If a run is interrupted, rerun the launcher and it will resume any seed that already has a last checkpoint.

## Execution portability

Reviewer scripts and notebooks must run without hard-coded Kaggle or Windows filesystem paths. Use repository-relative paths via `pathlib.Path` and environment variables such as `AQUA20_DATA_ROOT`, `AQUA20_F03_ROOT`, and `AQUA20_OUTPUT_ROOT`.

## Guardrail

All reviewer artifacts live under this directory and must remain clearly separated from the original experiments and notebooks so that the official AQUA20 campaign remains intact.
