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

## Execution gate

Reviewer training experiments are performed and verified **one at a time**.
Every training start or resume requires a separate explicit instruction. The
automatic sequential launcher is disabled and must not be used to continue to
later seeds.

The current and only authorized experiment is:

- R01 standard cross-entropy, seed 42.

Do not start CE seeds 123 or 2026, focal seeds 123 or 2026, R02 HR384,
calibration, or duplicate auditing. Keep their infrastructure in place for
later use. The original F01-F09 campaign remains completely frozen.

Before launching CE seed 42 on Azure, complete these checks in order:

1. Finish repository and Python environment setup.
2. Verify CUDA and the Tesla T4 with `python scripts/check_gpu.py`.
3. Locate or transfer the fixed AQUA20 dataset.
4. Run `python scripts/check_reviewer_environment.py` and confirm train =
   5,247, validation = 1,312, test = 1,612, and classes = 20.
5. Run the CE seed-42 sanity check:

   ```bash
   python scripts/reviewer/train_r01.py --loss ce --seed 42 --sanity-check
   ```

6. Start only CE seed 42 inside `tmux`:

```bash
tmux new -s aqua20-r01
mkdir -p "$AQUA20_OUTPUT_ROOT/R01/ce/seed_42"
python scripts/reviewer/train_r01.py --loss ce --seed 42 2>&1 | tee "$AQUA20_OUTPUT_ROOT/R01/ce/seed_42/training.log"
```

If a verified CE seed-42 run is interrupted, resume that run only after an
explicit instruction:

```bash
python scripts/reviewer/train_r01.py --loss ce --seed 42 --resume 2>&1 | tee -a "$AQUA20_OUTPUT_ROOT/R01/ce/seed_42/training.log"
```

Confirm that epoch metrics are printed, `tables/training_history.csv` updates,
and both `weights/best_model.pth` and `weights/last_checkpoint.pth` are saved in
the CE seed-42 run directory. Verify the resume path and log locations, then
stop. Do not begin another seed when CE seed 42 finishes.

The `seed_42` focal run remains a frozen reuse of the original F03 checkpoint
rather than a retraining run, but it is not part of the current execution.

## Execution portability

Reviewer scripts and notebooks must run without hard-coded Kaggle or Windows filesystem paths. Use repository-relative paths via `pathlib.Path` and environment variables such as `AQUA20_DATA_ROOT`, `AQUA20_F03_ROOT`, and `AQUA20_OUTPUT_ROOT`.

## Guardrail

All reviewer artifacts live under this directory and must remain clearly separated from the original experiments and notebooks so that the official AQUA20 campaign remains intact.
