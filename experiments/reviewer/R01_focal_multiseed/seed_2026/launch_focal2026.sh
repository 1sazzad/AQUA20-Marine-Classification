#!/usr/bin/env bash
set -uo pipefail

cd /home/m-sazzad-h/projects/AQUA20-Marine-Classification
source .venv/bin/activate

set -a
source .env
set +a

RUN_ROOT="$AQUA20_OUTPUT_ROOT/R01/focal/seed_2026"
LOG="$RUN_ROOT/logs/train_console.log"
mkdir -p "$RUN_ROOT/logs"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "R01 GENTLE CB FOCAL SEED 2026 — ORIGINAL DISTRIBUTION"
echo "Start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "============================================================"

python scripts/check_gpu.py
echo
nvidia-smi
echo

START_SECONDS=$(date +%s)

python -u scripts/reviewer/train_r01.py \
  --loss focal \
  --seed 2026 \
  --data-root "$AQUA20_DATA_ROOT" \
  --output-root "$AQUA20_OUTPUT_ROOT" \
  --max-epochs 40 \
  --patience 8 \
  --batch-size 16 \
  --workers 4 \
  --prefetch-factor 2 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --image-size 224 \
  --beta 0.999 \
  --gamma 1.0 \
  --max-grad-norm 1.0

STATUS=$?

END_SECONDS=$(date +%s)

echo
echo "============================================================"
echo "End: $(date --iso-8601=seconds)"
echo "Total runtime seconds: $((END_SECONDS - START_SECONDS))"
echo "Training exit code: $STATUS"
echo "============================================================"

nvidia-smi

exit "$STATUS"
