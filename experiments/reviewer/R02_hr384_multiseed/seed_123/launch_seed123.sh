#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/home/m-sazzad-h/projects/AQUA20-Marine-Classification"
RUN_ROOT="/home/m-sazzad-h/aqua20_reviewer_outputs/R02/seed_123"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/train_console.log"
META_FILE="$RUN_ROOT/launch_metadata.txt"

DATA_ROOT="/home/m-sazzad-h/datasets/AQUA20_OFFICIAL"
SOURCE_CHECKPOINT="/home/m-sazzad-h/aqua20_reviewer_outputs/R01/focal/seed_123/weights/best_model.pth"
OUTPUT_ROOT="/home/m-sazzad-h/aqua20_reviewer_outputs"

mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT" || exit 1
source .venv/bin/activate

if [[ -f "$RUN_ROOT/weights/best_model.pth" || -f "$RUN_ROOT/weights/last_checkpoint.pth" ]]; then
    echo "ABORT: Existing R02 seed123 training checkpoint detected."
    echo "Use the dedicated resume workflow instead of starting a new run."
    exit 2
fi

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

{
    echo "============================================================"
    echo "AQUA20 R02 HR384 — SEED123"
    echo "============================================================"
    echo "Start UTC: $START_UTC"
    echo "Host: $(hostname)"
    echo "Project: $PROJECT_ROOT"
    echo "Python: $(which python)"
    python --version
    echo
    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
      --format=csv,noheader
    echo
    echo "--- SOURCE CHECKPOINT ---"
    echo "$SOURCE_CHECKPOINT"
    sha256sum "$SOURCE_CHECKPOINT"
    echo
    echo "--- TRAINING ---"

    python scripts/reviewer/train_r02_hr384.py \
      --data-root "$DATA_ROOT" \
      --source-checkpoint "$SOURCE_CHECKPOINT" \
      --output-root "$OUTPUT_ROOT"
} 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
RUNTIME_SECONDS=$((END_EPOCH - START_EPOCH))

{
    echo "start_utc=$START_UTC"
    echo "end_utc=$END_UTC"
    echo "runtime_seconds=$RUNTIME_SECONDS"
    echo "exit_code=$EXIT_CODE"
    echo "log_file=$LOG_FILE"
} > "$META_FILE"

{
    echo
    echo "============================================================"
    echo "PROCESS END"
    echo "End UTC: $END_UTC"
    echo "Runtime: $RUNTIME_SECONDS sec"
    echo "Exit code: $EXIT_CODE"
    echo "============================================================"
} | tee -a "$LOG_FILE"

exit "$EXIT_CODE"
