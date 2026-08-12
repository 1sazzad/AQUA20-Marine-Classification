#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_one() {
	local loss="$1"
	local seed="$2"
	local run_root="${AQUA20_OUTPUT_ROOT:?}/R01/${loss}/seed_${seed}"
	local checkpoint="${run_root}/weights/last_checkpoint.pth"

	if [[ -f "$checkpoint" ]]; then
		python scripts/reviewer/train_r01.py --loss "$loss" --seed "$seed" --resume
	else
		python scripts/reviewer/train_r01.py --loss "$loss" --seed "$seed"
	fi
}

run_one ce 42
run_one ce 123
run_one ce 2026
run_one focal 123
run_one focal 2026