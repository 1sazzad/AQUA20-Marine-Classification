#!/usr/bin/env bash

set -euo pipefail

cat >&2 <<'EOF'
Automatic sequential R01 execution is disabled.

Reviewer experiments must be launched one at a time after explicit approval.
The only currently authorized run is R01 cross-entropy seed 42; launch it
directly with scripts/reviewer/train_r01.py after completing the documented
environment, dataset, GPU, and sanity checks.

This guard intentionally does not start or resume any training run.
EOF

exit 2
