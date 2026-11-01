#!/usr/bin/env bash
# Test a declared direct-action interpretation with the SAME frozen head/base assets.
# v2 head-off results are reused only after source-equivalence checks pass.
set -euo pipefail

A2C2_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$A2C2_PROJECT_ROOT"
A2C2_PYTHON="$A2C2_PROJECT_ROOT/.venv-realtime/bin/python"
A2C2_SERIES="${1:-artifacts/realtime-vla/a2c2-author-series-v3}"
A2C2_REFERENCE="${2:-artifacts/realtime-vla/a2c2-author-series-v2}"
A2C2_BRIDGE="${3:-artifacts/realtime-vla/pi05-ros-system}"

"$A2C2_PYTHON" -m experiments.realtime_vla.summarize_a2c2 \
  --series "$A2C2_SERIES" --reference-series "$A2C2_REFERENCE" \
  --bridge "$A2C2_BRIDGE" --register-only

"$A2C2_PYTHON" -m experiments.realtime_vla.evaluate_a2c2 \
  --variant a2c2 --decode-mode direct --protocol-version 3 --initial-states 0 \
  --control-hz 20 --wall-seconds 14 --prefetch-steps 30 --camera-hz 10 \
  --seed 20261102 --ros-port 8780 \
  --output "$A2C2_SERIES/development-a2c2" \
  2>&1 | tee "$A2C2_SERIES/development-a2c2-console.log"

"$A2C2_PYTHON" -m experiments.realtime_vla.evaluate_a2c2 \
  --variant a2c2 --decode-mode direct --protocol-version 3 --initial-states 1 2 3 4 5 \
  --control-hz 20 --wall-seconds 14 --prefetch-steps 30 --camera-hz 10 \
  --seed 20261102 --ros-port 8780 \
  --output "$A2C2_SERIES/formal-a2c2" \
  2>&1 | tee "$A2C2_SERIES/formal-a2c2-console.log"

"$A2C2_PYTHON" -m experiments.realtime_vla.summarize_a2c2 \
  --series "$A2C2_SERIES" --bridge "$A2C2_BRIDGE"
