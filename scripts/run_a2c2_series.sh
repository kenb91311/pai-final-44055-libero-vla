#!/usr/bin/env bash
# Run one predeclared author-base A2C2 series under an externally scheduled GPU lease.
# Each development episode may fail the TASK and still proceed; runtime/ROS errors stop.
set -euo pipefail

A2C2_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$A2C2_PROJECT_ROOT"
A2C2_PYTHON="$A2C2_PROJECT_ROOT/.venv-realtime/bin/python"
A2C2_SERIES="${1:-artifacts/realtime-vla/a2c2-author-series-v2}"
A2C2_BRIDGE="${2:-artifacts/realtime-vla/pi05-ros-system}"

"$A2C2_PYTHON" -m experiments.realtime_vla.summarize_a2c2 \
  --series "$A2C2_SERIES" --bridge "$A2C2_BRIDGE" --register-only

for A2C2_VARIANT in off a2c2; do
  "$A2C2_PYTHON" -m experiments.realtime_vla.evaluate_a2c2 \
    --variant "$A2C2_VARIANT" --initial-states 0 \
    --protocol-version 2 --decode-mode residual \
    --control-hz 20 --wall-seconds 14 --prefetch-steps 30 --camera-hz 10 \
    --seed 20261102 --ros-port 8780 \
    --output "$A2C2_SERIES/development-$A2C2_VARIANT" \
    2>&1 | tee "$A2C2_SERIES/development-$A2C2_VARIANT-console.log"

  "$A2C2_PYTHON" -m experiments.realtime_vla.evaluate_a2c2 \
    --variant "$A2C2_VARIANT" --initial-states 1 2 3 4 5 \
    --protocol-version 2 --decode-mode residual \
    --control-hz 20 --wall-seconds 14 --prefetch-steps 30 --camera-hz 10 \
    --seed 20261102 --ros-port 8780 \
    --output "$A2C2_SERIES/formal-$A2C2_VARIANT" \
    2>&1 | tee "$A2C2_SERIES/formal-$A2C2_VARIANT-console.log"
done

"$A2C2_PYTHON" -m experiments.realtime_vla.summarize_a2c2 \
  --series "$A2C2_SERIES" --bridge "$A2C2_BRIDGE"
