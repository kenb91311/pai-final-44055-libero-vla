#!/usr/bin/env bash
# Run the preregistered H50/R30 comparison with one frozen family at a time.
# Inputs: model assets, a ready ROS bridge on8780, and an exclusive GPU lease.
# Outputs: fresh development-init0 and formal-init1..5 directories for each method.
# Design: Preserve the original35 trials; stop on errors; never overwrite a run.
# Evidence: artifacts/realtime-vla/continuity-protocol.json fixes the conditions.
# Limits: Small fixed-state experiment, no training or hard real-time guarantee.
set -euo pipefail
family="${1:?Usage: run_continuity_series.sh smolvla|pi05 [OUTPUT_ROOT]}"
case "$family" in smolvla|pi05) ;; *) printf 'Unsupported family: %s\n' "$family" >&2; exit 2 ;; esac
series_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$series_dir/../.." && pwd)"
output_root="${2:-artifacts/realtime-vla}"
python_bin="${REALTIME_PYTHON:-$project_dir/.venv-realtime/bin/python}"
cd "$project_dir"
for variant in off rtc-stock rtc-vjp; do
    "$python_bin" "$series_dir/evaluate_continuity.py" \
        --model "$family" --variant "$variant" \
        --output "$output_root/development-continuity-$family-$variant" \
        --initial-states 0 --seed 20261102 --control-hz 20 --wall-seconds 14 \
        --prefetch-steps 30 --execution-horizon 30 --camera-hz 10 --ros-port 8780 --video
    "$python_bin" "$series_dir/evaluate_continuity.py" \
        --model "$family" --variant "$variant" \
        --output "$output_root/formal-continuity-$family-$variant" \
        --initial-states 1 2 3 4 5 --seed 20261102 --control-hz 20 --wall-seconds 14 \
        --prefetch-steps 30 --execution-horizon 30 --camera-hz 10 --ros-port 8780 --video
done
