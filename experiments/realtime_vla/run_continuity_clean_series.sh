#!/usr/bin/env bash
# Run the amended, host-coordinated continuity series using one process per method.
# Inputs: verified global GPU lease, continuous host GPU monitoring, ROS bridge8780.
# Outputs: new MODEL-METHOD folders under a fresh clean-series root.
# Design: Fixed0..5 order;0 is preflight only and never enters formal1..5 metrics.
# Limits: This script does not establish global exclusivity itself or stop other processes.
# Required: Read continuity-clean-protocol.json and start only after the parent grants the lease.
set -euo pipefail
family="${1:?Usage: run_continuity_clean_series.sh smolvla|pi05 [OUTPUT_ROOT]}"
case "$family" in smolvla|pi05) ;; *) printf 'Unsupported family: %s\n' "$family" >&2; exit 2 ;; esac
series_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$series_dir/../.." && pwd)"
output_root="${2:-artifacts/realtime-vla/continuity-clean}"
python_bin="${REALTIME_PYTHON:-$project_dir/.venv-realtime/bin/python}"
cd "$project_dir"
for variant in off rtc-stock rtc-vjp; do
    "$python_bin" "$series_dir/evaluate_continuity.py" \
        --model "$family" --variant "$variant" \
        --output "$output_root/$family-$variant" \
        --initial-states 0 1 2 3 4 5 --seed 20261102 --control-hz 20 --wall-seconds 14 \
        --prefetch-steps 30 --execution-horizon 30 --camera-hz 10 --ros-port 8780 --video
done
