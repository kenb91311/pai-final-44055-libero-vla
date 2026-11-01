#!/usr/bin/env bash
# Capture once, then profile frozen models serially under a coordinated GPU lease.
# Inputs: fresh output path; shared peer status explicitly released; cached models/assets.
# Outputs: shared raw observations plus separate native B-I-B model reports and logs.
# Design: Processes exit between models; the simulator exits before any timed inference.
# Constraints: This coordination does not reserve hardware or terminate other experiments.
set -euo pipefail

PROFILE_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROFILE_PROJECT_ROOT"
PROFILE_OUTPUT="${1:-artifacts/realtime-vla/inference-profile-v1}"
PROFILE_PYTHON="$PROFILE_PROJECT_ROOT/.venv-realtime/bin/python"
PROFILE_PHASE_FILE="$PROFILE_PROJECT_ROOT/.cache/gpu-coordination/realtime-status.json"
PROFILE_STOP_FILE="$PROFILE_PROJECT_ROOT/.cache/gpu-coordination/stop-inference-profile-monitor-v1"
PROFILE_PEER_STATUS_FILE="${PROFILE_PEER_STATUS_FILE:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HOME="$PROFILE_PROJECT_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$PROFILE_PROJECT_ROOT/.cache"
export MPLCONFIGDIR="$PROFILE_PROJECT_ROOT/.cache/matplotlib"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

profile_phase() {
    # Atomically record owned phase; the monitor never infers ownership from memory alone.
    python3 - "$PROFILE_PHASE_FILE" "$1" "$2" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json, sys
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
data = json.loads(path.read_text()) if path.exists() else {}
data.update(state=sys.argv[2], using_gpu=sys.argv[3]=='true',
            gpu_workers_running=sys.argv[3]=='true', utc=datetime.now(timezone.utc).isoformat(),
            estimated_gpu_seconds_after_start=900 if sys.argv[3]=='true' else 0)
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(data, indent=2)+'\n')
temporary.replace(path)
PY
}

profile_exit() {
    # No CUDA child is backgrounded here; all model processes have exited on return.
    if [[ "$1" == 0 ]]; then
        profile_phase inference_stage_profile_complete_gpu_released false
    else
        profile_phase inference_stage_profile_failed_gpu_process_exited false
    fi
    touch -- "$PROFILE_STOP_FILE"
}

python3 - "$PROFILE_OUTPUT" "$PROFILE_PEER_STATUS_FILE" <<'PY'
from pathlib import Path
import json, sys
if sys.argv[2]:
    peer = json.loads(Path(sys.argv[2]).read_text())
    if peer.get('using_gpu') is not False:
        raise SystemExit('Peer GPU work has not explicitly released; do not start this series.')
Path(sys.argv[1]).mkdir(parents=True, exist_ok=False)
PY
trap 'profile_exit $?' EXIT
cp -- artifacts/realtime-vla/inference-profile-protocol.json "$PROFILE_OUTPUT/protocol.json"
profile_phase inference_stage_profile_capture true
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PROFILE_PYTHON" -m experiments.realtime_vla.capture_profile_inputs \
    --output "$PROFILE_OUTPUT/observations" --initial-states 1 2 3 4 5 \
    --seed 20261102 --resolution 256 \
    2>&1 | tee "$PROFILE_OUTPUT/capture-console.log"

for PROFILE_MODEL in smolvla pi05 pi0fast; do
    profile_phase "inference_stage_profile_${PROFILE_MODEL}_native" true
    "$PROFILE_PYTHON" -m experiments.realtime_vla.profile_inference \
        --model "$PROFILE_MODEL" --observations "$PROFILE_OUTPUT/observations" \
        --output "$PROFILE_OUTPUT/$PROFILE_MODEL-native" \
        --warmup 10 --repeats 5 --seed 20261102 --threads 4 --interop-threads 1 \
        2>&1 | tee "$PROFILE_OUTPUT/$PROFILE_MODEL-console.log"
done
