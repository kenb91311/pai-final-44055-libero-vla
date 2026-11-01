#!/usr/bin/env bash
# Preserve both earlier runtimes; author-specific inference uses a separate overlay.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
if [[ ! -x .tools/uv || ! -x .venv-realtime/bin/python ]]; then
    # The published task setup installs its full140pin runtime independently.
    bash scripts/setup_realtime_vla.sh
fi
if [[ ! -x .venv-vlash/bin/python ]]; then
    .tools/uv venv --python .venv-realtime/bin/python .venv-vlash
fi
# Discover historical optional torch fallback without requiring it on a fresh task install.
.venv-realtime/bin/python experiments/realtime_vla/prepare_vlash_overlay.py --overlay .venv-vlash
.tools/uv pip install --python .venv-vlash/bin/python --no-deps -r requirements-vlash.lock
.venv-vlash/bin/python - <<'PY'
"""Expose the exact inherited LIBERO package at the isolated prefix's expected path.

Args: None. Returns: One overlay-only relative package alias. Raises: Version/path mismatch.
Design: Context: FrozenLiberoEnv deliberately derives its asset path from sys.prefix.
    Concern: A .pth fallback supplies imports but does not create that physical package directory.
    Decision: Resolve the installed hf-libero package and create only a local relative symlink.
    Alternatives: Edit shared simulator code or copy a different package version.
    Quality Attributes: Isolation, portability. Trade-offs: Shared read-only package files remain required.
Evidence: Internal: First VLASH d0 stopped before any task tick because the prefix path was absent.
    External: importlib package search locations. Findings: Existing source/assets links stay unchanged.
    Metrics: Exact version and resolved path. Confidence: Filesystem identity.
Constraints: No shared package/source/asset mutation. Reconsider When: Package layout changes.
References: vlash-protocol-amendment-03.json. ADR: None.
"""
from pathlib import Path
from importlib.util import find_spec
from importlib.metadata import version
import os,sys
assert version('hf-libero')=='0.1.4'
spec=find_spec('libero')
assert spec is not None and spec.submodule_search_locations
source=Path(next(iter(spec.submodule_search_locations))).resolve()
assert (source/'libero').is_dir()
target=Path(sys.prefix)/'lib/python3.12/site-packages/libero'
if target.is_symlink() or target.exists():
    assert target.resolve()==source
else:
    target.symlink_to(os.path.relpath(source,target.parent),target_is_directory=True)
assert target.resolve()==source
assets=Path('.cache/libero-assets').resolve()
assert (target/'libero/assets').resolve()==assets
print({'hf-libero':version('hf-libero'),'overlay_package_alias_verified':True,'shared_assets_target_verified':True})
PY
.venv-vlash/bin/python experiments/realtime_vla/prepare_vlash_source.py
.venv-vlash/bin/python - <<'PY'
"""Smoke-import the original author class without creating policy tensors.

Args: None. Returns: Version JSON. Raises: Import/version errors.
Design: Context: VLASH uses patched AdaRMS APIs. Concern: Contemporary Transformers lacks those APIs.
    Decision: Verify exact old backend and import original class. Alternatives: Rewrite model math.
    Quality Attributes: Fidelity. Trade-offs: Separate overlay. Evidence: Internal: Fixed source manifest.
    External: Author pyproject. Findings: No GPU allocation. Metrics: Package versions. Confidence: Import only.
Constraints: Does not prove task performance. Reconsider When: The backend changes.
References: vlash_runtime.py. ADR: None.
"""
from pathlib import Path
from importlib.metadata import version
import json,sys
sys.path.insert(0,str(Path('.cache/vlash-source/d2da223e2251a4a5020b667e77cf8ad57e594144').resolve()))
from vlash.policies.pi05.modeling_pi05 import PI05Policy
print(json.dumps({k:version(k) for k in ['lerobot','transformers','huggingface-hub','tokenizers','torch','numpy']},indent=2))
PY
