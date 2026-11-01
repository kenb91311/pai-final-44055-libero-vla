"""Retrieve the fixed LIBERO scene assets with file-level provenance.

Args: None; paths are derived from this script.
Returns: Local assets and a SHA256 manifest.
Raises: Network or integrity errors on incomplete assets.
Design:
    Context: LIBERO scene meshes are external community assets.
    Concern: A retained manifest does not imply files exist after cloning.
    Decision: Check actual files and retrieve the fixed snapshot when incomplete.
    Alternatives: Copying a workspace cache or rebuilding task objects.
    Quality Attributes: Reproducibility. Trade-offs: About422MB initial download.
Evidence:
    Internal: FrozenLiberoEnv uses the same revision/path.
    External: lerobot/libero-assets public snapshot.
    Findings: Asset provenance is separate from code licensing.
    Metrics: Bytes and SHA256. Confidence: Content identity.
Constraints: No model weights or training-data download.
Reconsider When: Asset snapshot or task changes.
References: docs/reference/realtime-vla-assets.md.
ADR: None.
"""
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path(__file__).resolve().parents[1]
output=root/'.cache/libero-assets'
revision='0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'
manifest=root/'artifacts/realtime-vla/libero-assets-manifest.json'
complete=False
if manifest.exists():
    saved=json.loads(manifest.read_text())
    complete=saved.get('revision')==revision and bool(saved.get('files'))
    for record in saved.get('files',[]):
        path=output/record['path']
        if not path.is_file() or path.stat().st_size!=record['bytes']:
            complete=False;break
if not complete:
    snapshot_download('lerobot/libero-assets',repo_type='dataset',revision=revision,local_dir=output)
    records=[]
    for path in sorted(output.rglob('*')):
        if path.is_file() and '.cache' not in path.relative_to(output).parts:
            records.append({'path':str(path.relative_to(output)),'bytes':path.stat().st_size,
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({'repository':'lerobot/libero-assets','revision':revision,'files':records},indent=2)+'\n')
assert (output/'stable_scanned_objects').is_dir()
print(json.dumps({'assets_revision':revision,'status':'ready'}))
