"""Download a pinned author SmolVLA/base and A2C2 residual candidate pair.

Args: --output: owned model asset directory.
Returns: Public fixed-revision checkpoints, VLM metadata, upstream sources and provenance.
Raises: Network, license declaration, revision or content integrity errors.
Design:
    Context: A2C2 needs a trained head and matching base; no new training is allowed.
    Concern: The previous official SmolVLA expert was not the author's base checkpoint.
    Decision: Pin author's spatial scratch base and September residual MSE checkpoint.
    Alternatives: An arbitrary feedback controller is not A2C2.
    Quality Attributes: Traceability, isolation. Trade-offs: Exact pair linkage is inferred.
Evidence:
    Internal: Author eval names smolvla_libero_spatial_scratch; head uses hidden960.
    External: github.com/k1000dai/a2c2-libero, arxiv.org/abs/2509.23224.
    Findings: Both model cards declare Apache-2.0; head omits a base revision.
    Metrics: Expected weight SHA256 and bytes. Confidence: Identity, not task success.
Constraints: No training, remote code execution, package changes or credentials.
Reconsider When: Author publishes an explicit pair or changes the released schema.
References: docs/reference/realtime-continuity-methods.md. ADR: None.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

if __package__:
    from .download_smolvla import VLM, VLM_REV, download_repo
else:
    from download_smolvla import VLM, VLM_REV, download_repo

BASE_REPO = 'k1000dai/smolvla_libero_spatial_scratch'
BASE_REV = 'caa0efcb24e261574c824366526c5775d3664cac'
BASE_WEIGHT_SHA = '45f3b6fc1b8ae0b7cf3ab8ebd22336ab23eb3798a8bfef027f5d45596c45a9be'
BASE_WEIGHT_BYTES = 906713328
HEAD_REPO = 'k1000dai/rt_libero_spatial_mse'
HEAD_REV = 'beaf66dd0fff7b18123ae9dbfe6906235d251981'
HEAD_WEIGHT_SHA = '39f174341754164fd8dc8cbdff8f87916763fba65ce93d42c0b22e9bc045b1c0'
HEAD_WEIGHT_BYTES = 123513628
UPSTREAM_REPO = 'k1000dai/a2c2-libero'
UPSTREAM_REV = '75fa9d202fb74150bdbc54ab182c2773f3ed5600'
DIRECT_DECODE_UPSTREAM_REV = 'f7b17e25871ecdf20cbd5a56bcffad31d3a7289a'
UPSTREAM_FILES = (
    'LICENSE', 'README.md', 'eval_libero/evaluation_libero.py',
    'eval_libero/create_dataset_for_residualpolicy.py',
    'src/lerobot/policies/residual_transformer/modeling_residual_transformer.py',
    'src/lerobot/policies/residual_transformer/configuration_residual_transformer.py',
    'src/lerobot/policies/smolvla/modeling_smolvla.py',
    'src/lerobot/policies/normalize.py',
)


def download(output):
    """Fetch only pinned public artifacts and verify both advertised weight hashes.

    Args: output: directory used only by this candidate reproduction.
    Returns: Full provenance mapping. Raises: ValueError for integrity/license failures.
    Design: Context: Legacy checkpoint needs normalization and exact upstream reference.
        Concern: A valid shape does not establish model correspondence.
        Decision: Preserve complete author configs and state the unresolved pair guarantee.
        Alternatives: Relabeling it as the existing official expert would be misleading.
        Quality Attributes: Auditability. Trade-offs: About 1.03 GB weight storage.
    Evidence: Internal: Fixed constants from public HF LFS metadata.
        External: Author model cards and evaluation script.
        Findings: Local inference uses no optimizer. Metrics: SHA/bytes. Confidence: High identity.
    Constraints: No model execution. Reconsider When: Checkpoint pairing is documented.
    References: Module constants. ADR: None.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = download_repo(BASE_REPO, BASE_REV, output/'base')
    head = download_repo(HEAD_REPO, HEAD_REV, output/'head')
    for record, expected_sha, expected_bytes in (
        (base, BASE_WEIGHT_SHA, BASE_WEIGHT_BYTES),
        (head, HEAD_WEIGHT_SHA, HEAD_WEIGHT_BYTES),
    ):
        weight = next(x for x in record['files'] if x['path'] == 'model.safetensors')
        if weight['sha256'] != expected_sha or weight['bytes'] != expected_bytes:
            raise ValueError('pinned weight identity mismatch')
        if record['license'] != 'apache-2.0':
            raise ValueError('expected Apache-2.0 model-card declaration')
    vlm = download_repo(VLM, VLM_REV, output/'base'/'vlm', metadata_only=True)
    sources = []
    for name in UPSTREAM_FILES:
        target = output/'upstream'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_REV}/{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        target.write_bytes(raw)
        sources.append({'file':name, 'url':url, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
    direct_name = 'src/lerobot/policies/residual_transformer/modeling_residual_transformer.py'
    direct_url = f'https://raw.githubusercontent.com/{UPSTREAM_REPO}/{DIRECT_DECODE_UPSTREAM_REV}/{direct_name}'
    with urllib.request.urlopen(direct_url, timeout=60) as response:
        direct_raw = response.read()
    direct_target = output/'upstream-direct'/direct_name
    direct_target.parent.mkdir(parents=True, exist_ok=True)
    direct_target.write_bytes(direct_raw)
    result = {
        'method':'A2C2 candidate reproduction', 'base':base, 'head':head,
        'vlm_metadata':vlm, 'upstream_repository':UPSTREAM_REPO,
        'upstream_revision':UPSTREAM_REV, 'upstream_sources':sources,
        'direct_decode_reference':{'revision':DIRECT_DECODE_UPSTREAM_REV, 'url':direct_url,
                                   'sha256':hashlib.sha256(direct_raw).hexdigest(), 'bytes':len(direct_raw)},
        'pair_guarantee':'Candidate only: author eval basename matches base; head metadata omits base SHA.',
        'is_previous_official_smolvla_expert':False, 'local_training':False,
        'credentials_used':False,
    }
    (output/'provenance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status':'complete', 'base_revision':BASE_REV, 'head_revision':HEAD_REV}), flush=True)
    return result


def main():
    """Run download explicitly; importing the module or requesting help transfers nothing.

    Args: CLI --output. Returns: None. Raises: Download errors.
    Design: Context: Reproducible asset preparation. Concern: Import side effects.
        Decision: argparse plus main guard. Alternatives: Implicit download.
        Quality Attributes: Predictability. Trade-offs: None.
    Evidence: Internal: No module-level network calls. External: argparse.
        Findings: Help works offline. Metrics: None. Confidence: High.
    Constraints: No inference. Reconsider When: CLI contract changes.
    References: download. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, default=Path('artifacts/realtime-vla/models/a2c2'))
    args = parser.parse_args()
    download(args.output)


if __name__ == '__main__':
    main()
