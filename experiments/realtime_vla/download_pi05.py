"""Download one official task-adapted pi0.5 LIBERO checkpoint with exact content checks.

Args: --output: model directory, default artifacts/realtime-vla/models/pi05.
Returns: Fixed-revision model assets, API/tree snapshots and SHA256 provenance.
Raises: Network, incomplete transfer, revision/license mismatch or digest errors.
Design:
    Context: Extend frozen LIBERO comparison without changing prior models or training.
    Concern: Generic pi05_base or random fallback is not an evaluated task checkpoint.
    Decision: Pin official pi05_libero_finetuned and verify published LFS content hashes.
    Alternatives: Download14.47GB generic/base F32 assets or substitute pi0/FAST.
    Quality Attributes: Reproducibility. Trade-offs:7.47GB download, upstream Gemma conditions.
Evidence:
    Internal: Fixed API revision and model.safetensors SHA256.
    External: https://huggingface.co/lerobot/pi05_libero_finetuned.
    Findings: Main and v044 names resolve to the same selected public revision.
    Metrics: All file sizes/digests. Confidence: Identity only; task success measured separately.
Constraints: No training, package changes, model inference or private credentials.
Reconsider When: Selected checkpoint or official model schema changes.
References: docs/reference/pi05-libero-assets.md. ADR: None.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO='lerobot/pi05_libero_finetuned'
REVISION='7152f13e34c6c115d808aea0e5d08aca74eac14b'
WEIGHT_SHA256='877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae'
WEIGHT_BYTES=7473096344


def file_digest(path):
    """Stream a large checkpoint digest without retaining weight bytes in memory.

    Args: path: local file. Returns: SHA256. Raises: IO errors.
    Design: Context: MultiGB assets. Concern: Additional RAM pressure.
        Decision:8MiB streaming. Alternatives: read_bytes.
        Quality Attributes: Bounded memory. Trade-offs: One disk pass.
    Evidence: Internal: HF LFS SHA verified. External: SHA256.
        Findings: Filename/length alone is insufficient.
    Constraints: Read-only. Reconsider When: Content checksum contract changes.
    References: download. ADR: None.
    """
    digest=hashlib.sha256()
    with path.open('rb') as source:
        while block:=source.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()


def download(output):
    """Fetch and verify all fixed official checkpoint files using resumable HTTP.

    Args: output: owned model directory. Returns: Provenance mapping.
    Raises: HTTP, integrity, metadata and filesystem errors.
    Design: Context: Interrupted multiGB downloads are recoverable.
        Concern: An incomplete .safetensors must never reach a policy loader.
        Decision: Partial filename, range resume, digest check before atomic rename.
        Alternatives: Treating a failed transfer as usable weights.
        Quality Attributes: Integrity, recoverability. Trade-offs: Hash passes add IO.
    Evidence: Internal: Explicit expected7,473,096,344bytes and LFS SHA.
        External: Hugging Face fixed revision API.
        Findings: HF card declares Gemma, even if general docs describe Apache code.
    Constraints: No secrets, training or package edits.
    Reconsider When: Official artifact hashes change.
    References: REPO and REVISION constants. ADR: None.
    """
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    api_url=f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(api_url,timeout=60) as response:api=json.load(response)
    assert api['sha']==REVISION and api.get('cardData',{}).get('license')=='gemma'
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true',timeout=60) as response:listing=json.load(response)
    (output/'model-api.json').write_text(json.dumps(api,indent=2)+'\n');(output/'official-tree.json').write_text(json.dumps(listing,indent=2)+'\n')
    records=[]
    for item in listing:
        name=item.get('path','')
        if item.get('type')!='file' or name=='.gitattributes':continue
        target=output/name
        if not target.resolve().is_relative_to(output):raise ValueError('unexpected upstream file path')
        target.parent.mkdir(parents=True,exist_ok=True);expected_size=int(item['size']);expected_hash=item.get('lfs',{}).get('oid')
        if name=='model.safetensors':assert expected_size==WEIGHT_BYTES and expected_hash==WEIGHT_SHA256
        if not target.is_file():
            partial=target.with_suffix(target.suffix+'.partial');offset=partial.stat().st_size if partial.exists() else 0
            request=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=180) as response:
                if offset and response.status!=206:offset=0
                count=offset;last=time.monotonic()
                with partial.open('ab' if offset else 'wb') as sink:
                    while block:=response.read(8*1024*1024):
                        sink.write(block);count+=len(block)
                        if time.monotonic()-last>=15:print(json.dumps({'file':name,'downloaded_bytes':count,'expected_bytes':expected_size}),flush=True);last=time.monotonic()
            if partial.stat().st_size!=expected_size:raise ValueError(f'incomplete download: {name}')
            actual=file_digest(partial)
            if expected_hash and actual!=expected_hash:raise ValueError(f'digest mismatch: {name}')
            partial.replace(target)
        actual=file_digest(target)
        if target.stat().st_size!=expected_size or (expected_hash and actual!=expected_hash):raise ValueError(f'checkpoint integrity mismatch: {name}')
        records.append({'file':name,'bytes':expected_size,'sha256':actual,'expected_lfs_sha256':expected_hash});print(json.dumps({'file':name,'status':'verified','bytes':expected_size}),flush=True)
    provenance={'repository':REPO,'revision':REVISION,'alias_repository':'lerobot/pi05_libero_finetuned_v044','license':'gemma','files':records,'local_training':False,'private_credentials_used':False,'model_type':'pi05','task':'LIBERO task-adapted finetuned checkpoint; not generic pi05_base','benchmark_note':'Official docs report97.5% four-suite average for their evaluation; no local API smoke task-success claim.'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps({'status':'complete','repository':REPO,'revision':REVISION,'bytes':sum(r['bytes'] for r in records)}),flush=True)
    return provenance


def main():
    """Run explicit fixed-checkpoint download; --help has no network side effects.

    Args: CLI --output. Returns: None. Raises: Download errors.
    Design: Context: Fresh standalone reproduction. Concern: Accidental transfer during import/help.
        Decision: Main guard and argparse before network work.
        Alternatives: Module-level download. Quality Attributes: Predictability. Trade-offs: None.
    Evidence: Internal: Runtime imports constants safely. External: argparse.
        Findings: Import does not touch model assets.
    Constraints: No inference. Reconsider When: CLI contract changes.
    References: download. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--output',type=Path,default=Path('artifacts/realtime-vla/models/pi05'));args=parser.parse_args();download(args.output)


if __name__=='__main__':main()
