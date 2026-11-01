"""Fetch the author's fixed asynchronous LIBERO pi0.5 assets for VLASH testing.

Args: --output: local model directory. Returns: Verified assets and provenance JSON.
Raises: HTTP, revision, size or digest errors.
Design:
    Context: VLASH needs its dedicated, previously trained checkpoint.
    Concern: Ordinary pi0.5 input replacement is not a VLASH reproduction.
    Decision: Pin author weights and official branch; keep weight terms distinct from code.
    Alternatives: Retrain locally or claim the ordinary model supports VLASH.
    Quality Attributes: Fidelity, recoverability. Trade-offs: 7.47GB transfer.
Evidence:
    Internal: Fixed HF LFS SHA and size. External: mit-han-lab/vlash issue10.
    Findings: Code is Apache2; the weight card does not declare a license.
    Metrics: File sizes and SHA256. Confidence: Asset identity, not task performance.
Constraints: No inference, training, redistribution of weights or private credentials.
Reconsider When: Author changes checkpoint or publishes explicit weight terms.
References: docs/reference/realtime-continuity-methods.md. ADR: None.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from download_pi05 import file_digest

REPO='mit-han-lab/vlash-pi05-libero-async5'
REVISION='c85d5962a7b823f9f391f97d60227d6ca8bd7bc7'
WEIGHT_SHA='8dd3bf4766dac481933ced7c8de5fbfc31c486e665cca33e6ddf970792a46a2e'
WEIGHT_BYTES=7472953952


def download(output):
    """Resume fixed files and admit weights only after content verification.

    Args: output: owned path. Returns: Provenance dict. Raises: IO/HTTP/integrity errors.
    Design: Context: Large public checkpoint. Concern: Interrupted or substituted data.
        Decision: API revision check, partial files, LFS hash then atomic rename.
        Alternatives: Trust file presence. Quality Attributes: Integrity. Trade-offs: Hash IO.
    Evidence: Internal: Published LFS digest. External: Fixed HF repository API.
        Findings: Model card absent; license is not inferred from the code license.
        Metrics: SHA/bytes. Confidence: Exact identity.
    Constraints: No model execution. Reconsider When: Asset format changes.
    References: REPO/REVISION. ADR: None.
    """
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=60) as r:api=json.load(r)
    assert api['sha']==REVISION
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true',timeout=60) as r:listing=json.load(r)
    (output/'model-api.json').write_text(json.dumps(api,indent=2)+'\n')
    records=[]
    for item in listing:
        name=item.get('path','')
        if item.get('type')!='file' or name=='.gitattributes':continue
        dst=output/name
        if not dst.resolve().is_relative_to(output):raise ValueError('invalid upstream path')
        dst.parent.mkdir(parents=True,exist_ok=True)
        size=int(item['size']);sha=item.get('lfs',{}).get('oid')
        if name=='model.safetensors':assert (size,sha)==(WEIGHT_BYTES,WEIGHT_SHA)
        if not dst.is_file():
            part=dst.with_suffix(dst.suffix+'.partial');offset=part.stat().st_size if part.exists() else 0
            req=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req,timeout=180) as r:
                if offset and r.status!=206:offset=0
                count=offset;last=time.monotonic()
                with part.open('ab' if offset else 'wb') as sink:
                    while block:=r.read(8*1024*1024):
                        sink.write(block);count+=len(block)
                        if time.monotonic()-last>20:print(json.dumps({'file':name,'bytes':count,'total':size}),flush=True);last=time.monotonic()
            if part.stat().st_size!=size or (sha and file_digest(part)!=sha):raise ValueError(f'incomplete or invalid download: {name}')
            part.replace(dst)
        actual=file_digest(dst)
        if dst.stat().st_size!=size or (sha and actual!=sha):raise ValueError(f'asset integrity mismatch: {name}')
        records.append({'file':name,'bytes':size,'sha256':actual,'expected_lfs_sha256':sha})
        print(json.dumps({'file':name,'status':'verified','bytes':size}),flush=True)
    provenance={'repository':REPO,'revision':REVISION,'files':records,'code_repository':'mit-han-lab/vlash','code_branch':'sim/libero','code_revision':'d2da223e2251a4a5020b667e77cf8ad57e594144','code_license':'Apache-2.0','weight_license':'not declared in model card/config; do not label Apache-2.0','local_training':False,'task_success_measured':False}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def main():
    """Download explicitly; imports and --help have no network side effects.

    Args: CLI. Returns: None. Raises: Download errors.
    Design: Context: Reproduction CLI. Concern: Accidental transfer.
        Decision: Main guard. Alternatives: Import side effects.
        Quality Attributes: Predictability. Trade-offs: None.
    Evidence: Internal: Argparse precedes transfer. External: Python CLI convention.
        Findings: No GPU calls. Metrics: None. Confidence: Contract.
    Constraints: Owned destination only. Reconsider When: CLI changes.
    References: download. ADR: None.
    """
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--output',type=Path,default=Path('artifacts/realtime-vla/models/vlash'));a=p.parse_args();download(a.output)


if __name__=='__main__':main()
