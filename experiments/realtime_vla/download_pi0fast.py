"""Download the fixed official LeRobot pi0-FAST LIBERO checkpoint without training.

Args: None; fixed repository, revision and output directory.
Returns: Checkpoint assets and SHA-verified provenance JSON.
Raises: HTTP errors, checksum mismatch or malformed metadata.
Design:
    Context: Compare frozen SmolVLA and pi0-FAST on one LIBERO task family.
    Concern: Base checkpoints and pi0/pi05 substitutes would invalidate the comparison.
    Decision: Pin the actual official pi0_fast task checkpoint and processor states.
    Alternatives: Old openpi JAX checkpoint needs explicit z-score compatibility repair.
    Quality Attributes: Reproducibility, provenance. Trade-offs: 7.73GB download.
Evidence:
    Internal: config declares type pi0_fast, LIBERO action7/chunk10 and MEAN_STD.
    External: https://huggingface.co/lerobot/pi0fast-libero.
    Findings: Current LeRobot supports pi0-FAST PyTorch with KV decoding.
    Metrics: SHA256 identity. Confidence: High for content identity, task success untested.
Constraints: No optimization; weights license inherits upstream Gemma terms, not MIT assertion.
Reconsider When: Comparison checkpoint changes.
References: Official HF repository and LeRobot policy implementation.
ADR: None.
"""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO='lerobot/pi0fast-libero'
REVISION='840f4b503f4c09110421c33c810a85b6684fd658'
OUT=Path('artifacts/realtime-vla/models/pi0fast')
OUT.mkdir(parents=True,exist_ok=True)
BASE=f'https://huggingface.co/{REPO}/resolve/{REVISION}/'
with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true',timeout=60) as response:
    listing=json.load(response)
(OUT/'official-tree.json').write_text(json.dumps(listing,indent=2)+'\n')
with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}',timeout=60) as response:
    api=json.load(response)
(OUT/'model-api.json').write_text(json.dumps(api,indent=2)+'\n')
records=[]
for item in listing:
    name=item.get('path','')
    if item.get('type')!='file' or name=='.gitattributes':continue
    target=OUT/name
    expected_size=int(item['size'])
    expected_hash=item.get('lfs',{}).get('oid')
    if not target.exists():
        temporary=target.with_suffix(target.suffix+'.partial')
        offset=temporary.stat().st_size if temporary.exists() else 0
        request=urllib.request.Request(BASE+name,headers={'Range':f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(request,timeout=180) as response:
            if offset and response.status!=206:offset=0
            mode='ab' if offset else 'wb'
            count=offset;last=time.monotonic()
            with temporary.open(mode) as stream:
                while chunk:=response.read(8*1024*1024):
                    stream.write(chunk);count+=len(chunk)
                    if time.monotonic()-last>15:
                        print(json.dumps({'file':name,'downloaded_bytes':count,'expected_bytes':expected_size}),flush=True);last=time.monotonic()
        temporary.rename(target)
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):digest.update(chunk)
    assert target.stat().st_size==expected_size,(name,target.stat().st_size,expected_size)
    if expected_hash:assert digest.hexdigest()==expected_hash,(name,digest.hexdigest(),expected_hash)
    records.append({'file':name,'bytes':target.stat().st_size,'sha256':digest.hexdigest(),'expected_lfs_sha256':expected_hash})
    print(json.dumps({'file':name,'status':'verified','bytes':target.stat().st_size}),flush=True)
provenance={'repository':REPO,'revision':REVISION,'files':records,'local_training':False,'license_note':'main model card absent; upstream PaliGemma/Gemma terms apply; official v044 sibling card declares gemma','task':'LIBERO pretrained checkpoint, not base pi0-FAST; no pi0/pi05 substitution','validation':'download and identity only; no task success measured by this script'}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'status':'complete','repository':REPO,'revision':REVISION,'bytes':sum(r['bytes'] for r in records)}),flush=True)
