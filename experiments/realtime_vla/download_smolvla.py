"""Fetch a fixed LIBERO-ready SmolVLA policy and only required VLM metadata.

Args: CLI --output model directory.
Returns: Official policy files, tokenizer/config metadata and SHA256 provenance.
Raises: Network/IO errors; ValueError on content digest mismatch.
Design:
    Context: The experiment compares released task experts without additional training.
    Concern: A generic base model or drifting revision would invalidate the comparison.
    Decision: Pin both the LIBERO expert and its processor repository; verify all LFS hashes.
    Alternatives: Downloading generic smolvla_base does not establish LIBERO competence.
    Quality Attributes: Reproducibility, integrity. Trade-offs: Requires about1.22GB download.
Evidence:
    Internal: Model config defines RGB2/state8/action7/chunk50.
    External: Official HuggingFaceVLA/smolvla_libero model API.
    Findings: Full policy weights include its VLM; original VLM weights need not be fetched twice.
    Metrics: SHA256 and byte count. Confidence: Content identity, not task performance.
Constraints: No model training, no untrusted remote code execution.
Reconsider When: Model or tokenizer revision changes.
References: https://huggingface.co/HuggingFaceVLA/smolvla_libero .
ADR: None.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

POLICY='HuggingFaceVLA/smolvla_libero'
POLICY_REV='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
VLM='HuggingFaceTB/SmolVLM2-500M-Instruct'
VLM_REV='7b375e1b73b11138ff12fe22c8f2822d8fe03467'


def download_repo(repo,revision,output,metadata_only=False):
    """Download an allowlisted pinned repository and verify advertised file identities.

    Args: repo/revision: fixed Hub identity; output: directory; metadata_only: omit weights/ONNX.
    Returns: Per-file provenance records.
    Raises: ValueError on byte or digest mismatch; network errors propagate.
    Design:
        Context: Tokenizers and normalizers are part of policy compatibility.
        Concern: Main-branch drift or truncated transfers could silently change inference.
        Decision: Resolve fixed revision, retain API metadata and stream to temporary files.
        Alternatives: Blind snapshot of VLM also duplicates2GB of unused base weights.
        Quality Attributes: Integrity. Trade-offs: Git text blobs receive computed SHA only.
    Evidence:
        Internal: Hub blobs expose LFS SHA256 and exact sizes.
        External: Official Hugging Face repositories.
        Findings: Policy safe tensors and normalizer safe tensors need exact hashes.
        Metrics: SHA256/size. Confidence: High.
    Constraints: Paths must remain inside output; no repository code is executed.
    Reconsider When: Repository layout changes.
    References: Module constants.
    ADR: None.
    """
    output.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
    data=json.load(urllib.request.urlopen(url,timeout=60))
    if data['sha']!=revision:raise ValueError('unexpected resolved revision')
    (output/'hub-api.json').write_text(json.dumps(data,indent=2)+'\n')
    records=[]
    for item in data['siblings']:
        name=item['rfilename']
        if name.startswith('.') or '/' in name:continue
        if metadata_only and name.endswith(('.safetensors','.bin','.onnx')):continue
        path=output/name
        expected=item.get('lfs',{}).get('sha256')
        valid=False
        if path.exists():
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            valid=path.stat().st_size==item['size'] and (not expected or digest==expected)
        if not valid:
            temp=path.with_suffix(path.suffix+'.partial')
            source=f'https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}'
            started=time.monotonic();last=started;count=0
            with urllib.request.urlopen(source,timeout=180) as response,temp.open('wb') as stream:
                while chunk:=response.read(8*1024*1024):
                    stream.write(chunk);count+=len(chunk)
                    if time.monotonic()-last>15:
                        print(json.dumps({'repo':repo,'file':name,'bytes':count,'expected':item['size']}),flush=True);last=time.monotonic()
            temp.replace(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size!=item['size'] or expected and digest!=expected:raise ValueError(f'file mismatch {name}')
        records.append({'path':name,'bytes':path.stat().st_size,'sha256':digest,'advertised_lfs_sha256':expected})
    return {'repository':repo,'revision':revision,'license':data.get('cardData',{}).get('license'),'files':records}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--output',type=Path,default=Path('artifacts/realtime-vla/models/smolvla'));args=parser.parse_args()
    policy=download_repo(POLICY,POLICY_REV,args.output)
    vlm=download_repo(VLM,VLM_REV,args.output/'vlm',metadata_only=True)
    result={'policy':policy,'vlm_metadata':vlm,'local_training':False,'original_vlm_weights_downloaded':False,'note':'load_vlm_weights=False restores full VLM from policy weights; VLM config/tokenizer assets retained separately'}
    (args.output/'provenance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'complete','policy_revision':POLICY_REV,'files':len(policy['files'])+len(vlm['files'])}))
