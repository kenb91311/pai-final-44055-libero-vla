"""Fetch and verify the exact author VLASH source without installing its hardware extras.

Args: --output: optional source-cache directory; --verify-only: prohibit downloads.
Returns: Source path and a content-integrity JSON summary on stdout.
Raises: HTTP, file, manifest or SHA256 errors.
Design:
    Context: The released VLASH policy needs the author's patched Transformers backend.
    Concern: Substituting the contemporary LeRobot implementation changes its architecture.
    Decision: Fetch an explicit allowlist at one commit, preserving Apache2 notices.
    Alternatives: Import the latest main branch or modify the previous experiment runtime.
    Quality Attributes: Reproducibility, isolation. Trade-offs: Separate inference overlay.
Evidence:
    Internal: docs/reference/vlash-source-manifest.json stores35 original file hashes.
    External: mit-han-lab/vlash sim/libero commit d2da223e2251a4a5020b667e77cf8ad57e594144.
    Findings: Original runtime uses LeRobot0.4.1 and a fixed Transformers Git revision.
    Metrics: Every source byte is compared. Confidence: Exact source identity only.
Constraints: No weights, training, GPU initialization or installed-source edits.
Reconsider When: Author publishes a new checkpoint/backend pairing.
References: docs/reference/realtime-continuity-methods.md. ADR: None.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
SOURCE_REVISION='d2da223e2251a4a5020b667e77cf8ad57e594144'
MANIFEST=ROOT/'docs/reference/vlash-source-manifest.json'


def prepare_source(output=None,verify_only=False):
    """Materialize the pinned allowlist and fail closed on any content mismatch.

    Args: output: owned directory; verify_only: do not contact the network.
    Returns: Verification dictionary. Raises: IO, HTTP or integrity errors.
    Design: Context: Small public author source. Concern: Silent branch drift.
        Decision: Fixed revision and per-file SHA, atomic temporary files.
        Alternatives: Unpinned pip install. Quality Attributes: Fidelity. Trade-offs:35small requests.
    Evidence: Internal: Committed manifest. External: Official author repository.
        Findings: Cached original files are sufficient for simulation inference.
        Metrics: File count and total bytes. Confidence: Content verified.
    Constraints: No path traversal or installed package modification.
    Reconsider When: The source manifest changes.
    References: MANIFEST. ADR: None.
    """
    manifest=json.loads(MANIFEST.read_text())
    if manifest['revision']!=SOURCE_REVISION:raise ValueError('VLASH source revision mismatch')
    output=Path(output or ROOT/'.cache/vlash-source'/SOURCE_REVISION).resolve()
    verified=[]
    for record in manifest['files']:
        dst=output/record['path']
        if not dst.resolve().is_relative_to(output):raise ValueError('invalid source path')
        if not dst.is_file():
            if verify_only:raise FileNotFoundError(record['path'])
            dst.parent.mkdir(parents=True,exist_ok=True)
            url=f'https://raw.githubusercontent.com/mit-han-lab/vlash/{SOURCE_REVISION}/{record["path"]}'
            with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise ValueError(f'author source integrity mismatch: {record["path"]}')
            part=dst.with_suffix(dst.suffix+'.partial');part.write_bytes(data);part.replace(dst)
        if dst.stat().st_size!=record['bytes'] or hashlib.sha256(dst.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'cached author source integrity mismatch: {record["path"]}')
        verified.append(record)
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'repository':manifest['repository'],'revision':SOURCE_REVISION,
            'source_directory':str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else output.name,
            'files_verified':len(verified),'source_bytes':sum(r['bytes'] for r in verified),
            'license':'Apache-2.0','model_weights_downloaded':False}


def main():
    """Expose explicit source preparation with import-safe CLI behavior.

    Args: CLI flags. Returns: None. Raises: Preparation errors.
    Design: Context: Reproduction setup. Concern: Import-time network traffic.
        Decision: Main guard. Alternatives: Implicit fetch. Quality Attributes: Predictability. Trade-offs: None.
    Evidence: Internal: --help exits before preparation. External: argparse convention.
        Findings: No model imports. Metrics: None. Confidence: Interface contract.
    Constraints: Download only on explicit invocation. Reconsider When: CLI changes.
    References: prepare_source. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();print(json.dumps(prepare_source(args.output,args.verify_only),indent=2))


if __name__=='__main__':main()
