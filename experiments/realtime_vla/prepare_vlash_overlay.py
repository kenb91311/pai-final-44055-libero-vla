"""Plan the author overlay from the standalone task runtime without importing CUDA torch.

Args: --overlay: isolated output environment; --check-only: print the plan without writes.
Returns: Optional .pth fallback plus a path-provenance summary.
Raises: Missing torch, incompatible Python or invalid environment path errors.
Design: Context: Public task runtime installs its complete140pins in .venv-realtime.
    Concern: Historical workspace used another optional torch fallback that public users do not have.
    Decision: Always inherit task site-packages and discover any legacy torch location dynamically.
    Alternatives: Require another project/environment or modify the measured runtime.
    Quality Attributes: Portability, isolation. Trade-offs: Same CPython3.12 ABI is required.
Evidence: Internal: Standalone task bootstrap and historical .pth overlay.
    External: importlib.find_spec and Python site semantics.
    Findings: Top-level torch location can be found without loading tensors or CUDA.
    Metrics: Explicit inherited path count. Confidence: Filesystem/import metadata only.
Constraints: No model import, package install, network or base-environment mutation.
Reconsider When: Task runtime Python ABI or torch packaging changes.
References: scripts/setup_vlash.sh; vlash-reproduction-packaging-note.json. ADR: None.
"""
import argparse
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import sysconfig

ROOT=Path(__file__).resolve().parents[2]


def inherited_sites(base_site,torch_origin):
    """Use task packages first and add only the discovered optional torch provider root.

    Args: base_site: task purelib; torch_origin: top-level torch __init__.py location.
    Returns: Ordered unique existing package roots. Raises: ValueError on unsupported/missing paths.
    Design: Context: Standalone torch is direct; archived workspace torch used a fallback.
        Concern: Hardcoded legacy directories break a fresh task checkout.
        Decision: Infer the real torch package parent and deduplicate it against task purelib.
        Alternatives: Assume .venv exists. Quality Attributes: Portability. Trade-offs: Wheel-style torch layout.
    Evidence: Internal: Both supported path layouts. External: Python package __init__ convention.
        Findings: Standalone produces one fallback root, historical layout two.
        Metrics: One/two roots. Confidence: Path checks, no CUDA calls.
    Constraints: No directory creation. Reconsider When: Torch uses an editable/namespace distribution.
    References: main. ADR: None.
    """
    base_site=Path(base_site).resolve();origin=Path(torch_origin).resolve()
    if not base_site.is_dir() or not origin.is_file() or origin.name!='__init__.py' or origin.parent.name!='torch':
        raise ValueError('task runtime must provide an installed wheel-style torch package')
    provider=origin.parent.parent
    return [base_site]+([] if provider==base_site else [provider])


def main():
    """Write only the selected overlay's fallback file or perform a read-only plan check.

    Args: CLI. Returns: None. Raises: ABI/import/path errors.
    Design: Context: Setup executes this helper with the task runtime's Python.
        Concern: Importing torch for discovery would consume resources during other experiments.
        Decision: Stdlib find_spec and sysconfig only; check-only writes nothing.
        Alternatives: torch.__file__ after a heavy import.
        Quality Attributes: Resource isolation. Trade-offs: Inference checked separately.
    Evidence: Internal: Source-only packaging repair after all VLASH trials.
        External: importlib metadata API. Findings: No measured model/evaluator changes.
        Metrics: Path plan. Confidence: Environment metadata.
    Constraints: CPython3.12, target environment must already exist.
    Reconsider When: Base layout changes. References: inherited_sites. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--overlay',type=Path,default=ROOT/'.venv-vlash');parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,12):raise RuntimeError('CPython3.12 task runtime is required')
    spec=find_spec('torch')
    if spec is None or spec.origin is None:raise RuntimeError('Run the standalone task setup_realtime_vla.sh first')
    base_site=Path(sysconfig.get_path('purelib')).resolve();roots=inherited_sites(base_site,spec.origin)
    relative_site=base_site.relative_to(Path(sys.prefix).resolve());target=args.overlay.resolve()/relative_site
    if not target.is_dir():raise FileNotFoundError('Create the isolated .venv-vlash before writing its fallback')
    if not args.check_only:
        (target/'project_runtime_fallback.pth').write_text(''.join(str(root)+'\n' for root in roots))
    print(json.dumps({'mode':'check-only' if args.check_only else 'written','base_runtime':Path(sys.prefix).name,
                      'inherited_sites':[root.relative_to(ROOT).as_posix() if root.is_relative_to(ROOT) else root.name for root in roots],
                      'legacy_torch_fallback_detected':len(roots)>1,'other_project_required':False,
                      'torch_imported':'torch' in sys.modules},indent=2))


if __name__=='__main__':main()
