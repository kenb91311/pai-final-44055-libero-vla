"""Package a fixed allowlist of inference evidence with audited home-path redaction.

Args: --independent-audit JSON; optional --write after all evidence is final.
Returns: Selection JSON or a fresh ZIP and external packaging audit JSON.
Raises: Missing/changing evidence, source snapshot mismatch, secret-pattern or archive checks.
Design: Context: Share actual frozen inference evidence while retaining local originals.
    Concern: Raw reports expose participant home paths; recursive archives include external assets.
    Decision: Fixed member allowlists, textual derivatives, two hashes per member and before/after checks.
    Alternatives: Modify originals or redistribute installed dependency source/model assets.
    Quality Attributes: Privacy, reproducibility. Trade-offs: Distributed report hashes differ after redaction.
Evidence: Internal: Native reports, source snapshot manifest and independent audit.
    External: Python zipfile/SHA256. Findings: FAST failed equality evidence is retained explicitly.
    Metrics: Original/distributed SHA, substitutions and unchanged-original checks. Confidence: Archive verified.
Constraints: No GPU, models, Git, network, originals mutation, weights, tokenizer bodies or caches.
Reconsider When: The experiment evidence schema or publication allowlist changes.
References: docs/experiments/realtime-inference-profiling.md. ADR: None.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path('artifacts/realtime-vla')
PRIMARY = ARTIFACT / 'inference-profile-v1'
SNAPSHOT = ARTIFACT / 'inference-profile-source-snapshot-v1'
TRACE = ARTIFACT / 'inference-profile-resource-trace-v1'
REPORT = ARTIFACT / 'inference-profile-report'
PUBLIC = Path('submissions/pai-final-44055-libero-vla')
MODELS = ('smolvla', 'pi05', 'pi0fast')
REPORT_FILES = ('summary.json', 'plot-data.json', 'totals.csv', 'stages.csv',
                'native-fullchunk.png', 'native-fullchunk.svg', 'stage-host.png', 'stage-host.svg',
                'stage-cuda-stream.png', 'stage-cuda-stream.svg')
FROZEN_SOURCES = (
    'experiments/realtime_vla/profile_inference.py', 'experiments/realtime_vla/profiling_hooks.py',
    'experiments/realtime_vla/capture_profile_inputs.py', 'experiments/realtime_vla/smolvla_runtime.py',
    'experiments/realtime_vla/pi05_runtime.py', 'experiments/realtime_vla/pi0fast_runtime.py',
    'experiments/realtime_vla/monitor_resources.py', 'experiments/realtime_vla/libero_env.py',
    'tests/test_inference_profiling.py', 'tests/test_profile_inputs.py',
    'experiments/realtime_vla/run_inference_profile_series.sh',
)
TEXT_SUFFIXES = {'.json', '.jsonl', '.log', '.py', '.sh', '.md', '.txt', '.csv', '.svg'}
SECRET_PATTERNS = (
    re.compile(rb'https?://[^/\s:@]+:[^/\s@]+@'),
    re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{20,}\b'),
    re.compile(rb'\bhf_[A-Za-z0-9]{20,}\b'),
    re.compile(rb'\bsk-[A-Za-z0-9_-]{24,}\b'),
    re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
)
HOME_PATTERNS = (
    ('linux_home', re.compile(rb'(/home/)([A-Za-z0-9][A-Za-z0-9_.-]*)(?=/|[\s\x22\x27]|$)')),
    ('wsl_windows_home', re.compile(rb'(/mnt/[a-z]/Users/)([A-Za-z0-9][A-Za-z0-9_. -]*)(?=/)', re.I)),
    ('mac_home', re.compile(rb'(/Users/)([A-Za-z0-9][A-Za-z0-9_. -]*)(?=/)')),
    ('windows_home', re.compile(rb'([A-Z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/))([A-Za-z0-9][A-Za-z0-9_. -]*)(?=\\|/)', re.I)),
)


def sha256(data):
    """Identify exact original or distributed bytes.

    Args: data: bytes. Returns: Hex digest. Raises: TypeError for nonbytes input.
    Design: Context: Redacted derivatives need separate identities. Concern: Confusing them with originals.
        Decision: Hash both byte streams. Alternatives: Paths only. Quality Attributes: Integrity. Trade-offs: None.
    Evidence: Internal: Package manifest. External: SHA256. Findings: None. Metrics: Digest. Confidence: High.
    Constraints: No mutation. Reconsider When: Hash policy changes. References: prepare_members. ADR: None.
    """
    return hashlib.sha256(data).hexdigest()


def checked_relative(path):
    """Require a regular, nonsymlinked selected file inside this workspace.

    Args: path: relative or absolute selected file. Returns: Workspace-relative POSIX path.
    Raises: ValueError for escape/symlink/missing file.
    Design: Context: Only explicit local evidence is authorized. Concern: Symlinks pull external assets.
        Decision: Check lexical and resolved membership and reject symlinks. Alternatives: Follow links.
        Quality Attributes: Boundedness. Trade-offs: Symlinked inputs require explicit copying outside this tool.
    Evidence: Internal: Fixed allowlists. External: Path.resolve. Findings: None. Metrics: Membership check.
        Confidence: Filesystem checked.
    Constraints: No directories. Reconsider When: External evidence is explicitly authorized.
    References: select_files. ADR: None.
    """
    candidate = ROOT / path
    if candidate.is_symlink() or not candidate.is_file() or not candidate.resolve().is_relative_to(ROOT):
        raise ValueError('Selected evidence must be an existing regular workspace file')
    return candidate.resolve().relative_to(ROOT).as_posix()


def validate_frozen_sources():
    """Verify all eleven frozen copies, current sources and the monitor source against preregistered hashes.

    Args: None. Returns: Source proof list. Raises: ValueError for missing/changed/unexpected sources.
    Design: Context: Primary run code was frozen before GPU execution.
        Concern: Packaging newer code as if it generated the measurements.
        Decision: Require the exact expected source set and compare snapshot/current/original digests.
        Alternatives: Copy arbitrary current source. Quality Attributes: Provenance. Trade-offs: Source edits block packaging.
    Evidence: Internal: Frozen manifest and model report fingerprints. External: SHA256.
        Findings: Dependency source remains represented by hashes only. Metrics: Eleven source equalities.
        Confidence: Exact bytes.
    Constraints: Read only. Reconsider When: A separate revised run is introduced.
    References: inference-profile-source-snapshot-v1/manifest.json. ADR: None.
    """
    manifest = json.loads((ROOT / SNAPSHOT / 'manifest.json').read_text())
    entries = manifest['files']
    if len(entries) != len(FROZEN_SOURCES) or {entry['path'] for entry in entries} != set(FROZEN_SOURCES):
        raise ValueError('Frozen source manifest does not match the eleven-file allowlist')
    proof = []
    for entry in entries:
        if entry['snapshot'] != Path(entry['path']).name:
            raise ValueError('Unexpected frozen source snapshot member')
        current = checked_relative(entry['path'])
        frozen = checked_relative(SNAPSHOT / entry['snapshot'])
        current_sha = sha256((ROOT / current).read_bytes())
        frozen_sha = sha256((ROOT / frozen).read_bytes())
        if current_sha != entry['sha256'] or frozen_sha != entry['sha256']:
            raise ValueError(f'Frozen source identity check failed: {current}')
        proof.append({'source': current, 'snapshot': frozen, 'preregistered_sha256': entry['sha256'],
                      'current_sha256': current_sha, 'snapshot_sha256': frozen_sha, 'all_equal': True})
    monitor_sha = sha256((ROOT / TRACE / 'monitor-source.py').read_bytes())
    expected = next(entry['sha256'] for entry in entries if entry['path'].endswith('/monitor_resources.py'))
    if monitor_sha != expected:
        raise ValueError('Resource trace monitor copy differs from the frozen monitor source')
    return proof


def select_files(independent_audit):
    """Expand explicit evidence/source/documentation selections without recursive directory copying.

    Args: independent_audit: final independent audit JSON inside workspace.
    Returns: Sorted relative paths and eleven-source proof.
    Raises: Missing/changed file, invalid snapshot or primary-validation errors.
    Design: Context: Public evidence includes a failed FAST paired comparison.
        Concern: Hiding failures or copying unrelated caches and model assets.
        Decision: Preserve all three raw reports/actions and require final report, audit, trace and fixed source/docs.
        Alternatives: Only include successful models or traverse artifacts recursively.
        Quality Attributes: Completeness, boundedness. Trade-offs: New artifacts need an explicit allowlist change.
    Evidence: Internal: Native validation flags and final report file contract. External: None.
        Findings: FAST remains failed-reference evidence. Metrics: Selected paths and validation flags.
        Confidence: Source files checked.
    Constraints: No weights/tokenizer/dependency source. Reconsider When: New experiment series is approved.
    References: prepare_members. ADR: None.
    """
    proof = validate_frozen_sources()
    paths = [ARTIFACT / 'inference-profile-protocol.json', PRIMARY / 'protocol.json',
             PRIMARY / 'observations/manifest.json', PRIMARY / 'capture-console.log',
             SNAPSHOT / 'manifest.json', independent_audit,
             Path('scripts/package_inference_profile.py'), Path('scripts/build_inference_profile_report.py'),
             Path('docs/experiments/realtime-inference-profiling.md'),
             Path('docs/reference/realtime-vla-assets.md'), Path('docs/reference/pi05-libero-assets.md'),
             PUBLIC / 'LICENSES/LeRobot-APACHE-2.0.txt']
    paths += [PRIMARY / f'observations/init{index}.npz' for index in range(1, 6)]
    paths += [PRIMARY / f'{model}-native/{filename}' for model in MODELS for filename in ('report.json', 'actions.npz')]
    paths += [PRIMARY / f'{model}-console.log' for model in MODELS]
    paths += [TRACE / filename for filename in ('configuration.json', 'summary.json', 'samples.jsonl', 'monitor-source.py')]
    paths += [REPORT / filename for filename in REPORT_FILES]
    paths += [Path(entry['snapshot']) for entry in proof]
    for optional in ('tests/test_inference_profile_report.py', 'tests/test_inference_profile_summary.py'):
        if (ROOT / optional).is_file():
            paths.append(Path(optional))
    for model in MODELS:
        report = json.loads((ROOT / PRIMARY / f'{model}-native/report.json').read_text())
        validation = report['validation']
        if not validation['weights_unchanged'] or not validation['all_nested_interval_accounting_valid']:
            raise ValueError(f'Primary frozen/accounting validation failed: {model}')
        if model != 'pi0fast' and not validation['all_paired_outputs_allclose']:
            raise ValueError(f'Primary eligible-model output validation failed: {model}')
    return sorted({checked_relative(path) for path in paths}), proof


def reject_secrets(data, member):
    """Reject credential-like byte patterns while never disclosing their matched content.

    Args: data: selected content; member: safe relative archive name for diagnostics.
    Returns: None. Raises: ValueError naming only the rejected member.
    Design: Context: Reports/source may contain logged credentials.
        Concern: Diagnostic output itself leaking a credential. Decision: Reject without printing matches.
        Alternatives: Log matching lines. Quality Attributes: Confidentiality. Trade-offs: Conservative pattern detection.
    Evidence: Internal: Existing package_realtime_vla.py secret patterns. External: None.
        Findings: Original and derivative text are both checked. Metrics: Pass/fail only. Confidence: Pattern coverage only.
    Constraints: Never return matching text. Reconsider When: Additional credential formats are encountered.
    References: prepare_members. ADR: None.
    """
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        raise ValueError(f'Credential-pattern check rejected archive member: {member}')


def redact_homes(data, aliases):
    """Replace participant home prefixes consistently while preserving distinct path identities.

    Args: data: UTF8 textual bytes; aliases: private in-memory prefix to anonymous-label mapping.
    Returns: Redacted bytes and substitution counts by path family.
    Raises: UnicodeDecodeError for non-UTF8 text.
    Design: Context: Raw reports contain absolute source/cache paths.
        Concern: A single replacement could collapse distinct JSON dictionary keys.
        Decision: Assign stable participant-N labels to distinct home prefixes; never serialize the reverse mapping.
        Alternatives: Strip all source paths or use the same label for every home.
        Quality Attributes: Privacy, provenance. Trade-offs: Participant linkage across one archive remains visible.
    Evidence: Internal: Native reports and trace path fields. External: None.
        Findings: Numeric timings, hashes and validation booleans remain unchanged.
        Metrics: Substitutions per file. Confidence: Textual transformation verified by distributed hashes.
    Constraints: Originals remain untouched; no dependency file contents are copied.
    Reconsider When: Paths use other operating-system home conventions.
    References: prepare_members. ADR: None.
    """
    data.decode('utf8')
    counts = {}
    for family, pattern in HOME_PATTERNS:
        counts[family] = 0

        def replace(match):
            """Assign one private prefix an anonymous label without exposing its original value.

            Args: match: home-prefix match. Returns: Public prefix bytes. Raises: None.
            Design: Context: Redaction callback. Concern: Colliding paths. Decision: Distinct stable aliases.
                Alternatives: One alias. Quality Attributes: Integrity. Trade-offs: In-memory map.
            Evidence: Internal: Redaction self-check. External: None. Findings: No reverse map exported.
                Metrics: Count. Confidence: Direct check.
            Constraints: Text only. Reconsider When: Pattern changes. References: redact_homes. ADR: None.
            """
            original = match.group(0)
            if original not in aliases:
                aliases[original] = b'participant-' + str(len(aliases) + 1).encode()
            counts[family] += 1
            return match.group(1) + aliases[original]

        data = pattern.sub(replace, data)
    return data, counts


def generated_members(test_receipt=None):
    """Build scoped README/license context without asserting a blanket license.

    Args: test_receipt: optional CPU pytest transcript bytes.
    Returns: Generated member-name to bytes mapping.
    Raises: Invalid public license provenance or missing notice errors.
    Design: Context: Installed dependency source and model assets are not redistributed.
        Concern: A broad archive license would misstate external asset terms.
        Decision: Preserve the targeted LeRobot license and derive a narrow notice from existing public provenance.
        Alternatives: Relabel all code/assets under one license. Quality Attributes: Accurate attribution.
        Trade-offs: Consumers still review upstream terms separately.
    Evidence: Internal: Public THIRD_PARTY_NOTICES.md and LICENSES/sources.json.
        External: LeRobot upstream license URL recorded in existing sources.json.
        Findings: Only the applicable license text is included. Metrics: Existing license SHA.
        Confidence: Exact copied license identity.
    Constraints: No external model/tokenizer license is replaced by a code license.
    Reconsider When: Additional third-party code is redistributed.
    References: docs/reference/realtime-vla-assets.md. ADR: None.
    """
    sources = json.loads((ROOT / PUBLIC / 'LICENSES/sources.json').read_text())
    lerobot = next(record for record in sources if record['file'] == 'LeRobot-APACHE-2.0.txt')
    license_path = ROOT / PUBLIC / 'LICENSES/LeRobot-APACHE-2.0.txt'
    if sha256(license_path.read_bytes()) != lerobot['sha256']:
        raise ValueError('Existing public LeRobot license provenance mismatch')
    notice_source = PUBLIC / 'THIRD_PARTY_NOTICES.md'
    notice_sha = sha256((ROOT / notice_source).read_bytes())
    readme = """# Frozen inference profiling evidence

This archive preserves workspace-relative paths and includes the actual five raw
LIBERO observation NPZs, all native action NPZs, protocols, derived reports, resource
trace, frozen experiment source, CPU tests and independent audit.

The SmolVLA/pi05 primary comparison passed its saved checks. The native pi0-FAST
run failed paired output equality/allclose and is retained as excluded reference
evidence. Its failed flags are preserved; do not treat its timing/overhead as an
accepted paired comparison. See docs/experiments/realtime-inference-profiling.md.

Text files are distributed derivatives with participant home prefixes replaced by
participant-N aliases. Local originals were not modified. INFERENCE-PROFILE-MANIFEST.json
records each original and distributed SHA256, substitutions, and unchanged-original
checks. Binary NPZ/PNG members are byte-identical. Source/dependency SHA values inside
raw reports refer to original files; installed dependency sources are not included.

CUDA event values are stream elapsed intervals including possible host launch/idle
gaps, not kernel busy time. Inclusive nested stages must not be summed. The experiment
measures frozen offline inference, not new task success or model training.

External weights, tokenizer bodies, virtual environments, caches, arbitrary dependency
sources and unrelated historical evidence are excluded. Existing per-file notices
apply; no blanket archive/model/data license is asserted. See THIRD_PARTY_NOTICES.md.
"""
    notice = ("# Inference profiling: attribution and scope\n\n"
              "Experiment harness sources retain their existing file-level notices. The targeted\n"
              "LeRobot Apache-2.0 license text and original source URL/SHA are included below\n"
              "submissions/pai-final-44055-libero-vla/LICENSES/. Installed LeRobot/Transformers/\n"
              "PyTorch dependency files are identified by hashes rather than redistributed.\n\n"
              "Model weights, tokenizer bodies and simulator asset files are excluded. Their\n"
              "upstream terms remain separate from implementation licenses. No blanket license\n"
              "is asserted for this archive, model-derived outputs or external assets. Consult\n"
              "docs/reference/realtime-vla-assets.md and docs/reference/pi05-libero-assets.md.\n\n"
              f"Existing public notice context: {notice_source.as_posix()} (SHA256 {notice_sha}).\n")
    members = {'README-inference-profile.md': readme.encode(), 'THIRD_PARTY_NOTICES.md': notice.encode(),
               'LICENSES/inference-profile-sources.json': (json.dumps([lerobot], indent=2) + '\n').encode()}
    if test_receipt is not None:
        members['TEST-RESULTS.txt'] = test_receipt
    return members


def prepare_members(paths, generated):
    """Read originals once, scan/redact textual derivatives, and record both byte identities.

    Args: paths: checked relative originals; generated: extra authored member bytes.
    Returns: Distributed bytes and manifest records.
    Raises: Secret-pattern, encoding or duplicate member errors.
    Design: Context: Original evidence must remain reviewable locally.
        Concern: Unrecorded redaction or mixed original/derived identities.
        Decision: Keep source paths relative and store original/distributed SHA, sizes and redaction counts.
        Alternatives: Rewrite raw files. Quality Attributes: Auditability.
        Trade-offs: Selected original bytes remain in memory until archive completion.
    Evidence: Internal: Fixed allowlist. External: SHA256. Findings: NPZ/PNG are unchanged.
        Metrics: Both hashes and substitution counts. Confidence: Exact bytes.
    Constraints: No original writes. Reconsider When: Archive size requires streaming.
    References: build_package. ADR: None.
    """
    members, records, aliases = {}, [], {}
    for relative, generated_data in [(path, None) for path in paths] + list(generated.items()):
        if relative in members or relative.startswith('/') or '..' in Path(relative).parts:
            raise ValueError('Unsafe or duplicate archive member')
        original = (ROOT / relative).read_bytes() if generated_data is None else generated_data
        is_text = Path(relative).suffix in TEXT_SUFFIXES
        if is_text:
            reject_secrets(original, relative)
            distributed, counts = redact_homes(original, aliases)
            reject_secrets(distributed, relative)
        else:
            distributed, counts = original, {}
        members[relative] = distributed
        records.append({'path': relative, 'original_path': relative if generated_data is None else None,
                        'original_sha256': sha256(original), 'distributed_sha256': sha256(distributed),
                        'original_bytes': len(original), 'distributed_bytes': len(distributed),
                        'redaction_counts': counts, 'redaction_count': sum(counts.values()),
                        'derived_text': distributed != original, 'generated_member': generated_data is not None,
                        'original_unchanged_checked': False if generated_data is None else None})
    return members, records


def run_cpu_tests():
    """Capture current CPU protocol/input tests without model, CUDA or simulator execution.

    Args: None. Returns: UTF8 command/result transcript bytes. Raises: RuntimeError if pytest fails.
    Design: Context: The package should include test evidence, not just test source.
        Concern: GPU/model activity during packaging. Decision: Run the two known CPU-only harness/input suites.
        Alternatives: Run the full GPU/simulator suite. Quality Attributes: Isolation.
        Trade-offs: This does not validate model performance or report rendering.
    Evidence: Internal: tests/test_inference_profiling.py, tests/test_profile_inputs.py.
        External: Pytest. Findings: Test logs are included in the ZIP.
        Metrics: Exit status and pytest count. Confidence: CPU contracts only.
    Constraints: No GPU imports in selected tests. Reconsider When: Test dependency scope changes.
    References: TEST-RESULTS.txt. ADR: None.
    """
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_inference_profiling.py', 'tests/test_profile_inputs.py']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError('CPU inference profiling tests failed; archive was not written')
    return ('UTC: ' + datetime.now(timezone.utc).isoformat() + '\nCommand: python -m pytest -q '
            'tests/test_inference_profiling.py tests/test_profile_inputs.py\nExit: 0\n'
            + result.stdout + result.stderr).encode()


def build_package(args):
    """Create only fresh owned output artifacts after full allowlist/source/archive verification.

    Args: args: parsed CLI controls. Returns: Summary mapping. Raises: Missing/changed/secrets/ZIP errors.
    Design: Context: Finalized evidence is copied for publication.
        Concern: Active writers and self-overwriting archives hide provenance changes.
        Decision: Refuse existing outputs, verify every CRC/hash/member and rehash originals after packaging.
        Alternatives: Append or replace an older evidence pack. Quality Attributes: Integrity.
        Trade-offs: Rebuilds require a separately named output or deliberate external removal.
    Evidence: Internal: Frozen source proof and all selected original hashes. External: zipfile.testzip.
        Findings: FAST invalid flags are retained rather than overridden.
        Metrics: Archive SHA/size, unchanged originals and redaction counts. Confidence: Exact archive check.
    Constraints: No writes to selected originals or public report/PDF/README candidates.
    Reconsider When: Resumable publication is explicitly requested.
    References: INFERENCE-PROFILE-MANIFEST.json. ADR: None.
    """
    paths, proof = select_files(args.independent_audit)
    if not args.write:
        _, records = prepare_members(paths, generated_members())
        return {'mode': 'dry_run', 'selected_original_files': len(paths), 'members_with_generated': len(records) + 1,
                'redaction_count': sum(record['redaction_count'] for record in records), 'source_snapshot_verified': len(proof),
                'original_files_written': False, 'ready': True}
    target, audit_target = args.output.resolve(), args.audit_output.resolve()
    if not target.is_relative_to(ROOT) or not audit_target.is_relative_to(ROOT):
        raise ValueError('Package outputs must stay inside this workspace')
    if target.exists() or audit_target.exists() or target == audit_target:
        raise ValueError('Package ZIP and audit destinations must both be fresh')
    tests = run_cpu_tests()
    members, records = prepare_members(paths, generated_members(tests))
    flags = {model: json.loads((ROOT / PRIMARY / f'{model}-native/report.json').read_text())['validation'] for model in MODELS}
    manifest = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'actual native inference evidence; failed FAST paired equality retained as excluded reference',
                'source_snapshot_proof': proof, 'source_snapshot_files_verified': len(proof),
                'original_primary_validation_flags': flags, 'files': records,
                'redaction_policy': 'UTF8 home prefixes only, consistent participant-N aliases; originals untouched; reverse alias mapping not exported',
                'external_installed_source': 'original paths/hashes only in redacted reports; no dependency files copied',
                'excluded': ['model weights', 'tokenizer bodies', 'virtual environments', 'caches', 'arbitrary dependency source',
                             'unrelated historical experiments', 'review PDF/README candidates', '.git', 'credentials']}
    target.parent.mkdir(parents=True, exist_ok=True)
    audit_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    if temporary.exists():
        raise ValueError('Temporary archive destination must be fresh')
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in members.items():
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                mode = 0o755 if name.endswith('.sh') else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
            for record in records:
                if record['original_path'] is not None:
                    unchanged = sha256((ROOT / record['original_path']).read_bytes()) == record['original_sha256']
                    record['original_unchanged_checked'] = unchanged
                    if not unchanged:
                        raise ValueError(f'Original changed during packaging: {record["original_path"]}')
            archive.writestr('INFERENCE-PROFILE-MANIFEST.json', json.dumps(manifest, indent=2).encode() + b'\n')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(members) | {'INFERENCE-PROFILE-MANIFEST.json'}:
                raise ValueError('Archive member/CRC validation failed')
            for record in records:
                if sha256(archive.read(record['path'])) != record['distributed_sha256']:
                    raise ValueError('Distributed archive member hash mismatch')
            stored_manifest = json.loads(archive.read('INFERENCE-PROFILE-MANIFEST.json'))
            if stored_manifest != manifest:
                raise ValueError('Stored package manifest differs from verified manifest')
        validate_frozen_sources()
        for record in records:
            if record['original_path'] is not None and sha256((ROOT / record['original_path']).read_bytes()) != record['original_sha256']:
                raise ValueError(f'Original changed during final archive validation: {record["original_path"]}')
        temporary.replace(target)
        audit = {'status': 'complete', 'zip': target.relative_to(ROOT).as_posix(), 'zip_sha256': sha256(target.read_bytes()),
                 'zip_bytes': target.stat().st_size, 'archive_members': len(members) + 1,
                 'redaction_count': sum(record['redaction_count'] for record in records),
                 'all_original_files_unchanged': True, 'all_distributed_member_hashes_verified': True,
                 'zip_crc_verified': True, 'source_snapshot_verified': True, 'secret_pattern_checks_passed': True,
                 'manifest': manifest}
        with audit_target.open('x') as stream:
            json.dump(audit, stream, indent=2)
            stream.write('\n')
        return {key: value for key, value in audit.items() if key != 'manifest'}
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    """Parse packaging controls and report safe summary metadata only.

    Args: argv: optional argument list. Returns: Zero on success. Raises: Package validation errors.
    Design: Context: Root waits until parallel documentation/audit writers finish.
        Concern: Premature publication. Decision: Dry-run default and explicit --write to fresh destinations.
        Alternatives: Write during import. Quality Attributes: Predictability. Trade-offs: Final invocation required.
    Evidence: Internal: allowlist and unchanged checks. External: argparse.
        Findings: Help/import need no model backend. Metrics: CLI exit. Confidence: CPU verified.
    Constraints: No network/Git/Site/model actions. Reconsider When: Publication workflow changes.
    References: build_package. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--independent-audit', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / ARTIFACT / 'inference-profile-evidence.zip')
    parser.add_argument('--audit-output', type=Path, default=ROOT / ARTIFACT / 'inference-profile-packaging-audit.json')
    args = parser.parse_args(argv)
    print(json.dumps(build_package(args), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
