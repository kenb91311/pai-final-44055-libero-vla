"""Correlate completed continuity trials with sampled host/GPU and bootstrap evidence.

Args: A completed formal report, shared NVML trace, monitor config, optional prior-data checksums.
Returns: New resource coverage and initial-action parity audit; no inference is performed.
Raises: Invalid/missing evidence or mismatched paired initial states.
Design:
    Context: Team-level model leases did not originally establish shared-host exclusivity.
    Concern: Unscoped GPU snapshots can be incorrectly attributed to a particular method.
    Decision: Use monotonic task-worker envelopes, bracketing trace samples and explicit scope limits.
    Alternatives: Claim process attribution from global memory alone rejected.
    Quality Attributes: Auditability. Trade-offs: Sub-sample activity can remain invisible.
Evidence:
    Internal: Actual chunk timestamps, complete raw trace prefixes and frozen-run reports.
    External: NVML whole-device measurement semantics.
    Findings: Coordination, continuous sampled coverage and byte-identical inputs are distinct evidence.
    Metrics: Coverage gaps, GPU bytes/utilization, host pressure, bootstrap equality and data preservation.
    Confidence: Scoped sampled evidence, not hardware isolation certification.
Constraints: CPU-only; never load a model, probe hardware, stop a process or alter experiment data.
Reconsider When: Independent process accounting or a dedicated GPU is available.
References: monitor_resources.py; summarize_continuity.py. ADR: None.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from summarize import ROOT,statistics_of


def read_json(path):
    """Read one existing recorded object, preserving missing evidence as an error.

    Args: path: recorded JSON. Returns: Parsed value. Raises: IO/JSON errors.
    Design: Context: Evidence-only analysis. Concern: Fabricated defaults.
        Decision: Strict read. Alternatives: Empty fallback rejected.
        Quality Attributes: Integrity. Trade-offs: Caller sees failures.
    Evidence: Internal: Files. External: JSON. Findings: No source writes.
        Metrics: None. Confidence: IO contract.
    Constraints: Local trusted data. Reconsider When: Remote streaming.
    References: main. ADR: None.
    """
    return json.loads(Path(path).read_text())


def trace_snapshot(path):
    """Read one complete append-only trace prefix without turning a partial line into a sample.

    Args: path: running or completed samples.jsonl. Returns: (rows, prefix identity).
    Raises: ValueError for absent, nonfinite or non-monotonic trace data.
    Design: Context: One monitor spans several serial experiment families.
        Concern: A later append changes whole-file SHA while old samples remain valid.
        Decision: Capture complete lines once, record prefix byte length and hash.
        Alternatives: Copy a live partially written row or retimestamp samples rejected.
        Quality Attributes: Traceability. Trade-offs: At most one incomplete trailing row excluded.
    Evidence: Internal: UTC/monotonic fields from root monitor.
        External: Append-only file semantics. Findings: Original bytes remain unmodified.
        Metrics: Captured sample/byte counts. Confidence: Exact byte prefix.
    Constraints: Do not infer process identity from whole-device counters.
    Reconsider When: Trace schema changes.
    References: monitor_resources.py. ADR: None.
    """
    data=Path(path).read_bytes();partial=bool(data and not data.endswith(b'\n'))
    if partial:data=data[:data.rfind(b'\n')+1]
    rows=[json.loads(line) for line in data.splitlines() if line]
    if not rows or any(not np.isfinite(row['monotonic']) for row in rows):raise ValueError('missing/invalid resource samples')
    if any(b['monotonic']<=a['monotonic'] for a,b in zip(rows,rows[1:])):raise ValueError('resource trace monotonic order invalid')
    path=Path(path).resolve()
    return rows,{'path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),'captured_bytes':len(data),
                 'captured_sha256':hashlib.sha256(data).hexdigest(),'captured_samples':len(rows),'partial_trailing_line_excluded':partial}


def window_coverage(rows,start,end,period):
    """Measure sampled coverage of a monotonic worker envelope using boundary brackets.

    Args: rows: complete chronological trace; start/end: worker envelope; period: configured seconds.
    Returns: Coverage/sampling/resource statistics. Raises: Invalid time window.
    Design: Context: Worker timestamps share the host monotonic clock with the monitor.
        Concern: Whole-trace peaks could include a later model's unrelated allocation.
        Decision: Scope samples to this envelope and retain nearest bracketing samples for gaps.
        Alternatives: Compare global trace maximum across serial models rejected.
        Quality Attributes: Causal scope. Trade-offs: Worker envelope excludes preceding model startup.
    Evidence: Internal: Actual start/finish values in chunks.json.
        External: Shared monotonic clock. Findings: Probe/host pressure remains observable.
        Metrics: Sample gaps, GPU bytes/utilization and probe cost.
        Confidence: Sampled counters only; not per-kernel attribution.
    Constraints: Coverage threshold is2×configured period and is reported explicitly.
    Reconsider When: Sampling cadence or clocks differ.
    References: trace_snapshot. ADR: None.
    """
    if not np.isfinite(start) or not np.isfinite(end) or end<start or period<=0:raise ValueError('invalid worker envelope')
    inside=[row for row in rows if start<=row['monotonic']<=end]
    before=[row for row in rows if row['monotonic']<=start];after=[row for row in rows if row['monotonic']>=end]
    bracketed=bool(before and after);cover=([before[-1]] if before else [])+inside+([after[0]] if after else [])
    unique={row['monotonic']:row for row in cover};cover=[unique[key] for key in sorted(unique)]
    gaps=[float(b['monotonic']-a['monotonic']) for a,b in zip(cover,cover[1:])]
    max_gap=max(gaps,default=None);gap_limit=2*period
    return {'status':'covered' if bracketed and max_gap is not None and max_gap<=gap_limit else 'coverage_gap',
            'start_monotonic':start,'end_monotonic':end,'duration_s':end-start,'samples_inside':len(inside),
            'bracketed_by_samples':bracketed,'maximum_bracketed_gap_s':max_gap,'gap_limit_s':gap_limit,
            'whole_gpu_used_MiB':statistics_of(float(row['whole_gpu_used_bytes']/1024**2) for row in inside),
            'whole_gpu_utilization_percent':statistics_of(float(row['whole_gpu_utilization_percent']) for row in inside),
            'host_memory_available_MiB':statistics_of(float(row['host_memory_available_bytes']/1024**2) for row in inside),
            'probe_ms':statistics_of(float(row['probe_wall_s']*1000) for row in inside),
            'phase_labels':sorted({str(row.get('phase')) for row in inside}),
            'phase_read_failure_count':sum(bool(row.get('phase_read_failed')) for row in inside),
            'scope':'whole-device/host samples within task-worker envelope; includes calibration/reset gaps between task chunks, excludes initial model restoration'}


def audit(report_path,trace_path,*,preservation_reference=None):
    """Audit formal/dev resource envelopes, exact initial chunks and optional prior-data hashes.

    Args: report_path; trace_path; preservation_reference: optional JSON with baseline35_data_sha256_before.
    Returns: New validation mapping. Raises: Missing/mismatched episode data or source identity.
    Design: Context: Clean matched series is separate from the initial provisional series.
        Concern: Identical sensors alone do not guarantee identical first model output.
        Decision: Compare physical/normalized bootstrap chunks by model/state across variants.
        Alternatives: Pool old successes or use screenshots as provenance rejected.
        Quality Attributes: Integrity. Trade-offs: Requires retained raw episode chunks.
    Evidence: Internal: Frozen report, initial hashes, chunk arrays and NVML source snapshots.
        External: None. Findings: Runtime guidance only starts after bootstrap.
        Metrics: Exact paired arrays, coverage windows and old-file checks.
        Confidence: File evidence and sampled resource scope.
    Constraints: No exclusive-GPU certification from memory alone.
    Reconsider When: Bootstrap itself is intentionally conditioned.
    References: summarize_continuity.py. ADR: None.
    """
    report=read_json(report_path);rows,identity=trace_snapshot(trace_path);config=read_json(Path(trace_path).parent/'configuration.json')
    source=Path(trace_path).parent/'monitor-source.py';source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if config['source_sha256']!=source_sha:raise ValueError('resource monitor source snapshot differs')
    formal=report['episodes'];development=report.get('development_episodes_excluded',[]);all_episodes=formal+development
    groups=[];bootstrap={}
    for group in report['groups']:
        model=group['model'];variant=group['variant'];selected=[row for row in all_episodes if row['model']==model and row['variant']==variant]
        chunks=[]
        for row in selected:
            episode_chunks=read_json(ROOT/row['source_directory']/'chunks.json');chunks.extend(episode_chunks)
            first=next(chunk for chunk in episode_chunks if chunk.get('bootstrap'))
            bootstrap[(model,row['init_state_index'],variant)]={'actions':np.asarray(first['actions']),
                'normalized':np.asarray(first['normalized']),'sensors':row['initial_sensor_hashes']}
        timed=[chunk for chunk in chunks if chunk.get('kind')=='chunk']
        coverage=window_coverage(rows,min(chunk['started_time'] for chunk in timed),max(chunk['finished_time'] for chunk in timed),config['period_s'])
        groups.append({'model':model,'variant':variant,'development_included_in_coverage':bool(development),
                       'formal_success_count':group['successes'],'formal_n':group['n'],'resource_window':coverage})
    paired=[]
    for model in sorted({key[0] for key in bootstrap}):
        for state in sorted({key[1] for key in bootstrap if key[0]==model}):
            variants=sorted(key[2] for key in bootstrap if key[:2]==(model,state));first=bootstrap[(model,state,variants[0])]
            pairs=[bootstrap[(model,state,variant)] for variant in variants[1:]]
            paired.append({'model':model,'state':state,'is_development_excluded':state==0,'variants':variants,
                'sensor_hashes_exact':all(first['sensors']==row['sensors'] for row in pairs),
                'physical_actions_exact':all(np.array_equal(first['actions'],row['actions']) for row in pairs),
                'normalized_actions_exact':all(np.array_equal(first['normalized'],row['normalized']) for row in pairs)})
    preservation={'status':'not_requested'}
    if preservation_reference is not None:
        expected=read_json(preservation_reference)['baseline35_data_sha256_before']
        changed=[path for path,sha in expected.items() if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha]
        preservation={'status':'passed' if not changed else 'changed_files_detected','checked_files':len(expected),'changed':changed,
                      'reference':str(preservation_reference),'reference_sha256':hashlib.sha256(Path(preservation_reference).read_bytes()).hexdigest()}
    return {'status':'passed' if all(g['resource_window']['status']=='covered' for g in groups) and all(r['sensor_hashes_exact'] and r['physical_actions_exact'] and r['normalized_actions_exact'] for r in paired) and preservation['status']!='changed_files_detected' else 'review_required',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),'formal_episodes':len(formal),'development_episodes_excluded':len(development),
        'trace_snapshot':identity,'monitor_source_sha256':source_sha,'configured_sample_period_s':config['period_s'],'groups':groups,
        'paired_initial_sensing_and_bootstrap':paired,'prior35_preservation':preservation,
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits':['Sampled whole-device counters and participant coordination support resource isolation; neither proves hardware exclusivity.',
                  'Device counters include desktop/rendering and cannot attribute individual kernels or every sub-sample allocation.',
                  'Task-worker envelopes are explicitly scoped; preceding model restoration and final state hashing are not the reported inference windows.']}


def main():
    """Write one new evidence sidecar after CPU-only validation.

    Args: CLI report/trace/output and optional preservation reference.
    Returns: None. Raises: Existing output or invalid source evidence.
    Design: Context: Shared monitor continues after this series.
        Concern: Reanalysis must not overwrite numerical experiment results.
        Decision: Separate fresh sidecar with captured-prefix provenance.
        Alternatives: Rewrite raw results rejected. Quality Attributes: Traceability.
        Trade-offs: Consumers read validation alongside metrics.
    Evidence: Internal: Source hashes and raw arrays. External: None.
        Findings: No GPU/model activity. Metrics: Coverage and pairing.
        Confidence: Scoped audit.
    Constraints: CPU only. Reconsider When: Report schema changes.
    References: audit. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--preservation-reference',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('resource audit output already exists')
    result=audit(args.report,args.trace,preservation_reference=args.preservation_reference)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'formal':result['formal_episodes'],'development_excluded':result['development_episodes_excluded'],
        'resource_windows':[(g['variant'],g['resource_window']['status']) for g in result['groups']]}))


if __name__=='__main__':main()
