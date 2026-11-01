"""Audit and summarize the separately registered H50/R30 RTC comparison.

Args: Completed per-variant run directories, registered protocol, bridge logs and output folder.
Returns: Formal-only metrics, initial-sensor pairing, frozen/source proof and independent ROS receipts.
Raises: ValueError for incomplete/mismatched protocol, state grid, hashes or recorded evidence.
Design:
    Context: Scheduler changes must not be mixed into the archived E10 benchmark.
    Concern: Metadata alone does not independently establish the ROS node path.
    Decision: Reuse existing episode metrics and match wire IDs against both nodes' receipt logs.
    Alternatives: Pool development episodes or infer task success from distance rejected.
    Quality Attributes: Traceability, comparability. Trade-offs: Requires retained bridge logs.
Evidence:
    Internal: Per-tick/chunk/initial/source manifests and complete-state model audits.
    External: Native LIBERO success predicate and Wilson descriptive interval.
    Findings: Chunk switching, control freshness and compute cost need separate measures.
    Metrics: Success, hold, actual/predicted delay, source age, control period and command deltas.
    Confidence: Five fixed initial states per method on one known task.
Constraints: Formal states1..5 only; no new experiments, hidden filtering or broad benchmark claims.
Reconsider When: Another timing protocol or random task sample is registered.
References: continuity-protocol.json; summarize.py. ADR: None.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from summarize import ROOT,read_episode,statistics_of,wilson


def load_json(path):
    """Read one strict recorded JSON object without altering source artifacts.

    Args: path: existing file. Returns: Parsed object. Raises: IO/JSON errors.
    Design: Context: Read-only evidence. Concern: Missing data must remain visible.
        Decision: No fallback. Alternatives: Empty fake data rejected.
        Quality Attributes: Integrity. Trade-offs: Caller handles failure.
    Evidence: Internal: Immutable episode folders. External: JSON.
        Findings: No source mutation. Metrics: None. Confidence: IO contract.
    Constraints: Trusted local artifacts. Reconsider When: Remote streaming is used.
    References: summarize_runs. ADR: None.
    """
    return json.loads(Path(path).read_text())


def source_record(path,*,captured_data=None):
    """Record source bytes and a project-relative location for reproducible evidence.

    Args: path: existing local file; captured_data: optional already-read byte snapshot.
    Returns: Path/byte count/SHA256 mapping.
    Raises: IO errors. Design: Context: Appended logs need snapshot-length provenance.
        Concern: Later appends change whole-file hashes.
        Decision: Record captured byte length so the original prefix can be verified.
        Alternatives: Filename alone rejected. Quality Attributes: Traceability.
        Trade-offs: Full evidence file read.
    Evidence: Internal: Bytes read once. External: SHA256.
        Findings: Future append-only data does not change this captured prefix.
        Metrics: Captured bytes. Confidence: Exact identity.
    Constraints: No model-weight files. Reconsider When: Logs become very large.
    References: audit_bridge. ADR: None.
    """
    path=Path(path).resolve();data=path.read_bytes() if captured_data is None else captured_data
    return {'path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def audit_bridge(episodes,bridge,*,sample_scope='formal'):
    """Match every task chunk's wire IDs to independent policy and broker receipt records.

    Args: episodes: compact records with source_directory; bridge: node log root;
        sample_scope: explicit formal or excluded-development label.
    Returns: Audit summary and the exact matched receipt events for archival.
    Raises: ValueError for missing or inconsistent wire identity/hash verification.
    Design: Context: Sensor/action RPC metadata is generated in a different process.
        Concern: A claimed receipt without a matching node event is weak evidence.
        Decision: Require both roles and matching source/received hashes and request ID.
        Alternatives: Count publications or metadata-only hashes rejected.
        Quality Attributes: Independent traceability. Trade-offs: Shared logs contain other runs.
    Evidence: Internal: events.jsonl from each actual ROS process.
        External: ROS2 reliable topic path. Findings: Wire UUID differs from model request ID.
        Metrics:2 exchanges and4 node receipts per generated task chunk.
        Confidence: Direct log correlation; implementation provenance also retained.
    Constraints: Bootstrap and drained task chunks included; untimed calibration excluded here.
    Reconsider When: ROS bridge log schema changes.
    References: ros_bridge.py; ros_client.py. ADR: None.
    """
    index=defaultdict(list);sources=[]
    for role in ('policy','broker'):
        path=Path(bridge)/role/'events.jsonl';data=path.read_bytes()
        if data and not data.endswith(b'\n'):raise ValueError('bridge snapshot ends in a partial event')
        record=source_record(path,captured_data=data);record['role']=role;record['captured_prefix_semantics']=True;sources.append(record)
        for line in data.splitlines():
            event=json.loads(line)
            if 'id' in event:index[(role,event['event'],event['id'])].append(event)
    matched=[];chunks_count=0
    for episode in episodes:
        for chunk in load_json(ROOT/episode['source_directory']/'chunks.json'):
            if chunk.get('kind')!='chunk':continue
            chunks_count+=1
            for side,policy_event,broker_event in (
                ('sensor_transport','sensor_join_receive','sensor_receipt'),
                ('action_transport','generated_chunk_receive','action_receipt')):
                metadata=chunk['metadata'][side];wire=metadata['transport_id'];expected=metadata['source_sha256']
                if metadata.get('verified') is not True or metadata.get('received_sha256')!=expected or metadata.get('peer_received_sha256')!=expected:
                    raise ValueError(f'metadata receipt mismatch: {chunk["request_id"]}')
                for role,event_name in (('policy',policy_event),('broker',broker_event)):
                    records=index[(role,event_name,wire)]
                    valid=[r for r in records if r.get('verified') is True and r.get('source_sha256')==expected==r.get('received_sha256') and r.get('request_id')==chunk['request_id']]
                    if not valid or len(valid)!=len(records):raise ValueError(f'missing/inconsistent independent {role}/{event_name} wire receipt')
                    matched.append({'episode':episode['source_directory'],'sample_scope':sample_scope,'model_request_id':chunk['request_id'],
                                    'side':side,'role':role,'wire_transport_id':wire,'event':valid[0]})
    return {'status':'passed','task_chunks':chunks_count,'verified_exchanges':2*chunks_count,
            'verified_node_receipts':len(matched),'sources':sources,
            'scope':sample_scope+' episode chunks including bootstrap and late-drained results; excludes untimed calibration'},matched


def overlap_switch_records(ticks,chunks):
    """Compare old planned and new executed commands at the same chunk-switch tick.

    Args: ticks: contiguous actuator records; chunks: native generated task chunks.
    Returns: (overlap disagreement records, number of switches after a hold).
    Raises: Key/index errors for inconsistent request provenance.
    Design: Context: Consecutive-action differences include ordinary planned motion.
        Concern: Calling all natural action changes a chunk mode jump overstates discontinuity.
        Decision: Reconstruct the old next command for the exact switch tick before replacement.
        Alternatives: Use action[t]-action[t-1] alone rejected as a mode-specific measure.
        Quality Attributes: Interpretability. Trade-offs: No counterfactual motion after exhaustion.
    Evidence: Internal: Full native chunks and executed chunk indices.
        External: Same-time action overlap semantics. Findings: Both commands are clipped identically.
        Metrics: Translation/rotation/gripper command disagreement, with sample counts.
        Confidence: Exact command reconstruction, not proof of multimodal latent intent.
    Constraints: Separate normalized control units; do not label differences as meters/radians.
    Reconsider When: Scheduler modifies commands beyond fixed clipping.
    References: ChunkQueue.accept; run_episode. ADR: None.
    """
    by_request={chunk['request_id']:chunk for chunk in chunks if chunk.get('kind')=='chunk'}
    overlap=[];after_hold=0
    for previous,current in zip(ticks,ticks[1:]):
        if not current['chunk_switched']:continue
        if previous['held']:
            after_hold+=1;continue
        old=by_request[previous['request_id']];next_index=previous['chunk_index']+1
        if next_index>=len(old['actions']):continue
        expected_old=np.clip(np.asarray(old['actions'][next_index],dtype=np.float64),-1,1)
        disagreement=np.asarray(current['action'],dtype=np.float64)-expected_old
        overlap.append({'tick':current['tick'],'previous_request_id':previous['request_id'],'new_request_id':current['request_id'],
            'previous_next_chunk_index':next_index,'translation_l2':float(np.linalg.norm(disagreement[:3])),
            'rotation_l2':float(np.linalg.norm(disagreement[3:6])),'gripper_abs':float(abs(disagreement[6]))})
    return overlap,after_hold


def validate_summary_state_order(configured,recorded,formal,*,allow_development_state0=False):
    """Accept exactly formal1..5, or explicitly ordered development0 then formal1..5.

    Args: configured/recorded: CLI and result state lists; formal: registered formal states;
        allow_development_state0: defaultFalse, explicit same-process development amendment.
    Returns: Separate formal/development state lists without silently dropping unknown states.
    Raises: ValueError on unexpected, duplicate, missing or reordered states.
    Design: Context: Reusing a warmed model saves repeated checkpoint loading.
        Concern: Treating all six states as formal would inflate the planned denominator.
        Decision: Gate0 by an explicit flag and require exact configured/result order.
        Alternatives: Filter arbitrary states silently rejected.
        Quality Attributes: Sample integrity. Trade-offs: Reordered valid data needs a new protocol.
    Evidence: Internal: State-grid boundary tests. External: None.
        Findings: Development0 remains explicit and excluded from all formal outcome metrics.
        Metrics:5formal/optional1development. Confidence: Contract-level.
    Constraints: This helper never selects hyperparameters using development success.
    Reconsider When: A different formal state grid is registered.
    References: summarize_runs. ADR: None.
    """
    expected=([0]+list(formal)) if allow_development_state0 else list(formal)
    if configured!=expected:raise ValueError('configured state order differs from declared summary mode')
    if recorded!=expected:raise ValueError('summary grid/order differs')
    return {'formal':list(formal),'development':[0] if allow_development_state0 else []}


def summarize_runs(run_directories,protocol_path,bridge,*,allow_development_state0_in_summary=False):
    """Require a complete registered formal grid and recompute scoped method metrics.

    Args: run_directories: completed variant folders; protocol_path; bridge: ROS log root;
        allow_development_state0_in_summary: explicitly accept ordered0..5 runs and keep0 separate.
    Returns: Report JSON mapping and independent matched node receipt list.
    Raises: ValueError for protocol/audit/grid inconsistencies.
    Design: Context: Smaller switch deltas alone do not prove improved task execution.
        Concern: Different chunk schedules, samples or stale observations confound comparison.
        Decision: Same protocol/state grid and separate success, age, hold and command measures.
        Alternatives: Mix original E10 results or development0 into these counts rejected.
        Quality Attributes: Causal clarity. Trade-offs: Small deterministic sample remains limited.
    Evidence: Internal: Recomputed tick metrics and original source hashes.
        External: Native success; descriptive Wilson bounds. Findings:5fixed states per variant.
        Metrics: Per-episode means plus pooled metrics explicitly labeled.
        Confidence: Empirical on this task/system, not a population benchmark.
    Constraints: No output artifact may replace original35-trial evidence.
    Reconsider When: New scheduling conditions are separately registered.
    References: read_episode; audit_bridge. ADR: None.
    """
    protocol=load_json(protocol_path);states=protocol['formal_initial_states'];episodes=[];development=[];audits={};sources=[];groups=[]
    identities=set()
    for directory in map(Path,run_directories):
        summary=load_json(directory/'summary.json');config=load_json(directory/'config.json');manifest=load_json(directory/'source-sha256.json')
        if summary.get('status')!='complete':raise ValueError(f'run incomplete: {directory.name}')
        model=config['model'];variant=config['variant'];identity=(model,variant)
        if identity in identities:raise ValueError('duplicate model/variant run')
        identities.add(identity)
        if model not in protocol['frozen_families'] or variant not in protocol['variants']:raise ValueError('unregistered method')
        for key,wanted in protocol['source_sha256'].items():
            if manifest.get(key)!=wanted:raise ValueError(f'registered source differs: {key}')
        audit=load_json(directory/'model-audit.json')
        if not audit.get('weights_unchanged') or audit.get('local_training') is not False or audit.get('requires_grad_parameter_count')!=0 or audit.get('parameters_with_grad')!=0:
            raise ValueError(f'frozen/gradient audit failed: {directory.name}')
        audits[f'{model}/{variant}']=audit
        validate_summary_state_order(config.get('initial_states'),[r['init_state_index'] for r in summary['results']],states,
                                     allow_development_state0=allow_development_state0_in_summary)
        if allow_development_state0_in_summary:
            dev_dir=directory/'realtime-init0';dev=read_episode(dev_dir)
            if (dev['status'],dev['model'],dev['variant'],dev['init_state_index'])!=('completed',model,variant,0) or not dev['ros_all_verified']:
                raise ValueError('explicit development0 record invalid')
            if (dev['control_hz'],dev['wall_budget_s'],dev['execute_steps'],dev['prefetch_steps'],dev['rtc_execution_horizon'],dev['merge'],dev['seed'],dev['ros_transport'],dev['camera_hz'])!=(20,14,50,30,30,'aligned',protocol['seed'],True,10):
                raise ValueError('development0 timing protocol differs')
            development.append(dev)
            for name in ('initial.json','result.json','ticks.jsonl','chunks.json','guided-calibration.json'):sources.append(source_record(dev_dir/name))
        rows=[];delays=[];coverage=[];actual_coverage=[];guidance_coverage=[];switches=[];overlap_disagreements=[];steady_chunks=[]
        for state in states:
            episode_dir=directory/f'realtime-init{state}';row=read_episode(episode_dir)
            if row['status']!='completed' or (row['model'],row['variant'],row['init_state_index'])!=(model,variant,state):raise ValueError('episode identity/status mismatch')
            if (row['control_hz'],row['wall_budget_s'],row['execute_steps'],row['prefetch_steps'],row['rtc_execution_horizon'],row['merge'],row['seed'],row['ros_transport'],row['camera_hz'])!=(20,14,50,30,30,'aligned',protocol['seed'],True,10):
                raise ValueError('episode timing protocol differs')
            if not row['ros_all_verified']:raise ValueError('episode metadata ROS hashes failed')
            chunks=load_json(episode_dir/'chunks.json')
            for chunk in chunks:
                if chunk.get('bootstrap') or chunk.get('kind')!='chunk':continue
                steady_chunks.append(chunk)
                if 'actual_delay_ticks' in chunk:
                    predicted=chunk['metadata']['predicted_delay_ticks'];raw=chunk['metadata']['raw_predicted_delay_ticks'];actual=chunk['actual_delay_ticks']
                    delays.append({'state':state,'request_id':chunk['request_id'],'predicted':predicted,'raw_predicted':raw,'actual':actual,'actual_minus_predicted':actual-predicted,
                                   'known_prefix_steps':chunk['metadata']['prefix_length'],'guidance_horizon':chunk['metadata']['execution_horizon'],
                                   'saturated':chunk['metadata']['delay_prediction_saturated']})
                    coverage.append(chunk['metadata']['prefix_covers_estimated_delay'])
                    actual_coverage.append(actual<=chunk['metadata']['prefix_length'])
                    guidance_coverage.append(actual<=chunk['metadata']['execution_horizon'])
            physical_ticks=[json.loads(line) for line in (episode_dir/'ticks.jsonl').read_text().splitlines() if line]
            overlap,after_hold_switches=overlap_switch_records(physical_ticks,chunks)
            row['overlap_disagreements']=overlap;row['switches_after_hold']=after_hold_switches;overlap_disagreements.extend(overlap)
            bowl_z=[tick['metrics']['bowl_position'][2] for tick in physical_ticks]
            start_z=row['initial_physical_metrics']['bowl_position'][2]
            row['task_progress_diagnostics']={'maximum_bowl_lift_mm':(max(bowl_z)-start_z)*1000,
                'final_bowl_height_change_mm':(bowl_z[-1]-start_z)*1000,
                'minimum_goal_xy_mm':min(tick['metrics']['goal_xy_error_m'] for tick in physical_ticks)*1000,
                'scope':'evaluator object measurements for failure localization; neither lift nor XY distance substitutes native success'}
            switches.extend(row['switches']);rows.append(row);episodes.append(row)
            for name in ('initial.json','result.json','ticks.jsonl','chunks.json','guided-calibration.json'):
                sources.append(source_record(episode_dir/name))
        successes=sum(r['success'] for r in rows)
        group={'model':model,'variant':variant,'n':len(rows),'successes':successes,'success_wilson95':wilson(successes,len(rows)),
            'success_by_state':{str(r['init_state_index']):r['success'] for r in rows},
            'successes_within_wall_budget':sum(r['success_within_wall_budget'] for r in rows),
            'pooled_hold_fraction':sum(r['hold_ticks'] for r in rows)/sum(r['ticks'] for r in rows),
            'final_goal_xy_mm':statistics_of(float(r['goal_xy_error_m']*1000) for r in rows),
            'effective_control_hz':statistics_of(r['effective_control_hz'] for r in rows),
            'effective_received_camera_hz':statistics_of(r['effective_received_camera_hz'] for r in rows),
            'realtime_factor':statistics_of(r['realtime_factor'] for r in rows),
            'switch_count':len(switches),'switch_translation_delta_l2':statistics_of(s['translation_delta_l2'] for s in switches),
            'switch_rotation_delta_l2':statistics_of(s['rotation_delta_l2'] for s in switches),
            'switch_gripper_delta_abs':statistics_of(s['gripper_delta_abs'] for s in switches),
            'overlap_disagreement_count':len(overlap_disagreements),'switches_after_hold':sum(r['switches_after_hold'] for r in rows),
            'overlap_disagreement_translation_l2':statistics_of(s['translation_l2'] for s in overlap_disagreements),
            'overlap_disagreement_rotation_l2':statistics_of(s['rotation_l2'] for s in overlap_disagreements),
            'overlap_disagreement_gripper_abs':statistics_of(s['gripper_abs'] for s in overlap_disagreements),
            'actual_delay_ticks':statistics_of(d['actual'] for d in delays),'predicted_delay_ticks':statistics_of(d['predicted'] for d in delays),
            'actual_minus_predicted_delay_ticks':statistics_of(d['actual_minus_predicted'] for d in delays),
            'prediction_saturation_count':sum(d['saturated'] for d in delays),
            'prefix_covered_estimated_delay_fraction':sum(coverage)/len(coverage) if coverage else None,
            'known_prefix_covered_actual_delay_fraction':sum(actual_coverage)/len(actual_coverage) if actual_coverage else None,
            'guidance_horizon_covered_actual_delay_fraction':sum(guidance_coverage)/len(guidance_coverage) if guidance_coverage else None,
            'maximum_bowl_lift_mm':statistics_of(r['task_progress_diagnostics']['maximum_bowl_lift_mm'] for r in rows),
            'task_progress_by_state':{str(r['init_state_index']):r['task_progress_diagnostics'] for r in rows},
            'received_chunks_with_delay_comparison':len(delays),'steady_chunks_including_drained':len(steady_chunks),
            'delay_pairs':delays,'peak_cuda_bytes':audit['peak_cuda_bytes'],'ros_all_verified':True}
        for key in ('model_inference_wall_ms_steady','full_chunk_worker_roundtrip_ms_steady','sensor_ros_rpc_ms_steady',
                    'action_ros_rpc_ms_steady','action_source_age_ms_steady','camera_sensor_age_ms_steady'):
            group[key+'_episode_mean']=statistics_of(r['metrics_recomputed'][key]['mean'] for r in rows)
            group[key+'_episode_p95']=statistics_of(r['metrics_recomputed'][key]['p95'] for r in rows)
        groups.append(group)
        for name in ('config.json','summary.json','model-audit.json','source-sha256.json'):sources.append(source_record(directory/name))
    evaluated_models=sorted({r['model'] for r in episodes})
    if identities!={(model,variant) for model in evaluated_models for variant in protocol['variants']}:
        raise ValueError('incomplete registered variant grid for requested families')
    pairing=[]
    for model in sorted({r['model'] for r in episodes}):
        for state in states:
            rows=[r for r in episodes if r['model']==model and r['init_state_index']==state]
            sensor={key:len({r['initial_sensor_hashes'].get(key) for r in rows})==1 and rows[0]['initial_sensor_hashes'].get(key) is not None for key in SENSORS}
            physical={key:all(np.array_equal(rows[0]['initial_physical_metrics'][key],r['initial_physical_metrics'][key]) for r in rows[1:]) for key in rows[0]['initial_physical_metrics']}
            pairing.append({'model':model,'state':state,'variants':[r['variant'] for r in rows],
                            'sensor_matches':sensor,'all_sensors_match':all(sensor.values()),'physical_matches':physical,'all_physical_match':all(physical.values())})
    bridge_audit,receipts=audit_bridge(episodes,bridge)
    development_bridge_audit,development_receipts=audit_bridge(development,bridge,sample_scope='development_excluded') if development else (None,[])
    return {'status':'complete','generated_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':protocol,
        'evaluated_models':evaluated_models,'not_included_registered_families':sorted(set(protocol['frozen_families'])-set(evaluated_models)),
        'allow_development_state0_in_summary':allow_development_state0_in_summary,'development_episodes_excluded':development,
        'development_independent_ros_audit':development_bridge_audit,
        'scope':'complete variant grid for evaluated_models; formal states1..5 only; development0 excluded; separate H50/R30 series from original35 trials',
        'metric_scope':'steady excludes bootstrap; inference includes drained requests; source age only executed non-held actions; switches are normalized OSC command deltas, not Cartesian meters or radians; overlap disagreement compares old planned versus new executed command at the same switch tick and excludes switches after hold/exhaustion',
        'groups':groups,'episodes':episodes,'model_audits':audits,'initial_condition_checks':pairing,
        'all_initial_sensor_pairs_match':all(r['all_sensors_match'] for r in pairing),
        'all_initial_physical_pairs_match':all(r['all_physical_match'] for r in pairing),
        'independent_ros_audit':bridge_audit,'source_files':[source_record(protocol_path),*sources],
        'analysis_source_files':[source_record(Path(__file__)),source_record(Path(__file__).with_name('summarize.py')),
                                 source_record(Path(__file__).with_name('run_continuity_series.sh'))],
        'limitations':['One known task and five fixed initial states; descriptive uncertainty is not broad benchmark evidence.',
            'All variants share the new H50/aligned/R30 scheduler; differences versus originalE10 are not RTC-only effects.',
            'Action continuity, observation freshness and task success are separate outcomes.',
            'Full VJP adds generation latency; prefix coverage and actual delay are measured rather than assumed.',
            'Soft real-time desktop simulation, not a hard real-time or physical-hardware guarantee.']},receipts+development_receipts


SENSORS=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def main():
    """Write a fresh final report and matched independent receipt evidence.

    Args: CLI paths. Returns: None. Raises: Incomplete evidence or existing output directory.
    Design: Context: Reports must remain reviewable after additional experiments.
        Concern: Silent overwrites erase an earlier interpretation's evidence.
        Decision: New output directory and deterministic raw-data recomputation.
        Alternatives: Rewrite original results.json rejected.
        Quality Attributes: Auditability. Trade-offs: Explicit new path for reanalysis.
    Evidence: Internal: Registered run directories. External: None.
        Findings: Strict validation precedes publication. Metrics: Final formal count.
        Confidence: Recorded-source consistency.
    Constraints: No experiment execution. Reconsider When: Streaming reports are requested.
    References: summarize_runs. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--protocol',type=Path,default=Path('artifacts/realtime-vla/continuity-protocol.json'))
    parser.add_argument('--bridge',type=Path,default=Path('artifacts/realtime-vla/pi05-ros-system'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-development-state0-in-summary',action='store_true',help='require ordered0..5 in each run, retain0 separately, aggregate formal1..5 only')
    args=parser.parse_args();report,receipts=summarize_runs(args.runs,args.protocol,args.bridge,allow_development_state0_in_summary=args.allow_development_state0_in_summary)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'matched-bridge-receipts.json').write_text(json.dumps(receipts,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','episodes':len(report['episodes']),'groups':[{k:g[k] for k in ('model','variant','n','successes','pooled_hold_fraction')} for g in report['groups']],
                      'independent_ros_chunks':report['independent_ros_audit']['task_chunks']}))


if __name__=='__main__':main()
