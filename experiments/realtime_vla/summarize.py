"""Aggregate paired frozen-VLA evidence without mixing pilot or bootstrap measurements.

Args: Completed formal run directories, expected initial states and report destination.
Returns: Downloadable JSON/CSV and fixed PNG/SVG comparison plots.
Raises: ValueError for missing pairs, incompatible protocols or inconsistent tick counts.
Design:
    Context: Native success, execution latency and sensor age answer different questions.
    Concern: Pooled pilot data, startup age and paused footage can misstate real-time behavior.
    Decision: Recompute metrics from immutable episode/tick/chunk evidence and label each scope.
    Alternatives: Comparing only summary means hides individual paired outcomes.
    Quality Attributes: Integrity, reproducibility. Trade-offs: Strict complete-input requirement.
Evidence:
    Internal: evaluate.py records physical outcomes and both worker/model timing.
    External: Wilson interval for a small binomial sample.
    Findings: CUDA-synchronized model wall time is not pure GPU-kernel execution time.
    Metrics: Official success, goalXY, holds, RTF, controlHz, sensor/action age and ROS verification.
    Confidence: Limited to the recorded task and five initial states.
Constraints: No source edits, pilot blending, fabricated missing outcomes or causal model rankings.
Reconsider When: Evaluation protocol or result schema changes.
References: docs/architecture/realtime-vla.md; make_demo.py.
ADR: None.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUNS=[ROOT/'artifacts/realtime-vla/formal-smolvla-e10',ROOT/'artifacts/realtime-vla/formal-pi0fast-e10']
COLORS={'smolvla':'#087f8c','pi0fast':'#d46928'}
NAMES={'smolvla':'SmolVLA','pi0fast':'π0-FAST'}


def statistics_of(values):
    """Summarize finite samples without manufacturing a value for empty groups.

    Args: values: optional numeric observations. Returns: count/mean/p50/p95/min/max.
    Raises: None for absent values; nonfinite samples are excluded and counted.
    Design: Context: Some failed runs have no applied chunks.
        Concern: NaN JSON or implicit zero latency. Decision: Use null for empty groups.
        Alternatives: Replacing missing values with zero. Quality Attributes: Integrity.
        Trade-offs: Consumers must handle null.
    Evidence: Internal: Timing is per recorded request.
        External: NumPy percentile. Findings: Count accompanies every statistic.
        Metrics: finite_count/excluded_count. Confidence: Deterministic arithmetic.
    Constraints: Booleans are not latency samples.
    Reconsider When: Weighted/censored estimators are needed.
    References: read_episode. ADR: None.
    """
    source=list(values);numbers=[float(v) for v in source if type(v) in (int,float) and math.isfinite(v)]
    return {'count':len(numbers),'excluded_count':len(source)-len(numbers),'mean':float(np.mean(numbers)) if numbers else None,'p50':float(np.percentile(numbers,50)) if numbers else None,'p95':float(np.percentile(numbers,95)) if numbers else None,'min':min(numbers) if numbers else None,'max':max(numbers) if numbers else None}


def wilson(successes,total):
    """Show sample uncertainty for native success without implying broad benchmark precision.

    Args: successes/total: observed binomial counts. Returns:95percent Wilson interval.
    Raises: ValueError for invalid counts.
    Design: Context: Five paired initial states are a small sample.
        Concern: k/5 should not imply a precise population rate.
        Decision: Display Wilson bounds and counts together.
        Alternatives: Unqualified percent-only ranking. Quality Attributes: Clarity.
        Trade-offs: Descriptive interval does not make this a broad randomized benchmark.
    Evidence: Internal: Fixed initial-state sample.
        External: Wilson score interval, z1.95996398454.
        Findings: Even5/5 has a wide interval. Metrics: Lower/upper bounds.
        Confidence: Formula only; task generalization remains limited.
    Constraints: Same denominator as planned completed sample.
    Reconsider When: Task-level random sampling is added.
    References: aggregate. ADR: None.
    """
    if not 0<=successes<=total or total<1:raise ValueError('invalid success counts')
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-half),min(1.,center+half)]


def read_episode(directory):
    """Recompute scoped latency and age metrics from a completed physical episode.

    Args: directory: result/ticks/chunks/initial files. Returns: Compact episode record.
    Raises: ValueError for disagreement between result and tick evidence.
    Design:
        Context: Bootstrap inference occurs before task time; late-drained chunks are unapplied.
        Concern: Summary averages can mix these scopes or conflate camera and policy-source age.
        Decision: Identify bootstrap request IDs, retain all/steady/bootstrap separately and count applied chunks.
        Alternatives: Dropping slow terminal requests biases timing downward.
        Quality Attributes: Auditability. Trade-offs: Several explicitly named metric scopes.
    Evidence:
        Internal: Chunk flags/request IDs and per-action timestamps.
        External: None. Findings: Steady generation includes requests finished after episode end,
            while applied-only generation is reported separately.
        Metrics: Model wall, full worker roundtrip, source age, camera age, control periods.
        Confidence: Recorded evidence only.
    Constraints: Controller tracking error is distinct from object-goalXY error.
    Reconsider When: More than one inference request may be in flight.
    References: evaluate.py, worker.py.
    ADR: None.
    """
    directory=Path(directory);result=json.loads((directory/'result.json').read_text())
    ticks=[json.loads(line) for line in (directory/'ticks.jsonl').read_text().splitlines() if line]
    chunks=json.loads((directory/'chunks.json').read_text());initial=json.loads((directory/'initial.json').read_text())
    if len(ticks)!=result['ticks'] or [t['tick'] for t in ticks]!=list(range(len(ticks))):raise ValueError(f'inconsistent ticks: {directory}')
    if sum(bool(t['held']) for t in ticks)!=result['hold_ticks']:raise ValueError(f'inconsistent hold count: {directory}')
    bootstrap=[c for c in chunks if c.get('bootstrap') and c.get('kind')=='chunk']
    steady=[c for c in chunks if not c.get('bootstrap') and c.get('kind')=='chunk' and 'finished_time' in c and 'started_time' in c]
    applied=[c for c in steady if c.get('applied') and not c.get('completed_after_episode')]
    bootstrap_ids={c['request_id'] for c in bootstrap};bootstrap_ticks=[t['tick'] for t in ticks if t.get('request_id') in bootstrap_ids]
    end_bootstrap=max(bootstrap_ticks,default=-1)
    active=[t for t in ticks if not t['held'] and t.get('observation_age_s') is not None]
    steady_active=[t for t in active if t.get('request_id') not in bootstrap_ids]
    camera_steady=[t for t in ticks if t['tick']>end_bootstrap]
    interval=statistics_of([t['tick_interval_s'] for t in ticks if t.get('tick_interval_s') is not None])
    generated=[c for c in chunks if c.get('kind')=='chunk']
    transport=[]
    for chunk in generated:
        for side in ('sensor_transport','action_transport'):
            record=chunk.get('metadata',{}).get(side)
            if record is not None:transport.append({'side':side,'verified':record.get('verified') is True and record.get('source_sha256')==record.get('received_sha256')==record.get('peer_received_sha256')})
    row=dict(result,source_directory=str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),initial_goal_xy_error_m=initial.get('metrics',{}).get('goal_xy_error_m'),bootstrap_request_ids=sorted(bootstrap_ids),bootstrap_action_ticks=len(bootstrap_ticks),steady_action_ticks=len(steady_active),steady_chunk_count=len(steady),steady_applied_chunk_count=len(applied),late_unapplied_chunk_count=sum(c.get('completed_after_episode',False) for c in steady))
    row['metrics_recomputed']={
        'model_inference_wall_ms_steady':statistics_of(c.get('metadata',{}).get('model_inference_ms') for c in steady),
        'full_chunk_worker_roundtrip_ms_steady':statistics_of((c['finished_time']-c['started_time'])*1000 for c in steady),
        'full_chunk_worker_roundtrip_ms_steady_applied':statistics_of((c['finished_time']-c['started_time'])*1000 for c in applied),
        'model_inference_wall_ms_bootstrap':statistics_of(c.get('metadata',{}).get('model_inference_ms') for c in bootstrap),
        'full_chunk_worker_roundtrip_ms_bootstrap':statistics_of((c['finished_time']-c['started_time'])*1000 for c in bootstrap),
        'sensor_ros_rpc_ms_steady':statistics_of(c.get('metadata',{}).get('sensor_transport',{}).get('rpc_wall_seconds',0)*1000 for c in steady if c.get('metadata',{}).get('sensor_transport') is not None),
        'action_ros_rpc_ms_steady':statistics_of(c.get('metadata',{}).get('action_transport',{}).get('rpc_wall_seconds',0)*1000 for c in steady if c.get('metadata',{}).get('action_transport') is not None),
        'action_source_age_ms_all':statistics_of(t['observation_age_s']*1000 for t in active),
        'action_source_age_ms_steady':statistics_of(t['observation_age_s']*1000 for t in steady_active),
        'action_source_age_ms_bootstrap':statistics_of(t['observation_age_s']*1000 for t in active if t.get('request_id') in bootstrap_ids),
        'camera_sensor_age_ms_all':statistics_of(t.get('camera_age_s')*1000 for t in ticks if t.get('camera_age_s') is not None),
        'camera_sensor_age_ms_steady':statistics_of(t.get('camera_age_s')*1000 for t in camera_steady if t.get('camera_age_s') is not None),
        'physical_step_ms':statistics_of(t['step_wall_s']*1000 for t in ticks),
        'control_interval_ms':statistics_of(t['tick_interval_s']*1000 for t in ticks if t.get('tick_interval_s') is not None),
        'eef_controller_tracking_error_mm':statistics_of(t.get('metrics',{}).get('eef_tracking_error_m')*1000 for t in ticks if t.get('metrics',{}).get('eef_tracking_error_m') is not None)}
    row['effective_control_hz']=1/interval['mean'] if interval['mean'] else None
    row['effective_received_camera_hz']=result.get('sensor_frames_received',0)/result['wall_seconds'] if result.get('camera_mode')=='asynchronous_clone' and result['wall_seconds']>0 else None
    row['ros_exchange_count']=len(transport);row['ros_expected_exchange_count']=2*len(generated)
    row['ros_all_verified']=bool(transport) and len(transport)==2*len(generated) and all(t['verified'] for t in transport)
    row['initial_state_file_sha256']=initial.get('environment',{}).get('initial_states_file_sha256')
    row['initial_sensor_hashes']=initial.get('sensor_hashes',{})
    row['initial_physical_metrics']={key:initial.get('metrics',{}).get(key) for key in ('bowl_position','plate_position','actuator_control','goal_xy_error_m')}
    return row


def initial_condition_checks(episodes):
    """Separate physical-state pairing from cached-versus-regenerated sensor equality.

    Args: episodes: compact records containing initial sensor hashes/physical metrics.
    Returns: Three distinct equality audits with per-key comparisons.
    Raises: None for absent pairs; they remain explicitly untested.
    Design:
        Context: Sync env.step uses cached observables; async cameras force state regeneration.
        Concern: Clone parity must not be overstated as lockstep-versus-realtime sensor identity.
        Decision: Test model pairing within mode and physical/sensor pairing across modes separately.
        Alternatives: Treating one shared initial-state file as proof of identical model inputs.
        Quality Attributes: Causal clarity. Trade-offs: Sensor sampling phase remains a system difference.
    Evidence:
        Internal: initial.json sensor_hashes and simulator-truth bowl/plate/control measurements.
        External: robosuite cached observables versus LIBERO regenerate_obs_from_state(force).
        Findings: Physical truth can match while cached camera/proprio samples differ.
        Metrics: Per-sensor hashes and exact physical-field equality.
        Confidence: Direct file comparison; cause requires separate post-evaluation diagnosis.
    Constraints: Never equate regenerated-clone parity with baseline raw-step parity.
    Reconsider When: A matched sensor-phase baseline is preregistered.
    References: environment-smoke/parallel-camera-result.json. ADR: None.
    """
    sensors=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')
    across_models=[];across_modes=[];physical=[]
    states=sorted({r['init_state_index'] for r in episodes})
    for mode in ('lockstep','realtime'):
        for state in states:
            rows={r['model']:r for r in episodes if r['mode']==mode and r['init_state_index']==state}
            if set(rows)!= {'smolvla','pi0fast'}:continue
            a,b=rows['smolvla'],rows['pi0fast'];matched={key:a['initial_sensor_hashes'].get(key) is not None and a['initial_sensor_hashes'].get(key)==b['initial_sensor_hashes'].get(key) for key in sensors}
            across_models.append({'mode':mode,'init_state_index':state,'all_match':all(matched.values()),'sensor_matches':matched})
    for model in ('smolvla','pi0fast'):
        for state in states:
            rows={r['mode']:r for r in episodes if r['model']==model and r['init_state_index']==state}
            if set(rows)!= {'lockstep','realtime'}:continue
            a,b=rows['lockstep'],rows['realtime'];matched={key:a['initial_sensor_hashes'].get(key) is not None and a['initial_sensor_hashes'].get(key)==b['initial_sensor_hashes'].get(key) for key in sensors}
            across_modes.append({'model':model,'init_state_index':state,'all_match':all(matched.values()),'sensor_matches':matched})
            fields={};deltas={}
            for key,left in a['initial_physical_metrics'].items():
                right=b['initial_physical_metrics'].get(key)
                fields[key]=left is not None and right is not None and bool(np.array_equal(left,right))
                if left is not None and right is not None:
                    left,right=np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
                    deltas[key]=float(np.max(np.abs(left-right))) if left.shape==right.shape else None
            physical.append({'model':model,'init_state_index':state,'all_match':all(fields.values()),'field_matches':fields,'max_abs_deltas':deltas})
    return {'initial_sensor_match_across_models_by_mode':{'all_match':all(r['all_match'] for r in across_models) if across_models else None,'comparisons':across_models},'physical_initial_metrics_same_across_modes':{'all_match':all(r['all_match'] for r in physical) if physical else None,'compared_fields':['bowl_position','plate_position','actuator_control','goal_xy_error_m'],'comparisons':physical},'initial_sensor_match_across_modes':{'all_match':all(r['all_match'] for r in across_modes) if across_modes else None,'comparisons':across_modes},'scope_note':'Clone byte parity was tested after regenerating the same snapshot in both scenes. Sync cached-step sensors and async regenerated sensors are compared separately here; their equality is not assumed.'}


def aggregate(run_directories,states,allow_partial=False,expected_prefetch=0,expected_modes=('lockstep','realtime'),expected_models=('smolvla','pi0fast'),expected_execute=10):
    """Require matched formal settings and retain each initial-state outcome.

    Args: run_directories: two model roots; states: planned indices;
        allow_partial: retain missing pairs; expected_prefetch/modes/models/execute: declared grid.
    Returns: Downloadable report mapping. Raises: ValueError on invalid protocol/completeness.
    Design:
        Context: Pilot init0 selected settings; formal init1..5 tests that fixed protocol.
        Concern: Unmatched conditions or selective successes would bias conclusions.
        Decision: Validate the declared model/mode/state grid; keep exploratory grids separate.
        Alternatives: Pooling every directory under artifacts. Quality Attributes: Fairness.
        Trade-offs: Final report waits for complete formal data.
    Evidence:
        Internal: Config, summary frozen-weight audit, exact initial state file hashes.
        External: Wilson intervals. Findings: Task success uses the native predicate.
        Metrics: Success counts, finalXY, holds, RTF and per-episode scoped timing.
        Confidence: Small fixed sample, no broad performance claim.
    Constraints: Canonical20Hz/ROStrue; caller declares executed prefix, prefetch and grid.
    Reconsider When: New conditions are preregistered.
    References: docs/architecture/realtime-vla.md.
    ADR: None.
    """
    episodes=[];audits={};sources=[];issues=[]
    for run in (Path(path).resolve() for path in run_directories):
        summary_path=run/'summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text());audit=summary.get('audit',{})
            if summary.get('status')!='complete':issues.append(f'incomplete summary: {run.name}')
            model=summary.get('results',[{}])[0].get('model',run.name)
            audits[model]=audit
            if not audit.get('weights_unchanged') or audit.get('local_training') is not False:issues.append(f'frozen-weight audit missing: {run.name}')
        else:issues.append(f'missing summary: {run.name}')
        for name in ('summary.json','config.json','model-audit.json'):
            source=run/name
            if source.exists():sources.append({'path':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bytes':source.stat().st_size})
        for path in sorted(run.glob('*-init*/result.json')):
            result=json.loads(path.read_text())
            if result.get('init_state_index') not in states:continue
            if result.get('model') not in expected_models or result.get('mode') not in expected_modes:raise ValueError(f'unknown episode identity: {path}')
            if (result.get('control_hz'),result.get('execute_steps'),result.get('prefetch_steps'),result.get('ros_transport'))!=(20,expected_execute,expected_prefetch,True):raise ValueError(f'formal protocol differs: {path}')
            if result.get('merge')!='fifo':raise ValueError(f'formal queue merge differs: {path}')
            if result['mode']=='realtime' and (result.get('camera_mode')!='asynchronous_clone' or result.get('camera_hz')!=10):raise ValueError(f'camera protocol differs: {path}')
            episodes.append(read_episode(path.parent))
            for name in ('result.json','ticks.jsonl','chunks.json','initial.json'):
                source=path.parent/name;sources.append({'path':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bytes':source.stat().st_size})
    expected={(m,mode,i) for m in expected_models for mode in expected_modes for i in states}
    found=[(r['model'],r['mode'],r['init_state_index']) for r in episodes]
    if len(found)!=len(set(found)):raise ValueError('duplicate formal episode identity')
    missing=sorted(expected-set(found))
    if missing:issues.append(f'missing episodes: {missing}')
    if issues and not allow_partial:raise ValueError('; '.join(issues))
    if len({r['initial_state_file_sha256'] for r in episodes})>1:raise ValueError('models used different initial-state files')
    if len({r['wall_budget_s'] for r in episodes})>1 or len({r['seed'] for r in episodes})>1:raise ValueError('models used different wall budgets or RNG seeds')
    groups=[]
    for model in expected_models:
        for mode in expected_modes:
            rows=[r for r in episodes if r['model']==model and r['mode']==mode]
            if not rows:continue
            successes=sum(bool(r['success']) for r in rows);n=len(rows)
            group={'model':model,'mode':mode,'n':n,'successes':successes,'success_rate':successes/n,'success_wilson95':wilson(successes,n),'errors':sum(r['status']!='completed' for r in rows),'states':[r['init_state_index'] for r in rows],'final_goal_xy_mm':statistics_of(r['goal_xy_error_m']*1000 for r in rows),'hold_fraction':statistics_of(r['hold_fraction'] for r in rows),'pooled_hold_fraction':sum(r['hold_ticks'] for r in rows)/sum(r['ticks'] for r in rows),'effective_control_hz':statistics_of(r['effective_control_hz'] for r in rows),'realtime_factor':statistics_of(r['realtime_factor'] for r in rows),'ros_all_verified':all(r['ros_all_verified'] for r in rows)}
            group['effective_received_camera_hz']=statistics_of(r['effective_received_camera_hz'] for r in rows)
            for key in ('model_inference_wall_ms_steady','full_chunk_worker_roundtrip_ms_steady','action_source_age_ms_steady','action_source_age_ms_bootstrap','camera_sensor_age_ms_steady'):
                group[key+'_episode_mean']=statistics_of(r['metrics_recomputed'][key]['mean'] for r in rows)
                group[key+'_episode_p95']=statistics_of(r['metrics_recomputed'][key]['p95'] for r in rows)
            groups.append(group)
    pairs=[]
    for model in expected_models:
        for index in states:
            rows={r['mode']:r for r in episodes if r['model']==model and r['init_state_index']==index}
            if set(rows)=={'lockstep','realtime'}:
                a,b=rows['lockstep'],rows['realtime'];pairs.append({'model':model,'init_state_index':index,'lockstep_success':a['success'],'realtime_success':b['success'],'lockstep_goal_xy_mm':a['goal_xy_error_m']*1000,'realtime_goal_xy_mm':b['goal_xy_error_m']*1000,'goal_xy_delta_rt_minus_lockstep_mm':(b['goal_xy_error_m']-a['goal_xy_error_m'])*1000,'same_initial_state_file':a['initial_state_file_sha256']==b['initial_state_file_sha256']})
    return {'status':'partial' if issues else 'complete','generated_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':{'task':'LIBERO spatial task0: black bowl between plate/ramekin onto plate','initial_states':list(states),'models':list(expected_models),'modes':list(expected_modes),'wall_budget_s':episodes[0]['wall_budget_s'] if episodes else None,'seed':episodes[0]['seed'] if episodes else None,'pilot_state_excluded':0,'control_hz':20,'camera_hz_realtime':10,'camera_rate_semantics':'maximum rendering rate; actual received cameraHz is measured separately','execute_steps':expected_execute,'prefetch_steps':expected_prefetch,'merge':'fifo','ros_transport':True,'training':False,'baseline':'lockstep: inference and synchronous rendering pause physical progress' if 'lockstep' in expected_modes else None,'realtime':'separate physics/camera/model processes; zero relative pose plus retained gripper during queue starvation','latency_label':'CUDA-synchronized model inference wall time; worker roundtrip spans sensor ROS, inference and action ROS, excluding parent IPC/poll delay; action-source age spans capture to command application','steady_definition':'exclude bootstrap request IDs; chunk timing includes late-drained requests but applied subset is separately available','camera_age_steady_definition':'all physical ticks after last bootstrap action; includes hold ticks'},'issues':issues,'groups':groups,'initial_condition_checks':initial_condition_checks(episodes),'paired_lockstep_realtime':pairs,'episodes':episodes,'model_audits':audits,'source_files':sources,'limitations':['Five fixed initial states from one known task; Wilson bounds are descriptive, not a broad VLA benchmark.','Static policy quality and runtime behavior differ; task and action-generation horizons differ across checkpoints.','Native success is primary; goalXY alone does not prove grasp/place completion.','Model-time counters include synchronized GPU work and Python overhead; they are not CUDA kernel-event timings.','Lockstep and real-time also differ in sensor scheduling; results describe the whole declared system.','Bootstrap, software setup and two warmup calls are outside task time; their latency is recorded separately.','No hard real-time or real-hardware safety claim.']}


def attach_ablation(primary,run_directories,states,allow_partial=False):
    """Keep an exploratory scheduler change separate from the preregistered primary counts.

    Args: primary: complete20episode report; run_directories: prefetch5runs;
        states: same fixed indices; allow_partial: explicit preview only.
    Returns: Primary mapping with scheduler_ablation attached.
    Raises: ValueError for incompatible/missing ablation data or checkpoint changes.
    Design:
        Context: Earlier requests may reduce hold ticks but increase action-source age.
        Concern: Post-primary tuning must not replace the original outcomes.
        Decision: Validate a separate realtime-only grid, pair within model/state and label exploratory.
        Alternatives: Pooling ten extra trials into primary success denominators.
        Quality Attributes: Transparency. Trade-offs: Reused states do not form a new holdout test.
    Evidence:
        Internal: Both conditions retain20Hz/E10/camera10Hz/ROS and frozen checkpoint revisions.
        External: None. Findings: This changes request timing, not model-conditioned RTC.
        Metrics: Paired success transitions, goalXY/hold/age and control frequency.
        Confidence: Exploratory repeated-state experiment only.
    Constraints: Primary episodes/groups/counts remain unchanged.
    Reconsider When: A separate held-out scheduler study is preregistered.
    References: aggregate, create_plots. ADR: None.
    """
    extra=aggregate(run_directories,states,allow_partial,expected_prefetch=5,expected_modes=('realtime',))
    for model,audit in extra['model_audits'].items():
        base=primary['model_audits'].get(model,{})
        if base.get('weights_before_sha256') and base.get('weights_before_sha256')!=audit.get('weights_before_sha256'):raise ValueError('ablation changed checkpoint weights')
    pairs=[]
    for after in extra['episodes']:
        matches=[r for r in primary['episodes'] if r['model']==after['model'] and r['mode']=='realtime' and r['init_state_index']==after['init_state_index']]
        if len(matches)!=1:
            if allow_partial:continue
            raise ValueError('ablation is missing its primary pair')
        before=matches[0]
        if (before['seed'],before['wall_budget_s'],before['initial_state_file_sha256'])!=(after['seed'],after['wall_budget_s'],after['initial_state_file_sha256']):raise ValueError('ablation changed seed/budget/initial states')
        pairs.append({'model':after['model'],'init_state_index':after['init_state_index'],'prefetch0_success':before['success'],'prefetch5_success':after['success'],'success_transition':f"{int(before['success'])}->{int(after['success'])}",'prefetch0_goal_xy_mm':before['goal_xy_error_m']*1000,'prefetch5_goal_xy_mm':after['goal_xy_error_m']*1000,'goal_xy_delta_p5_minus_p0_mm':(after['goal_xy_error_m']-before['goal_xy_error_m'])*1000,'prefetch0_hold_fraction':before['hold_fraction'],'prefetch5_hold_fraction':after['hold_fraction'],'hold_delta_p5_minus_p0':after['hold_fraction']-before['hold_fraction'],'initial_sensor_hashes_identical':before['initial_sensor_hashes']==after['initial_sensor_hashes'],'initial_physical_metrics_identical':before['initial_physical_metrics']==after['initial_physical_metrics']})
    changes=[]
    for model in ('smolvla','pi0fast'):
        rows=[p for p in pairs if p['model']==model]
        changes.append({'model':model,'paired_n':len(rows),'success_gained':sum(not p['prefetch0_success'] and p['prefetch5_success'] for p in rows),'success_lost':sum(p['prefetch0_success'] and not p['prefetch5_success'] for p in rows),'success_both':sum(p['prefetch0_success'] and p['prefetch5_success'] for p in rows),'failure_both':sum(not p['prefetch0_success'] and not p['prefetch5_success'] for p in rows)})
    primary['scheduler_ablation']={'status':extra['status'],'label':'Exploratory shared scheduler prefetch0→5; RTC disabled','protocol':extra['protocol'],'groups':extra['groups'],'episodes':extra['episodes'],'initial_condition_checks':extra['initial_condition_checks'],'paired_prefetch0_prefetch5':pairs,'paired_change_summary':changes,'model_audits':extra['model_audits'],'source_files':extra['source_files'],'issues':extra['issues'],'limitations':['Primary20trial counts remain unchanged.','Same five initial states were reused after observing primary outcomes; this is exploratory and not a fresh holdout.','Only the common queue request threshold changed; model-conditioned RTC is disabled.','Terminal errors are measured at each episode end, which may occur at different times after success.']}
    if extra['status']!='complete':primary['status']='partial'
    return primary


def attach_native_horizon(primary,run_directory,states,allow_partial=False):
    """Describe SmolVLA's full generated horizon as a separate exploratory deployment condition.

    Args: primary: E10report; run_directory: Smol E50/p0 realtime run;
        states: same fixed indices; allow_partial: explicit preview only.
    Returns: Report with native_horizon_smol and unchanged primary counts.
    Raises: ValueError for changed weights/protocol or missing primary pairs.
    Design:
        Context: E10 uses only ten of SmolVLA's fifty generated actions.
        Concern: Better compute amortization also extends open-loop time and source age.
        Decision: Keep E50/p0 within-model exploration separate from the matched E10 comparison.
        Alternatives: Ranking E50Smol against E10pi as though the execution conditions matched.
        Quality Attributes: Fairness, observability. Trade-offs: Same states are reused after primary.
    Evidence:
        Internal: Same checkpoint, seed, task,20Hz/camera10Hz/ROS and14second wall budget.
        External: Smol checkpoint native generation horizon50; pi native horizon10.
        Findings: Measure all five outcomes, including failures, without replacing E10 scores.
        Metrics: Paired success, holds, terminalXY and applied-command source age.
        Confidence: Exploratory deployment test only.
    Constraints: No additional pi trial implied; its E10native reference remains in primary.
    Reconsider When: A matched action-generation-horizon study is designed.
    References: aggregate, make_demo.py. ADR: None.
    """
    extra=aggregate([run_directory],states,allow_partial,expected_prefetch=0,expected_modes=('realtime',),expected_models=('smolvla',),expected_execute=50)
    if any(r['native_generation_horizon']!=50 for r in extra['episodes']):raise ValueError('native Smol run did not generate50actions')
    base_audit=primary['model_audits'].get('smolvla',{});audit=extra['model_audits'].get('smolvla',{})
    if base_audit.get('weights_before_sha256') and audit.get('weights_before_sha256') and base_audit['weights_before_sha256']!=audit['weights_before_sha256']:raise ValueError('native horizon run changed checkpoint weights')
    pairs=[]
    for after in extra['episodes']:
        matches=[r for r in primary['episodes'] if r['model']=='smolvla' and r['mode']=='realtime' and r['init_state_index']==after['init_state_index']]
        if len(matches)!=1:
            if allow_partial:continue
            raise ValueError('native horizon run is missing its primary pair')
        before=matches[0]
        if (before['seed'],before['wall_budget_s'],before['initial_state_file_sha256'])!=(after['seed'],after['wall_budget_s'],after['initial_state_file_sha256']):raise ValueError('native horizon changed seed/budget/initial state file')
        pairs.append({'model':'smolvla','init_state_index':after['init_state_index'],'execute10_success':before['success'],'execute50_success':after['success'],'success_transition':f"{int(before['success'])}->{int(after['success'])}",'execute10_goal_xy_mm':before['goal_xy_error_m']*1000,'execute50_goal_xy_mm':after['goal_xy_error_m']*1000,'goal_xy_delta_e50_minus_e10_mm':(after['goal_xy_error_m']-before['goal_xy_error_m'])*1000,'execute10_hold_fraction':before['hold_fraction'],'execute50_hold_fraction':after['hold_fraction'],'hold_delta_e50_minus_e10':after['hold_fraction']-before['hold_fraction'],'execute10_source_age_steady_ms':before['metrics_recomputed']['action_source_age_ms_steady']['mean'],'execute50_source_age_steady_ms':after['metrics_recomputed']['action_source_age_ms_steady']['mean'],'initial_sensor_hashes_identical':before['initial_sensor_hashes']==after['initial_sensor_hashes'],'initial_physical_metrics_identical':before['initial_physical_metrics']==after['initial_physical_metrics']})
    changes={'paired_n':len(pairs),'success_gained':sum(not p['execute10_success'] and p['execute50_success'] for p in pairs),'success_lost':sum(p['execute10_success'] and not p['execute50_success'] for p in pairs),'success_both':sum(p['execute10_success'] and p['execute50_success'] for p in pairs),'failure_both':sum(not p['execute10_success'] and not p['execute50_success'] for p in pairs)}
    primary['native_horizon_smol']={'status':extra['status'],'label':'Exploratory Smol native horizon E50/prefetch0; separate from matched E10 comparison','protocol':extra['protocol'],'groups':extra['groups'],'episodes':extra['episodes'],'paired_execute10_execute50':pairs,'paired_change_summary':changes,'model_audits':extra['model_audits'],'source_files':extra['source_files'],'issues':extra['issues'],'limitations':['Primary20trial counts and prefetch ablation counts remain unchanged.','The execution horizon changes from0.5s to2.5s at20Hz; this changes feedback rate and action-source age.','Same five initial states were reused after primary; exploratory rather than a new holdout.','pi native E10/p0 is already a primary reference; unequal horizons do not support a like-for-like model ranking.']}
    if extra['status']!='complete':primary['status']='partial'
    return primary


def create_plots(report,output):
    """Save transparent small-sample outcome and timing plots from fixed formal evidence.

    Args: report: aggregate mapping; output: report directory. Returns: Plot filenames.
    Raises: Rendering/IO errors; missing metric values remain absent.
    Design: Context: Readers must see each paired outcome and timing scope.
        Concern: Mean-only bars hide variance and controller/goal error distinctions.
        Decision: Individual episode markers, counts and explicit units/reference lines.
        Alternatives: Decorative dashboards without evidence points.
        Quality Attributes: Clarity. Trade-offs: Multiple focused plots.
    Evidence: Internal: Compact report episodes and immutable tick logs.
        External: Matplotlib. Findings: Different time axes are separately labelled.
        Metrics: Success, goalXY, latency, holds, frequency and age.
        Confidence: Visual review required after rendering.
    Constraints: Do not crop failure errors or call source age tracking accuracy.
    Reconsider When: More tasks/conditions require faceting.
    References: aggregate. ADR: None.
    """
    output=Path(output)
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf7','axes.facecolor':'#ffffff','savefig.facecolor':'#fafaf7'})
    output=Path(output);plots=[];episodes=report['episodes'];groups=report['groups']
    preview='PARTIAL PREVIEW • ' if report['status']!='complete' else ''
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j,g in enumerate(groups):
        c=COLORS[g['model']];alpha=.5 if g['mode']=='lockstep' else 1.;low,high=g['success_wilson95'];rate=g['success_rate']
        axes[0].bar(j,rate,color=c,alpha=alpha,width=.65);axes[0].errorbar(j,rate,yerr=[[max(0,rate-low)],[max(0,high-rate)]],color='#333333',capsize=4,fmt='none');axes[0].text(j,min(1.08,rate+.09),f"{g['successes']}/{g['n']}",ha='center',weight='bold')
        rows=[r for r in episodes if r['model']==g['model'] and r['mode']==g['mode']]
        for r in rows:axes[1].scatter(j+(r['init_state_index']-3)*.045,r['goal_xy_error_m']*1000,color=c,marker='o' if r['success'] else'x',s=62,alpha=alpha)
        axes[1].plot([j-.25,j+.25],[g['final_goal_xy_mm']['p50']]*2,color='#202020',lw=2)
    labels=[NAMES[g['model']]+'\n'+g['mode'] for g in groups]
    for ax in axes:ax.set_xticks(range(len(groups)),labels);ax.grid(axis='y',alpha=.15)
    axes[0].set(ylim=(0,1.18),ylabel='Native task success rate',title='Fixed initial states • count + Wilson95% interval')
    axes[1].set(ylabel='Final bowl–plate XY distance [mm]',title='Each episode • o success / x failure • black median')
    fig.suptitle(preview+'Frozen policies • E10 /20Hz /ROS2 • no retraining',weight='bold')
    for ext in ('png','svg'):fig.savefig(output/f'outcomes.{ext}',dpi=170);plots.append(f'outcomes.{ext}')
    plt.close(fig)
    rt=[r for r in episodes if r['mode']=='realtime'];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,model in enumerate(('smolvla','pi0fast')):
        rows=[r for r in rt if r['model']==model];c=COLORS[model]
        for offset,key,label in [(-.17,'model_inference_wall_ms_steady','Model wall'),(.17,'full_chunk_worker_roundtrip_ms_steady','Chunk + ROS')]:
            vals=[r['metrics_recomputed'][key]['mean'] for r in rows if r['metrics_recomputed'][key]['mean'] is not None]
            if vals:axes[0,0].bar(j+offset,np.mean(vals),width=.3,color=c,alpha=.5 if offset<0 else 1.,label=label if j==0 else None)
            for k,value in enumerate(vals):axes[0,0].scatter(j+offset+(k-2)*.015,value,color='#202020',s=14)
        for r in rows:
            x=j+(r['init_state_index']-3)*.045
            axes[0,1].scatter(x,r['hold_fraction']*100,color=c,s=55)
            axes[1,0].scatter(x,r['effective_control_hz'],color=c,s=55)
        for offset,key in [(-.17,'camera_sensor_age_ms_steady'),(.17,'action_source_age_ms_steady')]:
            vals=[r['metrics_recomputed'][key]['mean'] for r in rows if r['metrics_recomputed'][key]['mean'] is not None]
            if vals:axes[1,1].bar(j+offset,np.mean(vals),width=.3,color=c,alpha=.5 if offset<0 else 1.)
            for k,value in enumerate(vals):axes[1,1].scatter(j+offset+(k-2)*.015,value,color='#202020',s=14)
    for ax in axes.flat:ax.set_xticks([0,1],[NAMES['smolvla'],NAMES['pi0fast']]);ax.grid(axis='y',alpha=.15)
    axes[0,0].set(ylabel='Steady per-episode mean [ms]',title='Model wall versus complete chunk roundtrip')
    if axes[0,0].get_legend_handles_labels()[0]:axes[0,0].legend(frameon=False)
    axes[0,1].set(ylabel='Hold ticks [%]',title='Queue empty → zero pose delta / retain gripper',ylim=(0,100))
    axes[1,0].set(ylabel='Actual control rate [Hz]',title='Physical loop frequency');axes[1,0].axhline(20,color='#555555',ls='--',lw=1,label='20Hz target')
    observed_hz=[r['effective_control_hz'] for r in rt if r.get('effective_control_hz') is not None]
    axes[1,0].set_ylim(min([19.5]+[v-.2 for v in observed_hz]),max([20.5]+[v+.2 for v in observed_hz]))
    axes[1,0].ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[1,1].set(ylabel='Mean age [ms]',title='Light: camera age / dark: applied policy source age')
    fig.suptitle(preview+'Real-time system • startup excluded • points = initial states',weight='bold')
    for ext in ('png','svg'):fig.savefig(output/f'timing.{ext}',dpi=170);plots.append(f'timing.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,model in enumerate(('smolvla','pi0fast')):
        for col,mode in enumerate(('lockstep','realtime')):
            ax=axes[row,col]
            for episode in episodes:
                if episode['model']!=model or episode['mode']!=mode:continue
                tickfile=ROOT/episode['source_directory']/'ticks.jsonl';ticks=[json.loads(line) for line in tickfile.read_text().splitlines() if line]
                x=[(t['tick']+1)/20 if mode=='lockstep' else t['wall_time_since_start']+t['step_wall_s'] for t in ticks]
                y=[t['metrics'].get('goal_xy_error_m',0)*1000 for t in ticks]
                ax.plot(x,y,label=f"init{episode['init_state_index']} {'✓' if episode['success'] else '×'}",lw=1.5)
            ax.set(title=NAMES[model]+' / '+mode,ylabel='Bowl–plate XY [mm]',xlabel='Simulated seconds (paused reference)' if mode=='lockstep' else'Wall seconds (physical loop continues)');ax.grid(alpha=.15)
            if ax.get_legend_handles_labels()[0]:ax.legend(frameon=False,ncol=2,fontsize=8)
    if preview:fig.suptitle(preview+'incomplete formal dataset',weight='bold')
    for ext in ('png','svg'):fig.savefig(output/f'goal-trajectories.{ext}',dpi=170);plots.append(f'goal-trajectories.{ext}')
    plt.close(fig)
    ablation=report.get('scheduler_ablation')
    if ablation and ablation['episodes']:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        extra=ablation['episodes'];all_rt=rt+extra
        for model_index,model in enumerate(('smolvla','pi0fast')):
            positions=[model_index*3,model_index*3+1];c=COLORS[model]
            for position,prefetch,rows in zip(positions,(0,5),(rt,extra),strict=True):
                rows=[r for r in rows if r['model']==model]
                if not rows:continue
                successes=sum(bool(r['success']) for r in rows);rate=successes/len(rows);low,high=wilson(successes,len(rows))
                axes[0,0].bar(position,rate,color=c,alpha=.45 if prefetch==0 else 1.,width=.65)
                axes[0,0].errorbar(position,rate,yerr=[[max(0,rate-low)],[max(0,high-rate)]],fmt='none',color='#333333',capsize=4)
                axes[0,0].text(position,min(1.08,rate+.08),f'{successes}/{len(rows)}',ha='center',weight='bold')
            for pair in ablation['paired_prefetch0_prefetch5']:
                if pair['model']!=model:continue
                for ax,keys,multiplier in [(axes[0,1],('prefetch0_hold_fraction','prefetch5_hold_fraction'),100),(axes[1,0],('prefetch0_goal_xy_mm','prefetch5_goal_xy_mm'),1)]:
                    ax.plot(positions,[pair[key]*multiplier for key in keys],color=c,alpha=.45,marker='o',lw=1)
                matching=[r for r in all_rt if r['model']==model and r['init_state_index']==pair['init_state_index']]
                before=next((r for r in matching if r['prefetch_steps']==0),None);after=next((r for r in matching if r['prefetch_steps']==5),None)
                if before and after:axes[1,1].plot(positions,[r['metrics_recomputed']['action_source_age_ms_steady']['mean'] for r in (before,after)],color=c,alpha=.45,marker='o',lw=1)
        for ax in axes.flat:ax.set_xticks([0,1,3,4],['SmolVLA\np0','SmolVLA\np5','π0-FAST\np0','π0-FAST\np5']);ax.grid(axis='y',alpha=.15)
        axes[0,0].set(title='Native success • count + Wilson95%',ylabel='Success rate',ylim=(0,1.18))
        axes[0,1].set(title='Paired initial states • queue starvation',ylabel='Hold ticks [%]',ylim=(0,100))
        axes[1,0].set(title='Paired terminal goal distance',ylabel='Final bowl–plate XY [mm]')
        axes[1,1].set(title='Paired applied-command source age',ylabel='Steady per-episode mean [ms]')
        fig.suptitle(preview+'Exploratory prefetch0→5 • E10/20Hz/ROS • RTC disabled',weight='bold',fontsize=13)
        for ext in ('png','svg'):fig.savefig(output/f'prefetch.{ext}',dpi=170);plots.append(f'prefetch.{ext}')
        plt.close(fig)
    return plots


def main():
    """Write report artifacts only after formal pairs and frozen-weight audits exist.

    Args: CLI run roots/states/output, optional explicit partial preview.
    Returns: None. Raises: Validation/IO errors without changing source results.
    Design: Context: Formal runs may still be in progress.
        Concern: Partial data could be published as final.
        Decision: Require complete pairs by default and mark every preview partial.
        Alternatives: Silently ignoring missing directories.
        Quality Attributes: Integrity. Trade-offs: User must rerun after completion.
    Evidence: Internal: SHA256 source manifest.
        External: JSON/CSV. Findings: Downloadable values match plotted evidence.
        Metrics: Source count and episode count. Confidence: Validation-defined.
    Constraints: Output under artifacts/realtime-vla/report.
    Reconsider When: Another report destination is explicitly authorized.
    References: make_demo.py. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--runs',nargs='+',type=Path,default=DEFAULT_RUNS);parser.add_argument('--ablation-runs',nargs='+',type=Path);parser.add_argument('--native-smol-run',type=Path);parser.add_argument('--states',nargs='+',type=int,default=[1,2,3,4,5]);parser.add_argument('--output',type=Path,default=ROOT/'artifacts/realtime-vla/report');parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    allowed=(ROOT/'artifacts/realtime-vla/report').resolve();output=args.output.resolve()
    if not output.is_relative_to(allowed):raise ValueError('output must remain under artifacts/realtime-vla/report')
    report=aggregate(args.runs,args.states,args.allow_partial);output.mkdir(parents=True,exist_ok=True)
    if args.ablation_runs:report=attach_ablation(report,args.ablation_runs,args.states,args.allow_partial)
    if args.native_smol_run:report=attach_native_horizon(report,args.native_smol_run,args.states,args.allow_partial)
    sensor_audit=output/'sensor-phase-audit.json'
    if sensor_audit.exists():
        report['sensor_phase_diagnostic']=json.loads(sensor_audit.read_text())
        report['source_files'].append({'path':str(sensor_audit.relative_to(ROOT)),'sha256':hashlib.sha256(sensor_audit.read_bytes()).hexdigest(),'bytes':sensor_audit.stat().st_size})
    report['plots']=create_plots(report,output) if report['episodes'] else[]
    payload=json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    (output/'results.json').write_text(payload)
    (output/'summary.json').write_text(payload)
    fields=['model','mode','init_state_index','status','success','goal_xy_error_m','hold_fraction','effective_control_hz','effective_received_camera_hz','realtime_factor','wall_seconds','ticks','ros_all_verified']
    with (output/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k) for k in fields} for r in report['episodes'])
    if report.get('scheduler_ablation'):
        with (output/'scheduler-ablation.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields+['prefetch_steps']);writer.writeheader();writer.writerows({k:r.get(k) for k in fields+['prefetch_steps']} for r in report['scheduler_ablation']['episodes'])
    if report.get('native_horizon_smol'):
        with (output/'native-horizon-smol.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields+['execute_steps']);writer.writeheader();writer.writerows({k:r.get(k) for k in fields+['execute_steps']} for r in report['native_horizon_smol']['episodes'])
    print(json.dumps({'status':report['status'],'episodes':len(report['episodes']),'groups':report['groups'],'output':str(output)},ensure_ascii=False))

if __name__=='__main__':main()
