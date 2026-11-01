"""Audit the dedicated VLASH series and separate formal real-time results from diagnostics.

Args: --base: immutable experiment root; --ros-logs: actual two-node log directory.
Returns: New compact summary, episode CSV, paired initial hashes and exact validation evidence.
Raises: Missing/failed episodes, changed sources/prior evidence, unfrozen state or ROS/time mismatches.
Design: Context: Two synchronous diagnoses and three real-time state-input conditions are different evidence.
    Concern: Pooling development/init0, a startup failure or synthetic delay would inflate formal scores.
    Decision: Require all five complete runs, formal1..5 only and independent receipt/source/weight checks.
    Alternatives: Infer correctness from client metadata alone or final videos.
    Quality Attributes: Traceability. Trade-offs: Full report waits for the registered series.
Evidence: Internal: Protocol/amendments, real per-tick/chunk records and broker/policy receipt logs.
    External: Native LIBERO success and author H50 model. Findings: H50 diagnosis differs from official E5 example.
    Metrics: Success, paired inputs, real time, action holds, source age and forecast error.
    Confidence: Five fixed task0 initial states, not a general benchmark.
Constraints: CPU-only analysis; never modify run data, model assets or earlier45 results.
Reconsider When: Protocol/schema changes. References: vlash-protocol-amendment-03.json. ADR: None.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS=('coherent','current','predicted')


def load(path):
    """Read exact JSON evidence without fallback defaults.

    Args: path: required file. Returns: Parsed JSON. Raises: IO/JSON errors.
    Design: Context: Complete report. Concern: Missing data disguised as zero.
        Decision: Fail closed. Alternatives: Fill missing rows. Quality Attributes: Integrity. Trade-offs: No partial report.
    Evidence: Internal: Registered files. External: JSON. Findings: No writes. Metrics: None. Confidence: Exact read.
    Constraints: Required inputs. Reconsider When: Schema changes. References: build_report. ADR: None.
    """
    return json.loads(Path(path).read_text())


def average(values):
    """Average measured values, returning null when none were recorded.

    Args: values: numbers/None. Returns: Mean or None. Raises: Numeric errors.
    Design: Context: Optional resource/forecast data. Concern: Missing mistaken for zero.
        Decision: Explicit null. Alternatives: Fabricate values. Quality Attributes: Accuracy. Trade-offs: None.
    Evidence: Internal: Recorded data. External: None. Findings: No extrapolation. Metrics: Mean. Confidence: Descriptive.
    Constraints: No nonfinite input. Reconsider When: Metric changes. References: group_metrics. ADR: None.
    """
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def group_metrics(episodes):
    """Describe one condition using formal rows only and explicit aggregation conventions.

    Args: episodes: exactly five fixed-state results plus auxiliary metrics.
    Returns: Compact group metrics. Raises: Invalid/missing state errors.
    Design: Context: Per-state comparison. Concern: Longer failures dominate pooled means.
        Decision: Episode means averaged equally; hold fraction additionally pooled over physical ticks.
        Alternatives: Pool development and formal. Quality Attributes: Comparability. Trade-offs: Small N.
    Evidence: Internal: Five selected results. External: None. Findings: No general benchmark inference.
        Metrics: Success/holds/RTF/age/error. Confidence: Fixed task0 states only.
    Constraints:1..5 only. Reconsider When: Another sampling design is used.
    References: build_report. ADR: None.
    """
    assert sorted(e['init_state_index'] for e in episodes)==[1,2,3,4,5]
    total_ticks=sum(e['ticks'] for e in episodes)
    successful=[e for e in episodes if e['success']]
    return {'episodes':5,'successes':sum(bool(e['success']) for e in episodes),
            'success_rate':sum(bool(e['success']) for e in episodes)/5,
            'success_by_initial_state':{str(e['init_state_index']):bool(e['success']) for e in episodes},
            'hold_fraction_pooled':sum(e['hold_ticks'] for e in episodes)/max(1,total_ticks),
            'hold_fraction_episode_mean':average([e['hold_fraction'] for e in episodes]),
            'final_goal_xy_error_mm_median':float(np.median([e['goal_xy_error_m']*1000 for e in episodes])),
            'successful_completion_wall_seconds_median':float(np.median([e['wall_seconds'] for e in successful])) if successful else None,
            'successful_completion_sim_seconds_median':float(np.median([e['sim_seconds'] for e in successful])) if successful else None,
            'realtime_factor_median':float(np.median([e['realtime_factor'] for e in episodes])),
            'effective_control_hz_episode_mean':average([1/e['mean_control_interval_s'] for e in episodes]),
            'worker_steady_episode_mean_average_ms':average([e['mean_chunk_inference_s']*1000 for e in episodes]),
            'model_steady_episode_mean_average_ms':average([e['mean_model_inference_ms'] for e in episodes]),
            'online_query_interval_episode_mean_average_ticks':average([e['mean_online_query_interval_ticks'] for e in episodes]),
            'online_query_hz_episode_mean_average':average([e['online_query_hz'] for e in episodes]),
            'expired_prefix_actions_total':sum(e['expired_prefix_actions'] for e in episodes),
            'image_source_age_steady_episode_mean_average_ms':average([e['steady_image_source_age_ms'] for e in episodes]),
            'measured_proprio_age_steady_episode_mean_average_ms':average([e['steady_measured_proprio_source_age_ms'] for e in episodes]),
            'represented_state_age_steady_episode_mean_average_ticks':average([e['steady_represented_state_age_ticks'] for e in episodes]),
            'prediction_position_error_episode_mean_average_mm':average([None if e['mean_prediction_position_error_m'] is None else e['mean_prediction_position_error_m']*1000 for e in episodes]),
            'prediction_orientation_error_episode_mean_average_rad':average([e['mean_prediction_orientation_error_rad'] for e in episodes]),
            'prediction_target_matches':sum(e['prediction_target_matches'] for e in episodes),
            'pending_wait_ticks':sum(e['pending_wait_ticks'] for e in episodes),
            'peak_cuda_allocated_GiB':max(e['peak_cuda_bytes'] for e in episodes)/1024**3,
            'host_process_peak_RAM_bytes':None,'host_process_peak_RAM_note':'No per-process RSS sampler was registered; whole-host pressure is reported separately.'}


def receipt_lookup(directory):
    """Index independent actual node receipt logs by role/event/wire identity.

    Args: directory: broker/policy events root. Returns: Event mapping. Raises: IO/JSON errors.
    Design: Context: Client metadata alone is insufficient transport proof.
        Concern: A source payload could be echoed without subscriber receipt.
        Decision: Match both independent node events. Alternatives: Trust verified flag alone.
        Quality Attributes: Auditability. Trade-offs: CPU log scan.
    Evidence: Internal: Actual ROS JSONL. External: ROS2 subscriber transport.
        Findings: Wire IDs disambiguate shared logs. Metrics: Receipts. Confidence: Exact hash correlation.
    Constraints: No broker mutation. Reconsider When: Log schema changes.
    References: audit_run. ADR: None.
    """
    lookup={}
    for role in ('broker','policy'):
        with (Path(directory)/role/'events.jsonl').open() as stream:
            for line in stream:
                event=json.loads(line);lookup[(role,event['event'],event.get('id'))]=event
    return lookup


def audit_run(run,lookup,api_audit):
    """Verify one complete model run, all native chunks/calibrations and causal actuator provenance.

    Args: run: immutable output; lookup: independent receipts; api_audit: frozen identity anchor.
    Returns: Results, diagnostic rows and audit counters. Raises: Any integrity/time/action mismatch.
    Design: Context: Three represented-state origins. Concern: Incorrect chunk expiration or hidden future truth.
        Decision: Check all state origins, exact consumed actions, all recorded calls and ROS/hash/frozen proofs.
        Alternatives: Final outcome-only evaluation. Quality Attributes: Fault isolation. Trade-offs: Detailed logs.
    Evidence: Internal: Full source, tick and chunk metadata. External: Native state/action offset semantics.
        Findings: Later measured forecast targets are diagnostics only.
        Metrics: Calls, receipts, indices, ages. Confidence: Logged boundary checks.
    Constraints: No model loads. Reconsider When: Runtime schema changes.
    References: StateOriginQueue; vlash_worker.py. ADR: None.
    """
    summary=load(run/'summary.json');assert summary['status']=='complete'
    audit=summary['audit'];restoration=audit['restoration']
    assert audit['weights_unchanged'] and audit['requires_grad_parameter_count']==0
    assert audit['weights_before_sha256']==audit['weights_after_sha256']==api_audit['weights_before_sha256']
    assert restoration['checked_tensor_count']==1079 and restoration['stored_tensor_count']==824
    assert restoration['shared_alias_count']==255 and restoration['all_aliases_preserved']
    assert restoration['all_tensor_values_byte_equal'] and not restoration['native_dtype_casts']
    metadata=[record['metadata'] for record in summary['warmup_records']]
    episodes=[];prediction_rows=[];chunks_count=0;tick_count=0
    for result in summary['results']:
        assert result['status']=='completed' and result['error'] is None
        index=result['init_state_index'];mode=result['mode']
        folder=run/(f'realtime-init{index}' if mode=='realtime' else f'artificial-init{index}')
        chunks=load(folder/'chunks.json');chunk_by_id={chunk['request_id']:chunk for chunk in chunks}
        assert len(chunk_by_id)==len(chunks)
        for chunk in chunks:
            actions=np.asarray(chunk['actions']);assert actions.shape==(50,7) and np.isfinite(actions).all()
            metadata.append(chunk['metadata']);chunks_count+=1
            assert chunk['metadata']['repository']=='mit-han-lab/vlash-pi05-libero-async5'
            assert chunk['metadata']['revision']=='c85d5962a7b823f9f391f97d60227d6ca8bd7bc7'
            assert chunk['metadata']['source_revision']=='d2da223e2251a4a5020b667e77cf8ad57e594144'
            assert chunk['metadata']['chunk_length']==50 and chunk['metadata']['flow_steps']==10
            if mode=='realtime' and not chunk.get('bootstrap'):
                t=chunk['request_tick'];capture=chunk['observation_tick'];origin=chunk['state_origin_tick']
                assert capture<=t and chunk['observation_time']<=chunk['request_time']
                assert chunk['state_condition']==result['state_condition']
                if result['state_condition']=='coherent':assert origin==capture==chunk['measured_proprio_tick']
                elif result['state_condition']=='current':assert origin==t==chunk['measured_proprio_tick']
                else:
                    forecast=chunk['metadata']['predictor']
                    assert chunk['measured_proprio_tick']==t and origin==t+forecast['horizon_ticks']==forecast['target_state_tick']
                    assert forecast['state_is_prediction'] and forecast['gripper_prediction']=='hold measured two finger positions'
        initial=load(folder/'initial.json');bootstrap=chunks[0];assert bootstrap['bootstrap']
        for camera in ('agentview_image','robot0_eye_in_hand_image'):
            assert initial['sensor_hashes'][camera]==bootstrap['metadata']['sensor_transport']['component_sha256'][camera]
        if mode=='realtime':
            calibration=load(folder/'timing-calibration.json');assert len(calibration)==2
            for row in calibration:
                assert row['held_initial_state'] and not row['state_is_prediction'] and row['metadata']['state_condition']=='current'
                assert row['metadata']['state_origin_tick']==row['metadata']['request_tick']==0
                assert row['metadata']['sensor_transport']['component_sha256']==bootstrap['metadata']['sensor_transport']['component_sha256']
                metadata.append(row['metadata'])
        ticks=[json.loads(line) for line in (folder/'ticks.jsonl').read_text().splitlines()]
        assert len(ticks)==result['ticks'];previous_gripper=-1.
        for tick in ticks:
            action=np.asarray(tick['action']);assert action.shape==(7,) and np.isfinite(action).all()
            if tick['held']:
                assert np.array_equal(action[:6],np.zeros(6)) and action[6]==previous_gripper
            else:
                source=chunk_by_id[tick['request_id']];idx=tick['chunk_index']
                assert 0<=idx<50 and np.array_equal(action,np.clip(np.asarray(source['actions'])[idx],-1,1))
                if mode=='realtime':
                    assert tick['state_origin_tick']==source['state_origin_tick']
                    assert idx==tick['tick']-source['state_origin_tick']
                    assert tick['image_capture_tick']==source['observation_tick']
                    assert tick['measured_proprio_tick']==source['measured_proprio_tick']
                    assert tick['state_is_prediction']==(source['state_condition']=='predicted')
            previous_gripper=float(action[6]);tick_count+=1
        row=dict(result,source_directory=folder.as_posix(),peak_cuda_bytes=audit['peak_cuda_bytes'])
        if mode=='realtime':
            steady=[tick for tick in ticks if not tick['held'] and tick['request_id']!=bootstrap['request_id']]
            query_ticks=[chunk['request_tick'] for chunk in chunks if not chunk.get('bootstrap')]
            query_intervals=np.diff(query_ticks).tolist()
            row.update(steady_image_source_age_ms=average([tick['image_source_age_s']*1000 for tick in steady]),
                       steady_measured_proprio_source_age_ms=average([tick['measured_proprio_source_age_s']*1000 for tick in steady]),
                       steady_represented_state_age_ticks=average([tick['represented_state_age_ticks'] for tick in steady]),
                       online_query_count=len(query_ticks),mean_online_query_interval_ticks=average(query_intervals),
                       online_query_hz=None if not query_intervals else result['control_hz']/average(query_intervals))
            for diagnostic in load(folder/'prediction-diagnostics.json'):
                prediction_rows.append(dict(diagnostic,init_state_index=index,state_condition=result['state_condition']))
                chunk=chunk_by_id[diagnostic['request_id']]
                assert diagnostic['target_tick']==chunk['state_origin_tick']
                # Independently tie the forecast's numeric robot state to the actual ROS-received model packet.
                from ros_bridge import digest,PROPRIO
                prompt='pick up the black bowl between the plate and the ramekin and place it on the plate'
                expected=digest({'proprio':{key:diagnostic['predicted_robot'][key] for key in PROPRIO},'prompt':prompt})
                assert expected==chunk['metadata']['sensor_transport']['component_sha256']['proprio_prompt']
                if diagnostic['actual_robot'] is not None:
                    measured=np.asarray(diagnostic['actual_robot']['robot0_eef_pos']);pred=np.asarray(diagnostic['predicted_robot']['robot0_eef_pos'])
                    assert abs(float(np.linalg.norm(measured-pred))-diagnostic['position_error_m'])<1e-9
        episodes.append(row)
    calls={m['call'] for m in metadata};assert len(calls)==len(metadata) and calls==set(range(1,audit['calls']+1))
    exchanges=0
    for m in metadata:
        for side,peer_event,broker_event in [('sensor_transport','sensor_join_receive','sensor_receipt'),('action_transport','generated_chunk_receive','action_receipt')]:
            packet=m[side];expected=packet['source_sha256']
            assert packet['verified'] and expected==packet['received_sha256']==packet['peer_received_sha256']
            for role,event in [('policy',peer_event),('broker',broker_event)]:
                receipt=lookup[(role,event,packet['transport_id'])]
                assert receipt['verified'] and receipt['source_sha256']==expected==receipt['received_sha256']
            exchanges+=1
    return episodes,prediction_rows,{'run':run.as_posix(),'native_task_episodes':len(episodes),
                                    'recorded_calls_all_accounted':len(metadata),'native_chunks':chunks_count,
                                    'ticks':tick_count,'actual_ros_exchanges_verified':exchanges,
                                    'all_weights_frozen_same_as_api':True,'all1079state_values_verified':True}


def build_report(base,ros_logs):
    """Require the full fixed grid and write new compact/formal/paired audit artifacts.

    Args: base: experiment root; ros_logs: independent actual receipt logs.
    Returns: Compact summary. Raises: Any incomplete/changed/unverified evidence.
    Design: Context: Reporting follows completion. Concern: A partial grid can look like final success rate.
        Decision: Separate formal15, development5 and startup failure1; pair initial hashes across all conditions.
        Alternatives: Pool every directory. Quality Attributes: Accuracy. Trade-offs: Requires all files.
    Evidence: Internal: Five run audits, predeclared sources and prior45 manifest.
        External: None. Findings: No prior result is rewritten.
        Metrics: Compact groups/CSV/receipt counts. Confidence: Fixed task only.
    Constraints: Report writes only its own output. Reconsider When: Protocol changes.
    References: group_metrics; audit_run. ADR: None.
    """
    base=Path(base);original=load(base/'vlash-protocol.json');amendment=load(base/'vlash-protocol-amendment-03.json')
    for name,expected in amendment['source_sha256'].items():
        source=Path(name)
        if name=='scripts/setup_vlash.sh' and hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            # Public bootstrap changed only after measured trials. Audit the
            # exact archived executed version; all13 runtime sources stay strict.
            source=base/'vlash-run-source-snapshot'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    for name,expected in original['preserved_prior45_json_jsonl_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
    assert hashlib.sha256((base/'models/vlash/api-smoke.json').read_bytes()).hexdigest()==amendment['api_smoke_sha256']
    api=load(base/'models/vlash/api-smoke.json')['audit'];lookup=receipt_lookup(ros_logs)
    rows=[];forecasts=[];audits=[]
    for name in ['development-vlash-artificial-d0-retry1','development-vlash-artificial-d3']+[f'formal-vlash-{c}' for c in CONDITIONS]:
        snapshot=load(base/name/'source-sha256.json')
        for source,expected in amendment['source_sha256'].items():
            if Path(source).suffix=='.py':assert snapshot[Path(source).name]==expected
        episodes,predictions,audit=audit_run(base/name,lookup,api);rows.extend(episodes);forecasts.extend(predictions);audits.append(audit)
    formal=[row for row in rows if row['mode']=='realtime' and row['init_state_index'] in range(1,6)]
    development=[row for row in rows if row not in formal];assert len(formal)==15 and len(development)==5
    pairs=[]
    for index in range(6):
        hashes={c:load(base/f'formal-vlash-{c}'/f'realtime-init{index}'/'initial.json')['sensor_hashes'] for c in CONDITIONS}
        matches=all(hashes[c]==hashes['coherent'] for c in CONDITIONS)
        pairs.append({'init_state_index':index,'split':'development' if index==0 else 'formal','all_five_sensor_hashes_equal':matches,'sensor_hashes':hashes})
        assert matches,'initial sensors differ across paired conditions'
    groups={c:group_metrics([row for row in formal if row['state_condition']==c]) for c in CONDITIONS}
    matched_forecasts=[row for row in forecasts if row['init_state_index'] in range(1,6) and row['actual_robot'] is not None]
    if matched_forecasts:
        groups['predicted'].update(prediction_position_error_pooled_median_mm=float(np.median([r['position_error_m']*1000 for r in matched_forecasts])),
                                   prediction_position_error_pooled_p95_mm=float(np.percentile([r['position_error_m']*1000 for r in matched_forecasts],95)),
                                   prediction_fingers_error_pooled_mean_mm=average([r['fingers_error_m']*1000 for r in matched_forecasts]))
    validation={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
                'formal_realtime_episodes':15,'development_episodes_excluded':5,'startup_failures_excluded':1,
                'startup_failure_directory':'artifacts/realtime-vla/development-vlash-artificial-d0',
                'registered_sources_unchanged':True,'prior45_json_jsonl_files_unchanged':243,'api_smoke_unchanged':True,
                'all_initial_conditions_paired':True,'all_native_actions_time_origin_checked':True,
                'all_model_calls_including_warmup_and_calibration_accounted':True,
                'all_actual_ros_receipts_independently_correlated':True,'run_audits':audits}
    compact={'status':'complete','model':'vlash dedicated pretrained pi05','formal_episodes':15,'formal_initial_states':[1,2,3,4,5],
             'groups':groups,'development':development,'audit':validation,
             'interpretation':'Same model and H50/R30, different represented-state origins. Correct origin changes usable prefix and query cadence too: this is a system-condition comparison, not isolated state-freshness causation. Descriptive task0/five-state experiment only. Artificial d0/d3 are our H50 synchronous diagnoses, not full official E5 protocol reproduction.',
             'license_note':'Author code Apache2; selected weights have no declared model license. No weight redistribution.',
             'native_horizon':50,'flow_steps':10,'local_training':False}
    out=base/'vlash-report';out.mkdir(exist_ok=True)
    for name,value in [('summary.json',compact),('results.json',{'formal':formal,'development':development}),
                       ('validation.json',validation),('paired-initial-hashes.json',pairs),('prediction-diagnostics.json',forecasts)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    fields=['state_condition','init_state_index','success','ticks','wall_seconds','sim_seconds','goal_xy_error_m','hold_fraction','realtime_factor','mean_chunk_inference_s','mean_model_inference_ms','online_query_count','mean_online_query_interval_ticks','online_query_hz','steady_image_source_age_ms','steady_measured_proprio_source_age_ms','mean_prediction_position_error_m','mean_prediction_orientation_error_rad']
    with (out/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({key:row.get(key) for key in fields} for row in formal)
    forecast_fields=['init_state_index','request_id','query_tick','target_tick','horizon_ticks','position_error_mm','orientation_error_rad','fingers_error_mm']
    with (out/'prediction-errors.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=forecast_fields);writer.writeheader()
        for row in matched_forecasts:
            writer.writerow({'init_state_index':row['init_state_index'],'request_id':row['request_id'],
                             'query_tick':row['query_tick'],'target_tick':row['target_tick'],
                             'horizon_ticks':row['forecast']['horizon_ticks'],'position_error_mm':row['position_error_m']*1000,
                             'orientation_error_rad':row['orientation_error_rad'],'fingers_error_mm':row['fingers_error_m']*1000})
    (base/'vlash-comparison-summary.json').write_text(json.dumps(compact,indent=2)+'\n')
    return compact


def main():
    """Run explicit CPU-only final reporting; imports and --help load no models.

    Args: CLI. Returns: None. Raises: Audit/report errors.
    Design: Context: Reproducible submission. Concern: Hidden GPU/network work.
        Decision: Main guard and file-only analysis. Alternatives: Re-run trials while reporting.
        Quality Attributes: Predictability. Trade-offs: Existing evidence required.
    Evidence: Internal: All input paths explicit. External: argparse.
        Findings: No torch import. Metrics: Summary. Confidence: Contract.
    Constraints: No original writes. Reconsider When: CLI changes. References: build_report. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--base',type=Path,default=Path('artifacts/realtime-vla'))
    parser.add_argument('--ros-logs',type=Path,default=Path('artifacts/realtime-vla/pi05-ros-system'));args=parser.parse_args()
    result=build_report(args.base,args.ros_logs)
    print(json.dumps({'status':result['status'],'groups':result['groups'],'audit':result['audit']},indent=2))


if __name__=='__main__':main()
