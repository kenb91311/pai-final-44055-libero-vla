"""Evaluate author VLASH artificial delay and causal real-time state conditioning separately.

Args: Frozen model, current/coherent/predicted state, initial states and fixed timing protocol.
Returns: New per-tick/chunk/forecast evidence, outcomes and frozen/ROS audits.
Raises: Explicit asset/controller/worker/timing errors; failed runs retain evidence.
Design: Context: VLASH learns state/action offsets but its author evaluator imposes synthetic image delay.
    Concern: That evaluator is not a wall-clock inference-latency experiment.
    Decision: Keep author-style synchronous d0/d3 reproduction distinct from H50/R30 realtime;
        all realtime conditions align actions to their represented state, not automatically the image.
    Alternatives: Pool synthetic and real delay or use oracle future state as policy input.
    Quality Attributes: Causal comparison, traceability. Trade-offs: New isolated author backend.
Evidence: Internal: Unit-tested PD forecast and state-origin queue; fixed native assets.
    External: Author eval_libero.py and VLASHDataset offset logic.
    Findings: Old images/current or predicted state have different time origins.
    Metrics: Success, RTF, holds, source age, dropped prefix and measured forecast error.
    Confidence: Fixed-task measurements; no general benchmark claim.
Constraints: Real ROS image/chunk transport, one model, no local training or object truth input.
Reconsider When: Training fps/offset metadata or new author checkpoints become available.
References: docs/reference/vlash-libero-assets.md. ADR: None.
"""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
from evaluate import write_json
from libero_env import canonical_state
from vlash_worker import VLASHWorker
from vlash_timing import StateOriginQueue,forecast_robot_sensors,quaternion_log,quaternion_product

PROPRIO=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def robot_only(raw):
    """Copy only measured EEF pose and finger sensors for a causal history.

    Args: raw: simulator sensor mapping. Returns: Three-field dictionary. Raises: Missing fields.
    Design: Context: Prediction history. Concern: Whole observations contain evaluator truth.
        Decision: Explicit three-field allowlist. Alternatives: Store entire simulator state.
        Quality Attributes: Causality. Trade-offs: No joint/contact model.
    Evidence: Internal: Forecast API. External: Canonical LIBERO state8.
        Findings: No object state. Metrics: Keys. Confidence: Explicit boundary.
    Constraints: Copies detach mutable arrays. Reconsider When: Sensors change.
    References: build_request. ADR: None.
    """
    return {key:np.asarray(raw[key]).copy() for key in PROPRIO}


def verify_native_controller(env):
    """Require forecast scales/gains to match the actual installed controller before evaluation.

    Args: env: physical LIBERO wrapper. Returns: Read-only controller evidence.
    Raises: RuntimeError when native OSC assumptions differ.
    Design: Context: Analytic forecast depends on physical units. Concern: Defaults may not match a scene override.
        Decision: Read actual controller gains/scales and fail on differences.
        Alternatives: Assume class defaults. Quality Attributes: Reproducibility. Trade-offs: Fixed controller only.
    Evidence: Internal: Live OSC kp/kd/output ranges. External: robosuite1.4 controller source.
        Findings: Evaluator reads controller configuration, not object truth, and never feeds it as observation.
        Metrics: Exact gain/scale arrays. Confidence: Runtime configuration check.
    Constraints: No gain changes. Reconsider When: Controller config changes.
    References: forecast_robot_sensors. ADR: None.
    """
    controller=env.env.env.robots[0].controller
    values={key:np.asarray(getattr(controller,key),dtype=np.float64) for key in ('kp','kd','output_min','output_max')}
    expected=np.array([.05,.05,.05,.5,.5,.5])
    if not (np.allclose(values['kp'],150) and np.allclose(values['kd'],2*math.sqrt(150)) and
            np.allclose(values['output_max'],expected) and np.allclose(values['output_min'],-expected)):
        raise RuntimeError('native OSC controller differs from declared VLASH predictor assumptions')
    return dict(controller_class=type(controller).__name__,**{key:value.tolist() for key,value in values.items()})


def build_request(sensor,current,history,tick,proprio_time,condition,delay_ticks,queue,control_hz):
    """Construct mixed-age inputs with independently labelled image, measurement and state origin.

    Args: sensor: captured RGB/proprio; current/history: measured robot sensing;
        tick/proprio_time: query/current sensing; condition: three-state variant;
        delay_ticks: estimated completion steps; queue: known actuator plan; control_hz: physical rate.
    Returns: Raw request, worker timing kwargs and optional forecast diagnostic.
    Raises: Invalid causal source/forecast errors.
    Design: Context: Comparing one trained model under three conditioning choices.
        Concern: Images, actual proprioception and predicted state must retain their real provenance.
        Decision: Copy only raw sensor keys and use exact state-origin semantics.
        Alternatives: Relabel predictions as new measurements.
        Quality Attributes: Auditability. Trade-offs: Forecast is an explicit approximation.
    Evidence: Internal: StateOriginQueue/forecast unit tests. External: VLASH offset semantics.
        Findings: Coherent origin=capture, current=query, predicted=query+estimated delay.
        Metrics: Three ticks, query ages and planned prefix. Confidence: Causal construction.
    Constraints: No privileged truth enters request or forecast. Reconsider When: New conditions are tested.
    References: run_realtime. ADR: None.
    """
    capture=int(sensor['tick']);now=time.monotonic()
    if capture>tick or capture<0:raise ValueError('noncausal camera source')
    mixed={key:np.asarray(sensor[key]).copy() for key in ('agentview_image','robot0_eye_in_hand_image')}
    mixed['prompt']=sensor['prompt'];forecast=None;diagnostic=None
    if condition=='coherent':
        mixed.update(robot_only(sensor));origin=capture;measured_tick=capture;measured_time=sensor['capture_time']
    else:
        mixed.update(robot_only(current));origin=tick;measured_tick=tick;measured_time=proprio_time
        if condition=='predicted':
            prefix=np.asarray([item['action'] for item in queue.buffer.queue],dtype=np.float64).reshape(-1,7)
            overrides,forecast=forecast_robot_sensors(history[-1],history[-2] if len(history)>1 else None,
                                                     prefix,int(delay_ticks),control_hz=control_hz)
            mixed.update(overrides);origin=tick+int(delay_ticks)
            forecast.update(planned_prefix=prefix.tolist(),query_image_age_s=max(0.,now-sensor['capture_time']),
                            query_measured_proprio_age_s=max(0.,now-measured_time))
            diagnostic={'target_tick':origin,'query_tick':tick,'image_tick':capture,
                        'predicted_robot':{key:value.tolist() for key,value in overrides.items()},
                        'measured_start_robot':{key:value.tolist() for key,value in robot_only(current).items()},
                        'forecast':forecast,'actual_robot':None}
    kwargs={'request_tick':tick,'measured_proprio_tick':measured_tick,'measured_proprio_time':measured_time,
            'state_origin_tick':origin,'state_condition':condition,'predictor':forecast}
    return mixed,kwargs,diagnostic


def match_prediction_targets(diagnostics,raw,tick):
    """Record later measured prediction error after a target tick has actually happened.

    Args: diagnostics: past forecasts; raw: now measured robot sensing; tick: current physical tick.
    Returns: None. Raises: Invalid quaternion/sensor errors.
    Design: Context: Forecast validity. Concern: Future observations must never feed earlier inference.
        Decision: Posthoc target matching only after physical execution.
        Alternatives: Oracle future rollout as policy input.
        Quality Attributes: Causal diagnostics. Trade-offs: Targets after termination remain unmatched.
    Evidence: Internal: Tick-ordered loop. External: Sensor-based prediction evaluation.
        Findings: Object truth is unnecessary. Metrics: Position/rotation/finger errors.
        Confidence: Actual measured robot target values.
    Constraints: Diagnostic values are never passed back into the model. Reconsider When: Online adaptation is added.
    References: run_realtime. ADR: None.
    """
    for item in diagnostics:
        if item['target_tick']!=tick or item['actual_robot'] is not None:continue
        actual=robot_only(raw);pred=item['predicted_robot']
        q=np.asarray(pred['robot0_eef_quat']);inverse=np.r_[-q[:3],q[3]]
        item.update(actual_robot={key:value.tolist() for key,value in actual.items()},
                    position_error_m=float(np.linalg.norm(actual['robot0_eef_pos']-np.asarray(pred['robot0_eef_pos']))),
                    orientation_error_rad=float(np.linalg.norm(quaternion_log(quaternion_product(actual['robot0_eef_quat'],inverse)))),
                    fingers_error_m=float(np.linalg.norm(actual['robot0_gripper_qpos']-np.asarray(pred['robot0_gripper_qpos']))))


def fresh_camera(camera,env,episode_id):
    """Wait outside physical control for one camera result from the just-reset state.

    Args: camera/env: owned wrappers; episode_id: reset identity. Returns: Raw sensor packet.
    Raises: Timeout or camera errors.
    Design: Context: Reset/warm-start. Concern: Previous episode camera results can leak through queues.
        Decision: Capture-time and episode filter. Alternatives: Accept any available frame.
        Quality Attributes: Reproducibility. Trade-offs: Untimed initialization wait.
    Evidence: Internal: CameraWorker metadata. External: None. Findings: No timed-loop wait.
        Metrics: Capture time. Confidence: Explicit identity checks.
    Constraints: Physics is held here only. Reconsider When: Camera reset protocol changes.
    References: run_realtime. ADR: None.
    """
    refreshed=time.monotonic();camera.submit(env.snapshot(tick=0,episode_id=episode_id));deadline=refreshed+60
    while time.monotonic()<deadline:
        result=camera.wait(timeout=max(.1,deadline-time.monotonic()))
        if result['episode_id']==episode_id and result['capture_time']>=refreshed:return result
    raise TimeoutError('fresh camera initialization deadline')


def warm_once(worker,sensor,seed):
    """Perform two native warmups per loaded model, excluded from task timing.

    Args: worker: resident VLA; sensor: real initial observation; seed: fixed RNG.
    Returns: None. Raises: Worker/inference errors.
    Design: Context: Warm inference study. Concern: First-call kernel initialization affects latency.
        Decision: Exactly two warmups with reset. Alternatives: Count cold startup as steady latency.
        Quality Attributes: Comparability. Trade-offs: Cold start reported separately.
    Evidence: Internal: Existing baseline warm2. External: None. Findings: No environment step.
        Metrics: Warmup records. Confidence: Actual timings.
    Constraints: Held initial state. Reconsider When: Cold deployment is studied.
    References: VLASHWorker. ADR: None.
    """
    if worker.warmed:return
    for _ in range(2):
        worker.reset(seed);worker.predict(sensor,0,sensor['capture_time'])
        response=worker.wait();worker.warmup_records.append({'metadata':response['metadata'],'wall_seconds':response['finished_time']-response['started_time']})
    worker.warmed=True


def run_realtime(env,camera,worker,args,index,out):
    """Advance physical20Hz independently of H50 inference and activate by state origin.

    Args: env/camera/worker: owned processes; args/index/out: declared protocol and identity.
    Returns: Episode result. Raises: Initialization errors; timed-loop errors are preserved.
    Design: Context: Coherent/current/predicted conditions share H50/R30 scheduling.
        Concern: A future-state chunk can arrive early; waiting must not pause physical control.
        Decision: One inflight request, one pending result, aligned state-origin replacement.
        Alternatives: FIFO rebase or image-origin expiration for every condition.
        Quality Attributes: Causality, physical timing. Trade-offs:50ms activation resolution.
    Evidence: Internal: Unit-tested queue/PD predictor. External: Native OSC and author offset semantics.
        Findings: Actual later robot state is used only for forecast-error diagnostics.
            Equal per-episode calibration2 on the held measured state initializes latency, excluding cold warmups.
        Metrics: Holds, RTF, source ages and prediction error. Confidence: Logged task only.
    Constraints: No inference/ROS/camera waits in control ticks. Reconsider When: Multiple requests are used.
    References: build_request; StateOriginQueue. ADR: None.
    """
    out.mkdir(parents=True,exist_ok=False);period=1/args.control_hz;max_ticks=round(args.wall_seconds*args.control_hz)
    identity=f'vlash-{args.condition}-init{index}';current=env.reset(init_state_index=index,seed=args.seed)
    sensor=fresh_camera(camera,env,identity);warm_once(worker,sensor,args.seed)
    current=env.reset(init_state_index=index,seed=args.seed);sensor=fresh_camera(camera,env,identity)
    write_json(out/'initial.json',{'seed':args.seed,'init_state_index':index,'metrics':current['metrics'],
                                 'environment':env.metadata,'camera':camera.metadata,
                                 'sensor_hashes':{key:hashlib.sha256(np.asarray(sensor[key]).tobytes()).hexdigest()
                                                  for key in ('agentview_image','robot0_eye_in_hand_image',*PROPRIO)}})
    # Equal calibration for all conditions: use the actually measured held state,
    # never a forecast. Exclude both cold warmups from online delay estimation.
    calibrations=[]
    for _ in range(2):
        worker.predict(sensor,0,sensor['capture_time'],state_condition='current')
        calibration=worker.wait()
        calibrations.append({'wall_seconds':calibration['finished_time']-calibration['request_time'],
                             'metadata':calibration['metadata'],'held_initial_state':True,'state_is_prediction':False})
    write_json(out/'timing-calibration.json',calibrations)
    worker.reset(args.seed);worker.predict(sensor,0,sensor['capture_time']);first=worker.wait()
    queue=StateOriginQueue();first_event=queue.accept(first,0);chunks=[dict(first,bootstrap=True,ready_tick=0,event=first_event)]
    ticks=[];forecasts=[];switches=[];latencies=[r['wall_seconds'] for r in calibrations]
    history=deque([{'tick':0,'raw':robot_only(current)}],maxlen=4);writer=None
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(str(out/'rollout.mp4'),fps=args.control_hz,codec='libx264',macro_block_size=2)
    start=time.monotonic();next_tick=start;proprio_time=start;previous_tick=None;previous_action=None
    status='completed';error=None;frames=1;stale=0;pending_events=[]
    try:
        with (out/'ticks.jsonl').open('w') as stream:
            for tick in range(max_ticks):
                delay=next_tick-time.monotonic()
                if delay>0:time.sleep(delay)
                if time.monotonic()-start>=args.wall_seconds:break
                new=camera.poll()
                if new is not None and new['episode_id']==identity and new['tick']>=sensor['tick']:sensor=new;frames+=1
                now=time.monotonic();interval=None if previous_tick is None else now-previous_tick;previous_tick=now
                changed=False;ready=worker.poll() if worker.inflight else None
                if ready is not None:
                    event=queue.accept(ready,tick);chunks.append(dict(ready,bootstrap=False,ready_tick=tick,event=event))
                    latencies.append(ready['finished_time']-ready['request_time']);changed=event['queued_actions']>0
                activation=queue.advance(tick)
                if activation is not None:pending_events.append(activation);changed=activation['queued_actions']>0
                if len(queue)<=args.prefetch_steps and not worker.inflight and queue.pending is None:
                    if now-sensor['capture_time']<=args.max_sensor_age:
                        raw_delay=math.ceil(float(np.percentile(latencies[-10:],95))/period);predicted_delay=min(49,raw_delay)
                        mixed,kwargs,diagnostic=build_request(sensor,current,history,tick,proprio_time,args.condition,
                                                               predicted_delay,queue,args.control_hz)
                        if kwargs['predictor'] is not None:kwargs['predictor'].update(raw_delay_ticks=raw_delay,prediction_capped=raw_delay>49)
                        request=worker.predict(mixed,int(sensor['tick']),sensor['capture_time'],**kwargs)
                        if diagnostic is not None:diagnostic['request_id']=request;forecasts.append(diagnostic)
                    else:stale+=1
                depth=len(queue);item=queue.pop();action=item['action'];before=time.monotonic()
                current=env.step(action);after=time.monotonic();proprio_time=after
                history.append({'tick':tick+1,'raw':robot_only(current)})
                match_prediction_targets(forecasts,current,tick+1);camera.submit(env.snapshot(tick=tick+1,episode_id=identity))
                delta=None if previous_action is None else action-previous_action
                record={'tick':tick,'wall_time_since_start':before-start,'tick_interval_s':interval,
                        'deadline_lateness_s':max(0.,now-next_tick),'step_wall_s':after-before,
                        'queue_before':depth,'queue_after':len(queue),'held':item['held'],
                        'request_id':item['request_id'],'chunk_index':item['chunk_index'],
                        'image_capture_tick':item.get('image_capture_tick'),'state_origin_tick':item.get('state_origin_tick'),
                        'measured_proprio_tick':item.get('measured_proprio_tick'),'state_is_prediction':item.get('state_is_prediction',False),
                        'measured_proprio_source_age_s':None if item.get('measured_proprio_time') is None else before-item['measured_proprio_time'],
                        'image_source_age_s':None if item['observation_time'] is None else before-item['observation_time'],
                        'represented_state_age_ticks':None if item.get('state_origin_tick') is None else tick-item['state_origin_tick'],
                        'latest_camera_tick':int(sensor['tick']),'latest_camera_age_s':before-sensor['capture_time'],
                        'pending_target_tick':None if queue.pending is None else queue.pending['state_origin_tick'],
                        'action':action.tolist(),'action_delta_l2':None if delta is None else float(np.linalg.norm(delta)),
                        'translation_delta_l2':None if delta is None else float(np.linalg.norm(delta[:3])),
                        'rotation_delta_l2':None if delta is None else float(np.linalg.norm(delta[3:6])),
                        'gripper_delta_abs':None if delta is None else float(abs(delta[6])),
                        'chunk_switched':changed,'success':bool(current['success']),'metrics':current['metrics']}
                if changed and delta is not None:switches.append({k:record[k] for k in ('tick','translation_delta_l2','rotation_delta_l2','gripper_delta_abs')})
                ticks.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();previous_action=action.copy()
                if writer:writer.append_data(np.concatenate([np.rot90(sensor['agentview_image'],2),np.rot90(sensor['robot0_eye_in_hand_image'],2)],axis=1))
                next_tick=max(next_tick+period,after)
                if current['success'] or current['done']:break
    except Exception as exc:
        status='error';error=f'{type(exc).__name__}: {exc}';write_json(out/'error.json',{'error':error,'traceback':traceback.format_exc()})
    finally:
        elapsed=time.monotonic()-start
        if writer:writer.close()
        if worker.inflight:
            try:chunks.append(dict(worker.wait(),bootstrap=False,applied=False,completed_after_episode=True))
            except Exception as exc:status='error';error=f'drain failure: {exc}'
        write_json(out/'chunks.json',chunks);write_json(out/'pending-activation.json',pending_events);write_json(out/'prediction-diagnostics.json',forecasts)
    intervals=[row['tick_interval_s'] for row in ticks if row['tick_interval_s'] is not None]
    steady=[chunk for chunk in chunks if not chunk.get('bootstrap')]
    matched=[row for row in forecasts if row['actual_robot'] is not None]
    result={'status':status,'error':error,'model':'vlash','mode':'realtime','state_condition':args.condition,
            'init_state_index':index,'split':'development' if index==0 else 'formal','seed':args.seed,
            'success':bool(current['success']) and status=='completed',
            'success_within_wall_budget':bool(current['success']) and status=='completed' and elapsed<=args.wall_seconds+period,
            'ticks':len(ticks),'wall_seconds':elapsed,'sim_seconds':len(ticks)*period,'realtime_factor':len(ticks)*period/elapsed,
            'control_hz':args.control_hz,'wall_budget_s':args.wall_seconds,'native_generation_horizon':50,
            'prefetch_steps':args.prefetch_steps,'merge':'aligned_to_represented_state','hold_ticks':queue.buffer.holds,
            'hold_fraction':queue.buffer.holds/max(1,len(ticks)),'expired_prefix_actions':queue.buffer.discarded_expired,
            'replaced_actions':queue.buffer.discarded_overlap,'pending_wait_ticks':queue.pending_wait_ticks,
            'mean_control_interval_s':float(np.mean(intervals)) if intervals else None,
            'p95_control_interval_s':float(np.percentile(intervals,95)) if intervals else None,
            'mean_chunk_inference_s':float(np.mean([c['finished_time']-c['started_time'] for c in steady])) if steady else None,
            'mean_model_inference_ms':float(np.mean([c['metadata']['model_inference_ms'] for c in steady])) if steady else None,
            'mean_image_source_age_s':float(np.mean([t['image_source_age_s'] for t in ticks if t['image_source_age_s'] is not None])) if ticks else None,
            'sensor_frames_received':frames,'stale_sensor_ticks':stale,'goal_xy_error_m':current['metrics'].get('goal_xy_error_m'),
            'final_metrics':current['metrics'],'switches':switches,'prediction_target_matches':len(matched),
            'mean_prediction_position_error_m':float(np.mean([r['position_error_m'] for r in matched])) if matched else None,
            'mean_prediction_orientation_error_rad':float(np.mean([r['orientation_error_rad'] for r in matched])) if matched else None,
            'ros_transport':True,'camera_mode':'asynchronous_clone','camera_hz':args.camera_hz,'local_training':False,
            'configuration_note':'H50/R30, origin=capture/current/predicted target by condition; early chunks pending without blocking; warm2 plus per-episode held-current-state calibration2 and final seed-reset/bootstrap excluded; initial delay estimate uses only calibration2; no privileged policy inputs'}
    write_json(out/'result.json',result);return result


def run_artificial(env,worker,args,index,out):
    """Reproduce author old-image/current-state logic with paused-physics native H50 execution.

    Args: env/worker: synchronous sensors and model; args/index/out: declared delay and evidence paths.
    Returns: Explicit artificial-lockstep result. Raises: Worker/physics errors.
    Design: Context: Author evaluator buffers loop observations and consumes native queued actions.
        Concern: Synthetic delay must not be called actual real-time inference.
        Decision: Image lag d loop ticks, current state, action_quant1 and native50 action consumption.
        Alternatives: Simultaneous full-horizon R30 replacement.
        Quality Attributes: Faithful reproduction. Trade-offs: Physics pauses during each inference.
    Evidence: Internal: Fixed author eval_libero.py buffer logic. External: Released VLASH model.
        Findings: delay units are loop steps, separate from training sample offsets.
        Metrics: Task success and actual slow wall time. Confidence: Fixed-task dev only.
    Constraints: Condition=current; no wall-clock-success claim. Reconsider When: Author evaluator changes.
    References: run_realtime for actual asynchronous execution. ADR: None.
    """
    out.mkdir(parents=True,exist_ok=False);current=env.reset(init_state_index=index,seed=args.seed)
    current.update(tick=0,capture_time=time.monotonic());warm_once(worker,current,args.seed)
    current=env.reset(init_state_index=index,seed=args.seed);current.update(tick=0,capture_time=time.monotonic())
    write_json(out/'initial.json',{'seed':args.seed,'init_state_index':index,'environment':env.metadata,'metrics':current['metrics'],
                                 'sensor_hashes':{key:hashlib.sha256(np.asarray(current[key]).tobytes()).hexdigest()
                                                  for key in ('agentview_image','robot0_eye_in_hand_image',*PROPRIO)}})
    worker.reset(args.seed);worker.predict(current,0,current['capture_time']);first=worker.wait()
    queue=StateOriginQueue();queue.accept(first,0);chunks=[dict(first,bootstrap=True,ready_tick=0)]
    buffer=deque(maxlen=args.artificial_delay_ticks+1);ticks=[];writer=None;start=time.monotonic()
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(str(out/'rollout.mp4'),fps=args.control_hz,codec='libx264',macro_block_size=2)
    try:
        with (out/'ticks.jsonl').open('w') as stream:
            for tick in range(round(args.wall_seconds*args.control_hz)):
                buffer.append(current);sensor=buffer[0] if len(buffer)>args.artificial_delay_ticks else current
                if len(queue)==0:
                    mixed={key:np.asarray(sensor[key]).copy() for key in ('agentview_image','robot0_eye_in_hand_image')}
                    mixed.update(robot_only(current));mixed['prompt']=current['prompt']
                    worker.predict(mixed,int(sensor['tick']),sensor['capture_time'],request_tick=tick,measured_proprio_tick=tick,
                                   measured_proprio_time=current['capture_time'],state_origin_tick=tick,state_condition='current')
                    ready=worker.wait();event=queue.accept(ready,tick);chunks.append(dict(ready,bootstrap=False,ready_tick=tick,event=event))
                item=queue.pop();before=time.monotonic();current=env.step(item['action']);after=time.monotonic()
                current.update(tick=tick+1,capture_time=after)
                row={'tick':tick,'wall_time_since_start':before-start,'step_wall_s':after-before,'action':item['action'].tolist(),
                     'request_id':item['request_id'],'chunk_index':item['chunk_index'],'image_capture_tick':item.get('image_capture_tick'),
                     'state_origin_tick':item.get('state_origin_tick'),'artificial_delay_ticks':args.artificial_delay_ticks,
                     'held':item['held'],'success':bool(current['success']),'metrics':current['metrics']}
                ticks.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if writer:writer.append_data(np.concatenate([np.rot90(current['agentview_image'],2),np.rot90(current['robot0_eye_in_hand_image'],2)],axis=1))
                if current['success'] or current['done']:break
    finally:
        elapsed=time.monotonic()-start
        if writer:writer.close()
        write_json(out/'chunks.json',chunks)
    result={'status':'completed','error':None,'model':'vlash','mode':'artificial-lockstep','state_condition':'current',
            'init_state_index':index,'split':'development' if index==0 else 'formal','seed':args.seed,
            'success':bool(current['success']),'success_within_wall_budget':None,
            'ticks':len(ticks),'wall_seconds':elapsed,'sim_seconds':len(ticks)/args.control_hz,
            'realtime_factor':len(ticks)/args.control_hz/elapsed,'control_hz':args.control_hz,
            'sim_budget_s':args.wall_seconds,'native_generation_horizon':50,'prefetch_steps':0,
            'artificial_delay_ticks':args.artificial_delay_ticks,'artificial_delay_units':'environment loop steps, not declared training dataset samples',
            'hold_ticks':queue.buffer.holds,'hold_fraction':queue.buffer.holds/max(1,len(ticks)),
            'goal_xy_error_m':current['metrics'].get('goal_xy_error_m'),'final_metrics':current['metrics'],
            'ros_transport':True,'camera_mode':'synchronous','local_training':False,
            'configuration_note':'Author-style old image/current state, action_quant1, native H50 consumption, physics intentionally paused during inference; separate from real-time series'}
    write_json(out/'result.json',result);return result


def main():
    """Run only new VLASH series with explicit mode, source hashes and full frozen audit.

    Args: CLI flags. Returns: None. Raises: Config/environment/runtime/audit errors.
    Design: Context: Preserve all previous experiments. Concern: Accidental overwrite or pooled protocols.
        Decision: Exclusive output path and explicit artificial vs realtime modes.
        Alternatives: Append to old35/pi05 results. Quality Attributes: Reproducibility. Trade-offs: Separate summaries.
    Evidence: Internal: Config and all source SHA saved before loading.
        External: Native LIBERO success. Findings: Model loaded once per series.
        Metrics: Outcomes/transport/weights. Confidence: Fixed initial states only.
    Constraints: Launch .venv-vlash; exclusive resource lease; no old result modifications.
    Reconsider When: Model/backend changes. References: run_realtime; run_artificial. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mode',choices=['artificial','realtime'],required=True)
    parser.add_argument('--condition',choices=['coherent','current','predicted'],default='current')
    parser.add_argument('--artificial-delay-ticks',type=int,default=0)
    parser.add_argument('--modeldir',type=Path,default=Path('artifacts/realtime-vla/models/vlash'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--initial-states',nargs='+',type=int,default=[0])
    parser.add_argument('--seed',type=int,default=20261102);parser.add_argument('--control-hz',type=int,choices=[20],default=20)
    parser.add_argument('--wall-seconds',type=float,default=14.);parser.add_argument('--prefetch-steps',type=int,default=30)
    parser.add_argument('--max-sensor-age',type=float,default=.5);parser.add_argument('--camera-hz',type=float,default=10.)
    parser.add_argument('--ros-port',type=int,default=8780);parser.add_argument('--video',action='store_true');args=parser.parse_args()
    if not math.isfinite(args.wall_seconds) or args.wall_seconds<=0 or not 1<=args.prefetch_steps<50 or args.artificial_delay_ticks<0:
        parser.error('invalid timing/delay')
    if args.mode=='artificial' and args.condition!='current':parser.error('author artificial reproduction uses current state')
    if args.mode=='realtime' and args.artificial_delay_ticks:parser.error('real-time mode uses actual latency, not synthetic delay')
    for key,value in {'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
                      'GALLIUM_DRIVER':'d3d12','MESA_D3D12_DEFAULT_ADAPTER_NAME':'NVIDIA'}.items():os.environ.setdefault(key,value)
    if '/usr/lib/wsl/lib' not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        os.environ['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'config.json',vars(args))
    source=Path(__file__).parent;write_json(args.output/'source-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')})
    worker=None;env=None;camera=None;results=[]
    try:
        worker=VLASHWorker(args.modeldir,ros_port=args.ros_port)
        from libero_env import FrozenLiberoEnv
        from camera_worker import CameraWorker
        for index in args.initial_states:
            kwargs={'horizon':round(args.wall_seconds*args.control_hz),'control_freq':args.control_hz}
            env=FrozenLiberoEnv(**kwargs,output_dir=args.output/f'environment-init{index}',render_mode='sync' if args.mode=='artificial' else 'physics')
            write_json(args.output/f'controller-init{index}.json',verify_native_controller(env))
            if args.mode=='realtime':
                camera=CameraWorker(env_kwargs=kwargs,init_index=index,seed=args.seed,camera_hz=args.camera_hz)
                result=run_realtime(env,camera,worker,args,index,args.output/f'realtime-init{index}')
            else:result=run_artificial(env,worker,args,index,args.output/f'artificial-init{index}')
            results.append(result);write_json(args.output/'summary.json',{'status':'running','results':results})
            print(json.dumps({key:result[key] for key in ('mode','state_condition','init_state_index','success','hold_fraction','goal_xy_error_m')}),flush=True)
            if camera is not None:camera.close();camera=None
            env.close();env=None
            if result['status']!='completed':raise RuntimeError(result['error'])
        audit=worker.audit()
        if not audit['weights_unchanged'] or audit['requires_grad_parameter_count']!=0:raise RuntimeError('VLASH frozen audit failed')
        write_json(args.output/'model-audit.json',audit)
        write_json(args.output/'summary.json',{'status':'complete','results':results,'audit':audit,'worker_ready':worker.ready,'warmup_records':worker.warmup_records})
    except Exception as exc:
        write_json(args.output/'summary.json',{'status':'error','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),'results':results});raise
    finally:
        if camera is not None:camera.close()
        if env is not None:env.close()
        if worker is not None:worker.close()


if __name__=='__main__':main()
