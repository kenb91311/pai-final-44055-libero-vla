"""Measure full-horizon asynchronous chunk replacement with and without guided RTC.

Args: Frozen family, method variant, initial states and predeclared timing settings.
Returns: Immutable per-tick/chunk evidence, outcomes and weight/ROS audits.
Raises: Initialization/schema/runtime errors; failed episodes retain explicit status.
Design:
    Context: Original E10/P5 cannot cover measured generation delay.
    Concern: Scheduling improvements must not be attributed to RTC itself.
    Decision: Compare identical H50/aligned replacement/R30 with RTC off, stock and VJP.
    Alternatives: Compare a different horizon directly against original E10 as algorithm evidence.
    Quality Attributes: Causal comparison, traceability. Trade-offs: Separate experimental series.
Evidence:
    Internal: Original35 trials preserved; prefix phase unit tests.
    External: RTC paper2506.07339 and fixed LeRobot implementation.
    Findings: Sensor capture lag and inference delay both consume a chunk prefix.
    Metrics: Native success, time, holds, freshness, switch deltas, guidance cost and memory.
    Confidence: Per-state observations; small samples do not establish general superiority.
Constraints: Actual ROS roundtrip; one model; no learning, privileged policy input or blocking tick inference.
Reconsider When: Another action space or trained asynchronous checkpoint is introduced.
References: docs/reference/realtime-continuity-methods.md. ADR: None.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
from evaluate import write_json
from chunk_queue import ChunkQueue
from continuity_worker import ContinuityWorker


def run_episode(env,camera,worker,args,index,out):
    """Advance physics on real time and align every chunk to its captured sensor tick.

    Args: env/camera/worker: owned processes; args: fixed protocol; index/out: identity.
    Returns: Result mapping. Raises: Startup failures; loop errors recorded as failed trials.
    Design: Context: Guidance must match actual actuator execution, including holds/clipping.
        Concern: Prefix indexed at request time ignores older camera acquisition.
        Decision: Capture-origin executed history plus known queued future; drop actual elapsed prefix on arrival.
        Alternatives: Truncate H50 before dropping delay or blend commands without model conditioning.
        Quality Attributes: Fidelity, auditability. Trade-offs: Sensor age can reduce usable horizon.
    Evidence: Internal: ChunkQueue aligned semantics and per-tick actuator log.
        External: RTC published latency-prefix mechanism.
        Findings: Bootstrap is prepared while initial physics is held, outside task timing.
        Metrics: Native success and separately logged scheduler/guidance effects.
        Confidence: Recorded task/configuration only.
    Constraints: At most one inflight generation; tick loop never waits for a model/camera/ROS result.
    Reconsider When: Multiple inflight chunks are allowed.
    References: continuity_worker.py. ADR: None.
    """
    out.mkdir(parents=True,exist_ok=False);period=1./args.control_hz;max_ticks=round(args.wall_seconds*args.control_hz)
    episode_id=f'{args.model}-{args.variant}-init{index}'
    obs=env.reset(init_state_index=index,seed=args.seed)
    refreshed=time.monotonic();camera.submit(env.snapshot(tick=0,episode_id=episode_id));deadline=time.monotonic()+60
    while True:
        sensor=camera.wait(timeout=max(.1,deadline-time.monotonic()))
        if sensor['episode_id']==episode_id and sensor['capture_time']>=refreshed:break
        if time.monotonic()>deadline:raise TimeoutError('initial camera snapshot deadline')
    if not worker.warmed:
        for _ in range(2):
            worker.reset(args.seed);worker.predict(sensor,0,sensor['capture_time']);warm=worker.wait()
            worker.warmup_records.append({'metadata':warm['metadata'],'wall_seconds':warm['finished_time']-warm['started_time']})
        worker.warmed=True
    obs=env.reset(init_state_index=index,seed=args.seed)
    refreshed=time.monotonic();camera.submit(env.snapshot(tick=0,episode_id=episode_id));deadline=time.monotonic()+60
    while True:
        sensor=camera.wait(timeout=max(.1,deadline-time.monotonic()))
        if sensor['episode_id']==episode_id and sensor['capture_time']>=refreshed:break
        if time.monotonic()>deadline:raise TimeoutError('refreshed camera snapshot deadline')
    hashes={k:hashlib.sha256(np.asarray(sensor[k]).tobytes()).hexdigest() for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
    write_json(out/'initial.json',{'seed':args.seed,'init_state_index':index,'sensor_hashes':hashes,'metrics':obs['metrics'],'environment':env.metadata,'camera':camera.metadata})
    worker.reset(args.seed);worker.predict(sensor,0,sensor['capture_time']);provisional=worker.wait()
    calibrations=[]
    # Ordinary warmups exercise no guidance. Exercise the actual selected method
    # twice on a held initial scene before estimating its first online delay.
    for _ in range(2):
        estimate=min(49,math.ceil((calibrations[-1]['wall_seconds'] if calibrations else worker.warmup_records[-1]['wall_seconds'])/period))
        worker.predict(sensor,0,sensor['capture_time'],prefix_actions=np.clip(provisional['actions'],-1,1),
                       inference_delay=estimate,execution_horizon=args.execution_horizon,request_tick=0)
        measured=worker.wait();calibrations.append({'wall_seconds':measured['finished_time']-measured['started_time'],'metadata':measured['metadata']})
    write_json(out/'guided-calibration.json',calibrations)
    worker.reset(args.seed);worker.predict(sensor,0,sensor['capture_time']);first=worker.wait()
    buffer=ChunkQueue(50,'aligned');buffer.accept(first['actions'],0,first['observation_time'],0,first['request_id'])
    chunks=[dict(first,bootstrap=True,applied=True,ready_tick=0)];ticks=[];history=[];switches=[];latencies=[r['wall_seconds'] for r in calibrations]
    writer=None
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(str(out/'rollout.mp4'),fps=args.control_hz,codec='libx264',macro_block_size=2)
    start=time.monotonic();next_tick=start;previous_tick=None;previous_action=None;status='completed';error=None;frames=1;stale=0
    try:
        with (out/'ticks.jsonl').open('w') as stream:
            for tick in range(max_ticks):
                delay=next_tick-time.monotonic()
                if delay>0:time.sleep(delay)
                if time.monotonic()-start>=args.wall_seconds:break
                new=camera.poll()
                if new is not None and new['episode_id']==episode_id and new['tick']>=sensor['tick']:
                    sensor=new;frames+=1
                now=time.monotonic();interval=None if previous_tick is None else now-previous_tick;previous_tick=now
                ready=worker.poll() if worker.inflight else None;changed=False
                if ready is not None:
                    actual=tick-ready['observation_tick'];count=buffer.accept(ready['actions'],ready['observation_tick'],ready['observation_time'],tick,ready['request_id'])
                    chunks.append(dict(ready,bootstrap=False,applied=count>0,ready_tick=tick,actual_delay_ticks=actual,queued_actions=count))
                    latencies.append(ready['finished_time']-ready['started_time']);changed=count>0
                if len(buffer)<=args.prefetch_steps and not worker.inflight:
                    if now-sensor['capture_time']<=args.max_sensor_age:
                        capture=int(sensor['tick'])
                        if capture>tick or capture<0:raise ValueError('noncausal camera tick')
                        # history[t] is the exact clipped physical command already applied at tick t.
                        past=np.asarray(history[capture:tick],dtype=np.float32).reshape(-1,7)
                        future=np.asarray([x['action'] for x in buffer.queue],dtype=np.float32).reshape(-1,7)
                        prefix=np.concatenate([past,future],axis=0)[:50]
                        estimate=math.ceil(float(np.percentile(latencies[-10:],95))/period)
                        raw_predicted=tick-capture+estimate;predicted=min(49,raw_predicted)
                        horizon=min(args.execution_horizon,len(prefix),50)
                        worker.predict(sensor,capture,sensor['capture_time'],prefix_actions=prefix if len(prefix) else None,
                                       inference_delay=predicted,execution_horizon=horizon,request_tick=tick,raw_predicted_delay=raw_predicted)
                    else:stale+=1
                depth=len(buffer);item=buffer.pop();action=item['action'];history.append(action.copy())
                before=time.monotonic();obs=env.step(action);after=time.monotonic()
                camera.submit(env.snapshot(tick=tick+1,episode_id=episode_id))
                delta=None if previous_action is None else action-previous_action
                record={'tick':tick,'wall_time_since_start':before-start,'tick_interval_s':interval,'deadline_lateness_s':max(0.,now-next_tick),
                        'step_wall_s':after-before,'queue_before':depth,'queue_after':len(buffer),'held':item['held'],
                        'request_id':item['request_id'],'chunk_index':item['chunk_index'],'observation_tick':item['observation_tick'],
                        'observation_age_s':None if item['observation_time'] is None else before-item['observation_time'],
                        'camera_capture_tick':sensor['tick'],'camera_age_s':before-sensor['capture_time'],'action':action.tolist(),
                        'action_delta_l2':None if delta is None else float(np.linalg.norm(delta)),
                        'translation_delta_l2':None if delta is None else float(np.linalg.norm(delta[:3])),
                        'rotation_delta_l2':None if delta is None else float(np.linalg.norm(delta[3:6])),
                        'gripper_delta_abs':None if delta is None else float(abs(delta[6])),
                        'chunk_switched':changed,'success':bool(obs['success']),'metrics':obs['metrics']}
                if changed and delta is not None:switches.append({k:record[k] for k in ('tick','translation_delta_l2','rotation_delta_l2','gripper_delta_abs')})
                ticks.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();previous_action=action.copy()
                if writer:writer.append_data(np.concatenate([np.rot90(sensor['agentview_image'],2),np.rot90(sensor['robot0_eye_in_hand_image'],2)],axis=1))
                next_tick=max(next_tick+period,after)
                if obs['success'] or obs['done']:break
    except Exception as exc:
        status='error';error=f'{type(exc).__name__}: {exc}';write_json(out/'error.json',{'error':error,'traceback':traceback.format_exc()})
    finally:
        elapsed=time.monotonic()-start
        if writer:writer.close()
        if worker.inflight:
            try:chunks.append(dict(worker.wait(),bootstrap=False,applied=False,completed_after_episode=True))
            except Exception as exc:status='error';error=f'drain failure: {exc}'
        write_json(out/'chunks.json',chunks)
    intervals=[t['tick_interval_s'] for t in ticks if t['tick_interval_s'] is not None]
    steady=[c for c in chunks if not c.get('bootstrap') and 'finished_time' in c]
    ages=[t['observation_age_s'] for t in ticks if t['observation_age_s'] is not None]
    result={'status':status,'error':error,'model':args.model,'variant':args.variant,'mode':'realtime','init_state_index':index,'seed':args.seed,
            'success':bool(obs['success']) and status=='completed','success_within_wall_budget':bool(obs['success']) and status=='completed' and elapsed<=args.wall_seconds+period,
            'ticks':len(ticks),'wall_seconds':elapsed,'sim_seconds':len(ticks)*period,'realtime_factor':len(ticks)*period/elapsed,
            'control_hz':args.control_hz,'wall_budget_s':args.wall_seconds,'native_generation_horizon':50,'execute_steps':50,'merge':'aligned',
            'prefetch_steps':args.prefetch_steps,'rtc_execution_horizon':args.execution_horizon,'hold_ticks':buffer.holds,'hold_fraction':buffer.holds/max(1,len(ticks)),
            'expired_prefix_actions':buffer.discarded_expired,'replaced_actions':buffer.discarded_overlap,
            'mean_control_interval_s':float(np.mean(intervals)) if intervals else None,'p95_control_interval_s':float(np.percentile(intervals,95)) if intervals else None,
            'mean_chunk_inference_s':float(np.mean([c['finished_time']-c['started_time'] for c in steady])) if steady else None,
            'mean_observation_age_s':float(np.mean(ages)) if ages else None,'sensor_frames_received':frames,'stale_sensor_ticks':stale,
            'goal_xy_error_m':obs['metrics'].get('goal_xy_error_m'),'final_metrics':obs['metrics'],'switches':switches,
            'mean_switch_translation_delta_l2':float(np.mean([s['translation_delta_l2'] for s in switches])) if switches else None,
            'mean_switch_rotation_delta_l2':float(np.mean([s['rotation_delta_l2'] for s in switches])) if switches else None,
            'mean_switch_gripper_delta_abs':float(np.mean([s['gripper_delta_abs'] for s in switches])) if switches else None,
            'ros_transport':True,'camera_mode':'asynchronous_clone','camera_hz':args.camera_hz,
            'configuration_note':'H50/R30 aligned replacement, capture-origin executed+future prefix, p95 rolling inference-delay prediction; warm2, guided calibration2 and final reset/bootstrap excluded; new series, not original E10 baseline'}
    write_json(out/'result.json',result);return result


def main():
    """Run one method/model at a time, preserving original and failed evidence.

    Args: CLI. Returns: None. Raises: Asset, ROS, environment or runtime errors.
    Design: Context: Serial GPU comparisons on RTX5080.
        Concern: Asset or timing changes could overwrite prior evidence.
        Decision: Exclusive output directory and source hashes before startup.
        Alternatives: Rewrite original35 artifacts. Quality Attributes: Reproducibility.
        Trade-offs: One initialization per method.
    Evidence: Internal: Per-run frozen audit. External: LIBERO native success.
        Findings: No statistical generalization from five fixed initial states.
        Metrics: Outcomes/periods/prefix ages. Confidence: Empirical.
    Constraints: User controls experimental protocol, not hidden task correction.
    Reconsider When: Concurrent resource contention is the objective.
    References: run_episode. ADR: None.
    """
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--model',choices=['smolvla','pi05'],required=True)
    p.add_argument('--variant',choices=['off','rtc-stock','rtc-vjp'],required=True);p.add_argument('--modeldir',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--initial-states',nargs='+',type=int,default=[0]);p.add_argument('--seed',type=int,default=20261102)
    p.add_argument('--control-hz',type=int,choices=[20],default=20);p.add_argument('--wall-seconds',type=float,default=14.)
    p.add_argument('--prefetch-steps',type=int,default=30);p.add_argument('--execution-horizon',type=int,default=30)
    p.add_argument('--max-sensor-age',type=float,default=.5);p.add_argument('--camera-hz',type=float,default=10.);p.add_argument('--ros-port',type=int,default=8780);p.add_argument('--video',action='store_true')
    args=p.parse_args()
    if not math.isfinite(args.wall_seconds) or args.wall_seconds<=0 or not 1<=args.prefetch_steps<50 or not 1<=args.execution_horizon<=50:p.error('invalid timing/horizon')
    for key,val in {'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','GALLIUM_DRIVER':'d3d12','MESA_D3D12_DEFAULT_ADAPTER_NAME':'NVIDIA'}.items():os.environ.setdefault(key,val)
    if '/usr/lib/wsl/lib' not in os.environ.get('LD_LIBRARY_PATH','').split(':'):os.environ['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'config.json',vars(args))
    source_dir=Path(__file__).parent
    write_json(args.output/'source-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.glob('*.py')})
    worker=None;env=None;camera=None;results=[]
    try:
        worker=ContinuityWorker(args.model,args.modeldir or Path('artifacts/realtime-vla/models')/args.model,args.variant,ros_port=args.ros_port)
        from libero_env import FrozenLiberoEnv
        from camera_worker import CameraWorker
        for index in args.initial_states:
            kw={'horizon':round(args.wall_seconds*args.control_hz),'control_freq':args.control_hz}
            env=FrozenLiberoEnv(**kw,output_dir=args.output/f'environment-init{index}',render_mode='physics')
            camera=CameraWorker(env_kwargs=kw,init_index=index,seed=args.seed,camera_hz=args.camera_hz)
            result=run_episode(env,camera,worker,args,index,args.output/f'realtime-init{index}');results.append(result)
            write_json(args.output/'summary.json',{'status':'running','results':results});print(json.dumps({k:result[k] for k in ('model','variant','init_state_index','status','success','hold_fraction','goal_xy_error_m')}),flush=True)
            camera.close();camera=None;env.close();env=None
            if result['status']!='completed':raise RuntimeError(result['error'])
        audit=worker.audit()
        if not audit['weights_unchanged']:raise RuntimeError('model weights changed')
        write_json(args.output/'model-audit.json',audit);write_json(args.output/'summary.json',{'status':'complete','results':results,'audit':audit,'worker_ready':worker.ready,'warmup_records':worker.warmup_records})
    except Exception as exc:
        write_json(args.output/'summary.json',{'status':'error','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),'results':results});raise
    finally:
        if camera is not None:camera.close()
        if env is not None:env.close()
        if worker is not None:worker.close()


if __name__=='__main__':main()
