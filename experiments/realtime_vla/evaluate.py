"""Compare frozen VLA policies with lockstep and wallclock-driven chunk execution.

Args: CLI model, initial states, control frequency, chunk prefix and prefetch threshold.
Returns: Per-tick/chunk records, physical outcomes, timing and frozen-state audit.
Raises: RuntimeError on policy/environment failure; incomplete runs retain explicit errors.
Design:
    Context: Small-model latency may matter more than static policy quality.
    Concern: Waiting for inference in the control loop invalidates a realtime experiment.
    Decision: A separate spawned GPU worker, nonblocking polls and shared queue rules.
    Alternatives: Prior paused-simulator rollout is retained only as lockstep reference.
    Quality Attributes: Comparability, observability. Trade-offs: Soft realtime on a desktop OS.
Evidence:
    Internal: ChunkQueue and ModelWorker have explicit held/stale metadata.
    External: Official LIBERO task and LeRobot family preprocessing.
    Findings: Goal error, controller error and action smoothness are distinct metrics.
    Metrics: Success, goal XY error, observation age, holds, periods and realtime factor.
    Confidence: Restricted to measured model/task/configuration.
Constraints: Same clock, queue and task for both models; no learning or privileged model inputs.
Reconsider When: Real hardware, model-specific RTC or another control space is introduced.
References: docs/architecture/realtime-vla.md.
ADR: None.
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
from chunk_queue import ChunkQueue
from worker import ModelWorker


def write_json(path,value):
    """Persist finite scalar/array records without discarding failures.

    Args: path/value: artifact and JSON data. Returns: None. Raises: IO/ValueError.
    Design:
        Context: Numeric evidence must survive interrupted evaluation.
        Concern: Nonfinite values cannot be silently encoded as valid measurements.
        Decision: Strict JSON; numpy arrays become lists.
        Alternatives: Console-only reports are hard to audit.
        Quality Attributes: Traceability. Trade-offs: Full files are rewritten at episode boundaries.
    Evidence:
        Internal: Per-tick logs retain timing. External: JSON numeric specification.
        Findings: Failed episodes keep separate result status.
        Metrics: None. Confidence: IO contract.
    Constraints: No raw camera data in numeric logs.
    Reconsider When: Streaming database storage is used.
    References: evaluate.py.
    ADR: None.
    """
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')


def episode(env,worker,args,mode,index,out,camera=None):
    """Run a paired initial state under a declared schedule without hidden control waits.

    Args: env/worker: owned instances; args: controls; mode: lockstep or realtime;
        index: official initial state index; out: new artifact directory.
    Returns: Measured episode result.
    Raises: RuntimeError on invalid inference or physical error, retaining a failure record.
    Design:
        Context: The paused reference measures policy competence; realtime adds latency.
        Concern: Empty buffers and delayed chunks must affect actual physics.
        Decision: Preload first chunk, then hold zero pose delta during late inference.
        Alternatives: Freezing physics or repeating previous deltas invalidates the comparison.
        Quality Attributes: Fairness, auditability. Trade-offs: Warm startup is excluded and reported.
    Evidence:
        Internal: Every executed action has source time, tick and request ID.
        External: LIBERO relative OSC control.
        Findings: Real clock and simulated clock may diverge if the simulator overruns.
        Metrics: Wall budget, realtime factor, holds and native task success.
        Confidence: Functional timing evidence, not hard realtime certification.
    Constraints: One in-flight prediction; no task-specific corrective controller.
    Reconsider When: Model-conditioned chunk stitching is evaluated.
    References: ChunkQueue.
    ADR: None.
    """
    out.mkdir(parents=True,exist_ok=False)
    period=1./args.control_hz;max_ticks=int(round(args.wall_seconds*args.control_hz))
    obs=env.reset(init_state_index=index,seed=args.seed)
    obs_capture_time=time.monotonic();episode_id=f'{mode}-init{index}';sensor_frames=0;stale_sensor_ticks=0
    latest_sensor=obs
    if camera is not None:
        camera.submit(env.snapshot(tick=0,episode_id=episode_id))
        deadline=time.monotonic()+60
        while True:
            latest_sensor=camera.wait(timeout=max(.1,deadline-time.monotonic()))
            if latest_sensor['episode_id']==episode_id:break
            if time.monotonic()>deadline:raise TimeoutError('initial sensor identity deadline')
        obs_capture_time=latest_sensor['capture_time'];sensor_frames+=1
    initial_sensing={k:hashlib.sha256(np.asarray(latest_sensor[k]).tobytes()).hexdigest() for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
    write_json(out/'initial.json',{'init_state_index':index,'seed':args.seed,'sensor_hashes':initial_sensing,'metrics':obs['metrics'],'environment':env.metadata,'camera':camera.metadata if camera else None})
    if not worker.warmed:
        for warm_index in range(2):
            worker.reset(args.seed);worker.predict(latest_sensor,int(latest_sensor.get('tick',0)),obs_capture_time)
            warm=worker.wait();worker.warmup_records.append({'metadata':warm['metadata'],'wall_seconds':warm['finished_time']-warm['started_time']})
        worker.warmed=True
    # Refresh both families after untimed warmup; startup age still includes the
    # measured first prediction, but never the model-dependent warmup duration.
    obs=env.reset(init_state_index=index,seed=args.seed)
    latest_sensor=obs;obs_capture_time=time.monotonic()
    if camera is not None:
        refresh_started=time.monotonic()
        camera.submit(env.snapshot(tick=0,episode_id=episode_id))
        deadline=time.monotonic()+60
        while True:
            latest_sensor=camera.wait(timeout=max(.1,deadline-time.monotonic()))
            if latest_sensor['episode_id']==episode_id and latest_sensor['capture_time']>=refresh_started:break
            if time.monotonic()>deadline:raise TimeoutError('refreshed sensor identity deadline')
        obs_capture_time=latest_sensor['capture_time'];sensor_frames+=1
    initial_sensing={k:hashlib.sha256(np.asarray(latest_sensor[k]).tobytes()).hexdigest() for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
    write_json(out/'initial.json',{'init_state_index':index,'seed':args.seed,'sensor_hashes':initial_sensing,'metrics':obs['metrics'],'environment':env.metadata,'camera':camera.metadata if camera else None})
    worker.reset(args.seed)
    buffer=ChunkQueue(args.execute_steps,args.merge)
    first_request_time=time.monotonic();worker.predict(latest_sensor,int(latest_sensor.get('tick',0)),obs_capture_time)
    first=worker.wait();initial_chunk_seconds=time.monotonic()-first_request_time
    buffer.accept(first['actions'],0,first['observation_time'],0,first['request_id'])
    chunk_records=[dict(first,bootstrap=True,applied=True)]
    writer=None;ticks=[];status='completed';error=None
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(str(out/'rollout.mp4'),fps=args.control_hz,codec='libx264',macro_block_size=2)
    start=time.monotonic();next_tick=start;previous_tick=None;previous_action=None
    # Preserve the real bootstrap sensor timestamp; startup is excluded only from task budget.
    try:
        with (out/'ticks.jsonl').open('w') as stream:
            for tick in range(max_ticks):
                if mode=='realtime':
                    delay=next_tick-time.monotonic()
                    if delay>0:time.sleep(delay)
                    if time.monotonic()-start>=args.wall_seconds:break
                if camera is not None:
                    sensed=camera.poll()
                    if sensed is not None and sensed['episode_id']==episode_id and sensed['tick']>=latest_sensor['tick']:
                        latest_sensor=sensed;obs_capture_time=sensed['capture_time'];sensor_frames+=1
                tick_time=time.monotonic();interval=None if previous_tick is None else tick_time-previous_tick;previous_tick=tick_time
                ready=worker.poll() if worker.inflight else None
                if ready is not None:
                    if ready['kind']!='chunk':raise RuntimeError('unexpected worker response')
                    added=buffer.accept(ready['actions'],ready['observation_tick'],ready['observation_time'],tick,ready['request_id'])
                    chunk_records.append(dict(ready,bootstrap=False,applied=bool(added),ready_tick=tick,queued_actions=added))
                if mode=='lockstep' and len(buffer)==0:
                    worker.predict(latest_sensor,int(latest_sensor.get('tick',tick)),obs_capture_time);ready=worker.wait()
                    added=buffer.accept(ready['actions'],tick,obs_capture_time,tick,ready['request_id'])
                    chunk_records.append(dict(ready,bootstrap=False,applied=True,ready_tick=tick,queued_actions=added))
                elif mode=='realtime' and len(buffer)<=args.prefetch_steps and not worker.inflight:
                    if time.monotonic()-obs_capture_time<=args.max_sensor_age:
                        worker.predict(latest_sensor,int(latest_sensor.get('tick',tick)),obs_capture_time)
                    else:stale_sensor_ticks+=1
                queue_before=len(buffer);item=buffer.pop();action=item['action']
                before=time.monotonic();camera_age_at_action=max(0.,before-obs_capture_time);camera_capture_tick=latest_sensor.get('tick',tick)
                obs=env.step(action);after=time.monotonic()
                if camera is None:
                    latest_sensor=obs;obs_capture_time=after
                else:camera.submit(env.snapshot(tick=tick+1,episode_id=episode_id))
                age=None if item['observation_time'] is None else before-item['observation_time']
                record={'tick':tick,'wall_time_since_start':before-start,'tick_interval_s':interval,
                        'deadline_lateness_s':max(0.,tick_time-next_tick) if mode=='realtime' else None,
                        'step_wall_s':after-before,'queue_before':queue_before,'queue_after':len(buffer),
                        'held':item['held'],'request_id':item['request_id'],'chunk_index':item['chunk_index'],
                        'observation_tick':item['observation_tick'],'observation_age_s':age,
                        'action':action.tolist(),'action_delta_l2':None if previous_action is None else float(np.linalg.norm(action-previous_action)),
                        'success':bool(obs['success']),'metrics':obs['metrics'],'camera_capture_tick':camera_capture_tick,'camera_age_s':camera_age_at_action}
                previous_action=action.copy();ticks.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                if writer:
                    # Display camera orientation matches the official model preprocessor.
                    image=np.concatenate([np.rot90(latest_sensor['agentview_image'],2),np.rot90(latest_sensor['robot0_eye_in_hand_image'],2)],axis=1)
                    writer.append_data(image)
                next_tick=max(next_tick+period,after)
                if obs['success'] or obs['done']:break
    except Exception as exc:
        status='error';error=f'{type(exc).__name__}: {exc}'
        write_json(out/'error.json',{'error':error,'traceback':traceback.format_exc()})
    finally:
        elapsed=time.monotonic()-start
        if writer:writer.close()
        # Drain only after task timing ends; this chunk cannot improve the completed trial.
        if worker.inflight:
            try:
                late=worker.wait();chunk_records.append(dict(late,bootstrap=False,applied=False,completed_after_episode=True))
            except Exception as exc:
                status='error';error=f'drain failure: {type(exc).__name__}: {exc}'
                chunk_records.append({'kind':'error','error':error,'applied':False,'completed_after_episode':True})
        write_json(out/'chunks.json',chunk_records)
    intervals=[t['tick_interval_s'] for t in ticks if t['tick_interval_s'] is not None]
    ages=[t['observation_age_s'] for t in ticks if t['observation_age_s'] is not None]
    inference=[c['finished_time']-c['started_time'] for c in chunk_records if not c.get('bootstrap') and 'finished_time' in c and 'started_time' in c]
    steady_ages=[t['observation_age_s'] for t in ticks if t['observation_age_s'] is not None and t['request_id']!=first['request_id']]
    sim_seconds=len(ticks)*period
    success=bool(obs['success']) and status=='completed'
    within_budget=None if mode=='lockstep' else success and elapsed<=args.wall_seconds+period
    result={'status':status,'error':error,'model':args.model,'mode':mode,'init_state_index':index,'seed':args.seed,
            'success':success,'success_within_wall_budget':within_budget,'control_hz':args.control_hz,'wall_budget_s':args.wall_seconds,
            'execute_steps':args.execute_steps,'native_generation_horizon':len(first['actions']),'merge':args.merge,'prefetch_steps':args.prefetch_steps,
            'ticks':len(ticks),'policy_actions':sum(not t['held'] for t in ticks),'hold_ticks':buffer.holds,
            'hold_fraction':buffer.holds/max(1,len(ticks)),'wall_seconds':elapsed,'sim_seconds':sim_seconds,
            'realtime_factor':sim_seconds/elapsed if elapsed>0 else None,'startup_initial_chunk_seconds':initial_chunk_seconds,
            'goal_xy_error_m':obs['metrics'].get('goal_xy_error_m'),'final_metrics':obs['metrics'],
            'mean_control_interval_s':float(np.mean(intervals)) if intervals else None,'p95_control_interval_s':float(np.percentile(intervals,95)) if intervals else None,
            'overrun_period_count':sum(v>period*1.1 for v in intervals),
            'mean_observation_age_s':float(np.mean(ages)) if ages else None,'p95_observation_age_s':float(np.percentile(ages,95)) if ages else None,
            'mean_steady_observation_age_s':float(np.mean(steady_ages)) if steady_ages else None,'p95_steady_observation_age_s':float(np.percentile(steady_ages,95)) if steady_ages else None,
            'mean_chunk_inference_s':float(np.mean(inference)) if inference else None,'p95_chunk_inference_s':float(np.percentile(inference,95)) if inference else None,
            'expired_prefix_actions':buffer.discarded_expired,'replaced_actions':buffer.discarded_overlap,
            'configuration_note':'common executed prefix; two untimed warmup calls and first chunk before task clock; physics continues during asynchronous inference; no catch-up bursts',
            'camera_mode':'asynchronous_clone' if camera else 'synchronous','camera_hz':args.camera_hz if camera else None,
            'sensor_frames_received':sensor_frames,'stale_sensor_ticks':stale_sensor_ticks,'ros_transport':args.ros_transport,
            'video_time_note':'raw video uses simulated tick rate; wall timestamps are in ticks.jsonl; realtime factor must be reported'}
    write_json(out/'result.json',result)
    return result


def main():
    """Evaluate one model at a time with identical environment and scheduler settings.

    Args: CLI --model smolvla/pi0fast/pi05 plus fixed run settings.
    Returns: None; writes immutable run directory and audited summary.
    Raises: Initialization failures before episode execution.
    Design:
        Context: RTX5080 memory and resource contention affect latency.
        Concern: Simultaneous model serving makes speed comparison unreliable.
        Decision: One model worker per run, explicit shutdown and fixed initial states.
        Alternatives: Co-loading two models changes memory/compute pressure.
        Quality Attributes: Comparability. Trade-offs: Sequential benchmark duration.
    Evidence:
        Internal: ModelWorker owns its child process.
        External: LIBERO initial-state assets.
        Findings: Different task frequencies are separate declared conditions.
        Metrics: Success, latency, memory and weights. Confidence: Audited empirical evidence.
    Constraints: No model training, no hidden task-state features.
    Reconsider When: Multi-model deployment contention is the study objective.
    References: docs/architecture/realtime-vla.md.
    ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model',choices=['smolvla','pi0fast','pi05'],required=True)
    parser.add_argument('--modeldir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-states',type=int,nargs='+',default=[0])
    parser.add_argument('--conditions',nargs='+',choices=['lockstep','realtime'],default=['lockstep','realtime'])
    parser.add_argument('--control-hz',type=int,choices=[10,20],default=20)
    parser.add_argument('--wall-seconds',type=float,default=14.)
    parser.add_argument('--execute-steps',type=int,default=10)
    parser.add_argument('--prefetch-steps',type=int,default=5)
    parser.add_argument('--merge',choices=['fifo','aligned'],default='fifo')
    parser.add_argument('--seed',type=int,default=20261102)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--camera-hz',type=float,default=10.)
    parser.add_argument('--max-sensor-age',type=float,default=.5)
    parser.add_argument('--sync-camera',action='store_true',help='diagnostic: render cameras synchronously even in realtime')
    parser.add_argument('--ros-transport',action='store_true')
    parser.add_argument('--ros-port',type=int,default=8780)
    args=parser.parse_args()
    if args.execute_steps<1 or not 0<=args.prefetch_steps<=args.execute_steps or not math.isfinite(args.wall_seconds) or args.wall_seconds<=0:
        parser.error('invalid chunk prefix, prefetch or wall budget')
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl');os.environ.setdefault('OMP_NUM_THREADS','4')
    # WSL's default Mesa fallback is CPU llvmpipe; select the measured NVIDIA D3D12 path.
    os.environ.setdefault('GALLIUM_DRIVER','d3d12');os.environ.setdefault('MESA_D3D12_DEFAULT_ADAPTER_NAME','NVIDIA')
    if '/usr/lib/wsl/lib' not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        os.environ['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'config.json',vars(args))
    worker=None;env=None;camera=None;results=[];environment_records=[]
    try:
        worker=ModelWorker(args.model,args.modeldir or Path('artifacts/realtime-vla/models')/args.model,ros_transport=args.ros_transport,ros_port=args.ros_port)
        from libero_env import FrozenLiberoEnv
        for index in args.initial_states:
            for mode in args.conditions:
                kwargs={'horizon':int(round(args.wall_seconds*args.control_hz)),'control_freq':args.control_hz}
                env=FrozenLiberoEnv(**kwargs,output_dir=args.output/f'environment-{mode}-init{index}',render_mode='physics' if mode=='realtime' and not args.sync_camera else 'sync')
                if mode=='realtime' and not args.sync_camera:
                    from camera_worker import CameraWorker
                    camera=CameraWorker(env_kwargs=kwargs,init_index=index,seed=args.seed,camera_hz=args.camera_hz)
                environment_records.append({'mode':mode,'init_index':index,'physics':env.metadata,'camera':camera.metadata if camera else None})
                result=episode(env,worker,args,mode,index,args.output/f'{mode}-init{index}',camera=camera)
                results.append(result);print(json.dumps({k:result[k] for k in ('model','mode','init_state_index','success','ticks','hold_fraction','realtime_factor','goal_xy_error_m')}),flush=True)
                write_json(args.output/'summary.json',{'status':'running','results':results})
                if camera is not None:camera.close();camera=None
                env.close();env=None
        audit=worker.audit();assert audit['weights_unchanged'],audit
        write_json(args.output/'model-audit.json',audit)
        write_json(args.output/'summary.json',{'status':'complete','results':results,'audit':audit,'worker_ready':worker.ready,'environments':environment_records,'warmup_records':worker.warmup_records})
    except Exception as exc:
        write_json(args.output/'summary.json',{'status':'error','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),'results':results})
        raise
    finally:
        if camera is not None:camera.close()
        if env is not None:env.close()
        if worker is not None:worker.close()


if __name__=='__main__':main()
