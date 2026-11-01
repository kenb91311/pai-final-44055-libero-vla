"""Validate canonical snapshot sensor parity and nonblocking20Hz physics.

Args: None. Returns: JSON artifact. Raises: AssertionError on lost identity/parity.
Design: Context: Rendering exceeds physics budget. Concern: Changing visual inputs.
    Decision: Compare raw camera bytes across official scene clones and measure concurrent steps.
    Alternatives: Lower-resolution cameras. Quality Attributes: Compatibility.
    Trade-offs: Startup excluded, finite desktop timing sample.
Evidence: Internal: CameraWorker and FrozenLiberoEnv. External: LIBERO canonical snapshots.
    Findings: Recorded at execution. Metrics: Hash parity, tick periods, physicalstep p95.
    Confidence: Single-scene smoke.
Constraints: No model inference, no policy-visible snapshot.
Reconsider When: Scene assets change.
References: artifacts/realtime-vla/environment-smoke. ADR: None.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import hashlib,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.realtime_vla.libero_env import FrozenLiberoEnv
from experiments.realtime_vla.camera_worker import CameraWorker


def main():
    """Run same-state parity followed by a fixed five-second concurrent sensor trial.

    Args: None. Returns: None. Raises: Assertions/runtime errors.
    Design: Context: Visual semantics must match. Concern: Hidden scene drift.
        Decision: Exact byte hashes plus synchronized state assertions.
        Alternatives: Visual inspection only. Quality Attributes: Traceability.
        Trade-offs: One scene/state.
    Evidence: Internal: Recorded per-step times. External: None.
        Findings: Saved results. Metrics: Array equality and timing.
        Confidence: Integration smoke.
    Constraints: Explicit episode IDs. Reconsider When: IPC/scene changes.
    References: Module contract. ADR: None.
    """
    out=Path('artifacts/realtime-vla/environment-smoke');worker=None;env=None
    try:
        worker=CameraWorker(env_kwargs={'control_freq':20},init_index=0,seed=0,camera_hz=10)
        env=FrozenLiberoEnv(control_freq=20);env.reset(init_state_index=0,seed=0)
        snapshot=env.snapshot(episode_id='parity');expected=env.observe_snapshot(snapshot)
        worker.submit(snapshot);actual=worker.wait(timeout=60)
        parity={key:{'equal':bool(np.array_equal(expected[key],actual[key])),'expected_sha256':hashlib.sha256(expected[key].tobytes()).hexdigest(),'received_sha256':hashlib.sha256(actual[key].tobytes()).hexdigest(),'max_abs_difference':float(np.max(np.abs(expected[key].astype(np.float64)-actual[key].astype(np.float64))))} for key in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
        assert all(value['equal'] for value in parity.values()),parity
        assert actual['capture_time']==snapshot['capture_time'] and actual['episode_id']=='parity' and actual['tick']==0
        print('PARITY',json.dumps(parity),flush=True)
        env.close();env=FrozenLiberoEnv(control_freq=20,render_mode='physics');env.reset(init_state_index=0,seed=0)
        initial=env.snapshot(episode_id='timing');worker.submit(initial);worker.wait(timeout=60)
        steps=[];intervals=[];sensor_ages=[];sensor_ticks=[];submits=[];start=time.monotonic();previous=None
        with threadpool_limits(limits=1):
            for tick in range(100):
                deadline=start+tick*.05;delay=deadline-time.monotonic()
                if delay>0:time.sleep(delay)
                t=time.monotonic()
                if previous is not None:intervals.append(t-previous)
                previous=t;env.step(env.hold_action());steps.append(time.monotonic()-t)
                snap=env.snapshot(tick=tick+1,episode_id='timing');before=time.monotonic();worker.submit(snap);submits.append(time.monotonic()-before)
                sensor=worker.poll()
                if sensor is not None:
                    assert sensor['episode_id']=='timing';sensor_ages.append(time.monotonic()-sensor['capture_time']);sensor_ticks.append(sensor['tick'])
        elapsed=time.monotonic()-start
        result={'parity':parity,'renderer':worker.metadata,'startup_s':worker.startup_seconds,'physics_steps':100,'wall_seconds':elapsed,'sim_seconds':5.,'rtf':5./elapsed,'step_mean_ms':float(np.mean(steps)*1000),'step_p95_ms':float(np.percentile(steps,95)*1000),'step_max_ms':float(max(steps)*1000),'interval_p95_ms':float(np.percentile(intervals,95)*1000),'submit_max_ms':float(max(submits)*1000),'sensor_frames_received':len(sensor_ages),'sensor_age_mean_ms':float(np.mean(sensor_ages)*1000),'sensor_age_p95_ms':float(np.percentile(sensor_ages,95)*1000),'sensor_ticks':sensor_ticks,'accepted_submissions':worker.accepted,'dropped_submissions':worker.dropped}
        (out/'parallel-camera-result.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps(result),flush=True)
    finally:
        if env is not None:env.close()
        if worker is not None:worker.close()

if __name__=='__main__':main()
