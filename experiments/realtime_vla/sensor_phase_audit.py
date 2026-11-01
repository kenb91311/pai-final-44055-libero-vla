"""Quantify cached-step versus forced-snapshot sensing after all formal GPU trials.

Args: Fixed LIBERO initial-state index, seed and report output directory.
Returns: Sensor-difference JSON, two-camera comparison plot and raw observation NPZ files.
Raises: AssertionError if state restoration changes physical state/control; startup errors propagate.
Design:
    Context: Formal cross-model inputs match within each mode, but lockstep/RT inputs differ.
    Concern: Regenerated scene-clone parity does not prove equality to cached step observations.
    Decision: Compare cached reset sensors to forced sensing from the exact same snapshot,
        verifying MuJoCo state and actuator commands remain unchanged.
    Alternatives: Re-running or changing the formal benchmark would alter its fixed protocol.
    Quality Attributes: Causal clarity, traceability. Trade-offs: One diagnostic state cannot
        estimate the task-success effect of a sampling-phase difference.
Evidence:
    Internal: initial.json audits and environment-smoke clone byte parity.
    External: robosuite Observable.update sampling cache and LIBERO force-update regeneration.
    Findings: Report actual pixel/proprio differences; do not assume their magnitude or impact.
    Metrics: Pixel change fraction/MAE, EEF position/quaternion angle, gripper difference.
    Confidence: Single-state mechanical diagnostic only.
Constraints: Run only after root confirms every GPU evaluation has ended; no model/training.
Reconsider When: A matched sensor-phase experiment is preregistered separately.
References: summarize.py initial_condition_checks; libero_env.py.
ADR: None.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SENSORS=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def compare_sensors(cached,regenerated):
    """Measure image and robot-sensor differences without changing either input.

    Args: cached/regenerated: raw sensing dictionaries from one physical snapshot.
    Returns: Per-key hashes, pixel statistics and physically interpretable proprio deltas.
    Raises: ValueError for incompatible shape/dtype or nonfinite proprioception.
    Design: Context: A tiny difference changes SHA256 completely.
        Concern: Hash inequality alone says nothing about error magnitude.
        Decision: Pair exact byte checks with pixel/proprio norms and quaternion angle.
        Alternatives: Inferring large observation error from unequal hashes.
        Quality Attributes: Interpretability. Trade-offs: Does not measure learned-policy sensitivity.
    Evidence: Internal: Native camera/state arrays.
        External: Quaternion geodesic angle from normalized absolute dot product.
        Findings: Calculated at execution. Metrics: Pixel MAE and geometric deltas.
        Confidence: Direct deterministic arithmetic.
    Constraints: Quaternion signs are equivalent for angular comparison.
    Reconsider When: Sensor representation changes.
    References: main. ADR: None.
    """
    records={}
    for key in SENSORS:
        a,b=np.asarray(cached[key]),np.asarray(regenerated[key])
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('incompatible sensor arrays')
        difference=np.abs(a.astype(np.float64)-b.astype(np.float64))
        record={'shape':list(a.shape),'dtype':str(a.dtype),'byte_equal':a.tobytes()==b.tobytes(),'cached_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'regenerated_sha256':hashlib.sha256(b.tobytes()).hexdigest(),'max_abs_difference':float(difference.max()),'mean_abs_difference':float(difference.mean())}
        if a.ndim==3:
            changed=np.any(a!=b,axis=2)
            record.update(changed_pixels=int(changed.sum()),total_pixels=int(changed.size),changed_pixel_fraction=float(changed.mean()),mean_abs_channel_difference_0_255=float(difference.mean()),rmse_channel_0_255=float(np.sqrt(np.mean(difference*difference))))
        else:record.update(cached=a.tolist(),regenerated=b.tolist(),l2_difference=float(np.linalg.norm(a-b)))
        records[key]=record
    qa=np.asarray(cached['robot0_eef_quat'],dtype=np.float64);qb=np.asarray(regenerated['robot0_eef_quat'],dtype=np.float64)
    dot=abs(float(np.dot(qa,qb)/(np.linalg.norm(qa)*np.linalg.norm(qb))))
    return {'sensors':records,'eef_position_l2_m':records['robot0_eef_pos']['l2_difference'],'eef_quaternion_angle_rad':2*math.acos(min(1.,max(-1.,dot))),'gripper_position_l2_m':records['robot0_gripper_qpos']['l2_difference'],'all_sensor_bytes_equal':all(r['byte_equal'] for r in records.values())}


def plot_difference(cached,regenerated,output):
    """Show original, regenerated and actual per-pixel difference for both cameras.

    Args: cached/regenerated: raw arrays; output: report directory.
    Returns: Plot filenames. Raises: Plot/IO errors.
    Design: Context: Hash mismatch needs a readable spatial explanation.
        Concern: Amplified heatmaps can appear to be changed task geometry.
        Decision: Keep camera views intact and label a separate mean-channel difference scale.
        Alternatives: Unlabelled amplified difference images.
        Quality Attributes: Clarity. Trade-offs: Heatmap scale differs between cameras if needed.
    Evidence: Internal: Same raw arrays as the JSON audit.
        External: Matplotlib. Findings: Display rotation matches official policy image orientation.
        Metrics: Difference values remain0–255channel units.
        Confidence: Visual review after generation.
    Constraints: No image resynthesis or source mutation.
    Reconsider When: Camera dimensions change.
    References: compare_sensors. ADR: None.
    """
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for row,(key,name) in enumerate((('agentview_image','Agent view'),('robot0_eye_in_hand_image','Wrist view'))):
        a,b=np.rot90(cached[key],2),np.rot90(regenerated[key],2);difference=np.mean(np.abs(a.astype(float)-b.astype(float)),axis=2)
        axes[row,0].imshow(a);axes[row,0].set_title(name+' • cached step')
        axes[row,1].imshow(b);axes[row,1].set_title(name+' • same snapshot regenerated')
        plot=axes[row,2].imshow(difference,cmap='magma',vmin=0,vmax=max(1.,float(difference.max())));axes[row,2].set_title('Mean absolute channel difference');fig.colorbar(plot,ax=axes[row,2],label='RGB channel units [0–255]',shrink=.8)
        for ax in axes[row]:ax.set_axis_off()
    fig.suptitle('Sensor sampling-phase diagnostic • same physical state • no policy inference',weight='bold')
    for extension in ('png','svg'):fig.savefig(output/f'sensor-phase.{extension}',dpi=170)
    plt.close(fig)
    return ['sensor-phase.png','sensor-phase.svg']


def main():
    """Perform one post-evaluation diagnostic without changing any formal run artifacts.

    Args: CLI init/seed/output. Returns: None. Raises: Invalid paths or state-change assertions.
    Design: Context: Formal scores must remain tied to their original fixed implementation.
        Concern: A diagnostic may accidentally alter inputs mid-benchmark.
        Decision: Standalone optional script with separate output and no evaluator edits.
        Alternatives: Fixing sensor timing during active runs.
        Quality Attributes: Isolation. Trade-offs: Diagnostic is explanatory rather than causal.
    Evidence: Internal: Before/after flattened MuJoCo state and ctrl hashes.
        External: Official environment regeneration API.
        Findings: Actual sensor differences written even if state invariance fails.
        Metrics: State equality, sensing differences and fixed initial state identity.
        Confidence: One initial configuration.
    Constraints: Do not execute concurrently with formal model evaluations.
    Reconsider When: A follow-up controlled benchmark is authorized.
    References: Module contract. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--init',type=int,default=1);parser.add_argument('--seed',type=int,default=20261102);parser.add_argument('--output',type=Path,default=ROOT/'artifacts/realtime-vla/report');args=parser.parse_args()
    output=args.output.resolve();allowed=(ROOT/'artifacts/realtime-vla/report').resolve()
    if not output.is_relative_to(allowed):raise ValueError('output must remain under artifacts/realtime-vla/report')
    output.mkdir(parents=True,exist_ok=True)
    from libero_env import FrozenLiberoEnv
    from threadpoolctl import threadpool_limits
    env=None
    try:
        with threadpool_limits(limits=1):
            env=FrozenLiberoEnv(control_freq=20,render_mode='sync');cached=env.reset(init_state_index=args.init,seed=args.seed)
            before=env.snapshot(episode_id='post-evaluation-sensor-phase');before_ctrl=np.asarray(env.env.sim.data.ctrl).copy()
            regenerated=env.observe_snapshot(before);after=env.snapshot(episode_id='post-evaluation-sensor-phase');after_ctrl=np.asarray(env.env.sim.data.ctrl).copy()
            audit=compare_sensors(cached,regenerated)
            state_equal=bool(np.array_equal(before['physics_state'],after['physics_state']));ctrl_equal=bool(np.array_equal(before_ctrl,after_ctrl))
            audit.update(status='passed' if state_equal and ctrl_equal else'failed',initial_state_index=args.init,seed=args.seed,environment=env.metadata,physics_state_byte_equal=before['physics_state'].tobytes()==after['physics_state'].tobytes(),physics_state_array_equal=state_equal,actuator_control_array_equal=ctrl_equal,physics_time_before=before['simulation_time'],physics_time_after=after['simulation_time'],scope='One diagnostic state; cached observations versus forced regeneration. Does not estimate any causal effect on formal task success.',source_mechanism='robosuite Observable.update caches one sample per control period; LIBERO regenerate_obs_from_state calls sim.forward and force-updates observables.',raw_arrays=['sensor-phase-cached.npz','sensor-phase-regenerated.npz'])
            np.savez(output/'sensor-phase-cached.npz',**{k:cached[k] for k in SENSORS});np.savez(output/'sensor-phase-regenerated.npz',**{k:regenerated[k] for k in SENSORS})
            audit['plots']=plot_difference(cached,regenerated,output)
            (output/'sensor-phase-audit.json').write_text(json.dumps(audit,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
            print(json.dumps(audit,ensure_ascii=False))
            if not state_equal or not ctrl_equal:raise AssertionError('snapshot regeneration changed physical state or actuator control')
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
