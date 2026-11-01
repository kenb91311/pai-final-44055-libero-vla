"""Validate native RTC chunk generation and frozen-state integrity on one GPU.

Args: Fixed model/raw-sensor paths, family and a fresh output directory.
Returns: Three calls per RTC variant, paired action arrays, resource/audit JSON and parity data.
Raises: Asset/model/GPU errors; partial failures remain explicit in the output.
Design:
    Context: Analytical CPU VJP tests cannot prove actual VLA backward compatibility.
    Concern: A successful constructor is not proof of finite chunks or frozen weights.
    Decision: Actual recorded LIBERO sensors, sequential model residency and exact audits.
    Alternatives: Synthetic zero images or simultaneous serving obscure real compatibility.
    Quality Attributes: Fidelity, isolation. Trade-offs: Startup/audits take more time than inference.
Evidence:
    Internal: Raw environment-smoke observation and continuity runtime CPU tests.
    External: Official LeRobot guided RTC API.
    Findings: This checks inference only, not ROS delivery, real-time physics or task success.
    Metrics: Per-call total/model time, native shapes, CUDA peak and weight/gradient checks.
    Confidence: Restricted to the recorded sensing and selected GPU/software stack.
Constraints: No learning, model quantization, simulator stepping or privileged model inputs.
Reconsider When: Checkpoint, device or software version changes.
References: continuity_runtime.py; docs/reference/realtime-continuity-methods.md. ADR: None.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from continuity_runtime import FrozenContinuityPolicy

PROMPT='pick up the black bowl between the plate and the ramekin and place it on the plate'
SENSORS=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def write_json(path,value):
    """Persist finite smoke metadata, leaving raw action arrays in separate NumPy files.

    Args: path: output path; value: JSON mapping. Returns: None. Raises: IO/ValueError.
    Design: Context: Partial smoke failures need durable evidence.
        Concern: Nonfinite metrics cannot silently become valid JSON.
        Decision: Strict finite encoding. Alternatives: Console-only records rejected.
        Quality Attributes: Auditability. Trade-offs: Small repeated file writes.
    Evidence: Internal: Per-variant checkpoints. External: JSON numeric grammar.
        Findings: NaN action outputs fail earlier. Metrics: None. Confidence: IO contract.
    Constraints: No camera pixels in JSON. Reconsider When: Evidence moves to a database.
    References: main. ADR: None.
    """
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):
    """Hash small code/sensor source files for the recorded smoke configuration.

    Args: path: existing file. Returns: SHA256. Raises: IO errors.
    Design: Context: Runtime can evolve after a smoke run.
        Concern: Unpinned implementation weakens reproducibility.
        Decision: Record code bytes. Alternatives: Timestamp alone rejected.
        Quality Attributes: Traceability. Trade-offs: Untimed file reads.
    Evidence: Internal: Stored source-sha256 mapping. External: SHA256.
        Findings: No weights read by this helper. Metrics: None. Confidence: Byte identity.
    Constraints: Small files only. Reconsider When: Large streaming assets are hashed.
    References: main. ADR: None.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_prediction(family,modeldir,raw,prompt,seed,output):
    """Run the original frozen adapter once, outside every RTC variant's residency.

    Args: family/modeldir: verified model; raw/prompt; seed; output: fresh smoke folder.
    Returns: Physical action array and original predictor metadata/audit.
    Raises: Restoration, inference or frozen-audit errors.
    Design: Context: New runtime replaces inference_mode with no_grad.
        Concern: That refactor could unintentionally change ordinary model output.
        Decision: Pair same raw sensors and RNG before comparing first off chunk.
        Alternatives: Shape-only comparison rejected. Quality Attributes: Compatibility.
        Trade-offs: Another untimed model load.
    Evidence: Internal: Saved baseline-actions.npy. External: Existing frozen family adapters.
        Findings: Exact equality is measured; hardware determinism is not assumed.
        Metrics: Max/mean absolute output difference. Confidence: One paired seed/observation.
    Constraints: Baseline unloaded before the new runtime is created.
    Reconsider When: Memory contention itself is the experiment.
    References: FrozenSmolVLA.predict; FrozenPI05.predict. ADR: None.
    """
    if family=='smolvla':
        from smolvla_runtime import FrozenSmolVLA
        runtime=FrozenSmolVLA(modeldir,device='cuda')
    else:
        from pi05_runtime import FrozenPI05
        runtime=FrozenPI05(modeldir,device='cuda')
    try:
        runtime.reset(seed);actions,metadata=runtime.predict(raw,prompt);audit=runtime.audit()
        if not audit['weights_unchanged']:raise RuntimeError('baseline weights changed')
        np.save(output/'baseline-actions.npy',actions)
        return actions,{'metadata':metadata,'audit':audit,'scope':'same-seed first native chunk; inference_mode original baseline'}
    finally:
        if hasattr(runtime,'close'):runtime.close()
        runtime=None;gc.collect();torch.cuda.empty_cache()


def run_variant(family,modeldir,variant,raw,prompt,seed,output,baseline=None):
    """Generate one bootstrap then two prefix-conditioned chunks for one RTC variant.

    Args: family/modeldir/variant; raw/prompt; seed/output; baseline: optional parity array.
    Returns: Variant result with all call records, frozen audit and optional parity.
    Raises: Invalid finite shapes, mutation, unwanted parameter gradients or runtime errors.
    Design: Context: Native H50 is needed before real-time scheduling can be tested.
        Concern: Calling no-prefix RTC never exercises the guidance/backward path.
        Decision: Use bootstrap's clipped physical H50 as known prefix for calls2/3.
        Alternatives: Random prefixes are less representative of actual queued actions.
        Quality Attributes: Fidelity. Trade-offs: Held sensor replay is not an on-policy rollout.
    Evidence: Internal: Both model/physical arrays saved per call.
        External: LeRobot prefix normalization. Findings: Prefix weights use d16/s30.
        Metrics: Cold/warm per-call latency and peak memory, not a steady benchmark average.
        Confidence: GPU API compatibility only.
    Constraints: No simulator progress or ROS transport in this smoke.
    Reconsider When: A different delay/horizon configuration is evaluated.
    References: FrozenContinuityPolicy. ADR: None.
    """
    started=time.perf_counter()
    runtime=FrozenContinuityPolicy(family,modeldir,variant=variant,device='cuda',execution_horizon=30)
    result={'variant':variant,'status':'running','startup_seconds':time.perf_counter()-started,'calls':[],
            'timing_note':'call1 bootstrap without prefix; calls2/3 use a known H50 prefix; cold/warm mix, not steady latency'}
    try:
        runtime.reset(seed);prefix=None
        for index in range(3):
            generated=runtime.predict(raw,prompt,prev_actions=prefix,inference_delay=0 if index==0 else 16,execution_horizon=30)
            actions=generated['actions'];normalized=generated['normalized']
            if actions.shape!=normalized.shape or actions.shape!=(50,7):raise RuntimeError('paired native chunk shape mismatch')
            if not np.isfinite(actions).all() or not np.isfinite(normalized).all():raise RuntimeError('nonfinite generated chunk')
            np.save(output/f'{variant}-physical-{index}.npy',actions)
            np.save(output/f'{variant}-normalized-{index}.npy',normalized)
            restored=runtime.normalize_actions(actions)
            roundtrip=float(np.max(np.abs(restored-normalized)))
            if not np.allclose(restored,normalized,rtol=2e-5,atol=2e-5):raise RuntimeError('physical/normalized processor roundtrip mismatch')
            result['calls'].append({'index':index,'has_prefix':prefix is not None,'normalized_roundtrip_max_abs':roundtrip,
                                    'shape':list(actions.shape),**generated['metadata']})
            if index==0:
                prefix=np.clip(actions,-1,1).copy()
                if baseline is not None:
                    difference=np.abs(actions-baseline)
                    result['baseline_parity']={'exact_equal':bool(np.array_equal(actions,baseline)),
                        'allclose_rtol_1e_5_atol_1e_5':bool(np.allclose(actions,baseline,rtol=1e-5,atol=1e-5)),
                        'max_abs_difference':float(difference.max()),'mean_abs_difference':float(difference.mean()),
                        'same_seed':seed,'scope':'first no-prefix off chunk no_grad versus original frozen inference_mode'}
            write_json(output/f'{variant}.json',result)
        audit=runtime.audit();result['audit']=audit
        if not audit['weights_unchanged'] or audit['requires_grad_parameter_count'] or audit['parameters_with_grad']:
            raise RuntimeError('frozen-weight or parameter-gradient audit failed')
        result['status']='passed';write_json(output/f'{variant}.json',result)
        return result
    except Exception as error:
        result['status']='error';result['error']=f'{type(error).__name__}: {error}'
        write_json(output/f'{variant}.json',result);raise
    finally:
        runtime.close();runtime=None;gc.collect();torch.cuda.empty_cache()


def main():
    """Validate actual GPU guidance sequentially and retain failures without task-success claims.

    Args: CLI model/raw/output paths and optional original-baseline parity skip.
    Returns: None. Raises: RuntimeError/IO errors after writing explicit failure status.
    Design: Context: Parent grants one GPU lease after previous formal experiments finish.
        Concern: Smoke must not contend with another model.
        Decision: One runtime at a time, explicit fresh output and source hashes.
        Alternatives: Background co-loading rejected. Quality Attributes: Comparability.
        Trade-offs: Baseline plus three model initializations.
    Evidence: Internal: Per-call native arrays and final frozen audits.
        External: PyTorch CUDA stats. Findings: No task/ROS success is inferred here.
        Metrics: Finite chunks, GPU bytes and timing. Confidence: API scope only.
    Constraints: Caller owns GPU exclusion. Reconsider When: MultiGPU hardware is used.
    References: run_variant. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model',choices=['smolvla','pi05'],default='smolvla')
    parser.add_argument('--modeldir',type=Path)
    parser.add_argument('--raw',type=Path,default=Path('artifacts/realtime-vla/environment-smoke/raw-observation.npz'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=20261102)
    parser.add_argument('--skip-baseline-parity',action='store_true');args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this actual GPU smoke')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    with np.load(args.raw,allow_pickle=False) as saved:
        raw={key:saved[key].copy() for key in SENSORS}
    folder=Path(__file__).parent
    write_json(args.output/'source-sha256.json',{name:digest(folder/name) for name in ('smoke_continuity.py','continuity_runtime.py','smolvla_runtime.py','pi05_runtime.py')})
    result={'status':'running','model':args.model,'seed':args.seed,'device':torch.cuda.get_device_name(),
        'scope':'recorded raw LIBERO init0 inference only; no simulator progress, ROS transport or task-success evidence',
        'raw_source_sha256':digest(args.raw),'sensor_sha256':{key:hashlib.sha256(raw[key].tobytes()).hexdigest() for key in SENSORS},
        'prompt':PROMPT,'local_training':False,'variants':[]}
    modeldir=args.modeldir or Path('artifacts/realtime-vla/models')/args.model
    try:
        baseline=None
        if not args.skip_baseline_parity:
            baseline,result['baseline']=baseline_prediction(args.model,modeldir,raw,PROMPT,args.seed,args.output)
            write_json(args.output/'summary.json',result)
        for variant in ('off','rtc-stock','rtc-vjp'):
            record=run_variant(args.model,modeldir,variant,raw,PROMPT,args.seed,args.output,baseline if variant=='off' else None)
            result['variants'].append(record);write_json(args.output/'summary.json',result)
            print(json.dumps({'variant':variant,'status':record['status'],'last_call_ms':record['calls'][-1]['inference_ms'],
                              'peak_cuda_bytes':record['audit']['peak_cuda_bytes'],'weights_unchanged':record['audit']['weights_unchanged']}),flush=True)
        result['status']='passed';write_json(args.output/'summary.json',result)
    except Exception as error:
        result['status']='error';result['error']=f'{type(error).__name__}: {error}'
        write_json(args.output/'summary.json',result);raise


if __name__=='__main__':main()
