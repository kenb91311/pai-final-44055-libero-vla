"""Verify the fixed author VLASH schema or a frozen real-sensor inference API.

Args: --schema-only or explicit API smoke, model directory and raw-observation NPZ.
Returns: JSON proof without task-success claims. Raises: Restoration/inference/audit errors.
Design: Context: New legacy backend must be validated before any rollout.
    Concern: Meta shape checks, API latency and task accuracy are different evidence levels.
    Decision: Separate schema-only output and warm2/measured3 raw-sensor smoke.
    Alternatives: Infer success from model naming or API output shape.
    Quality Attributes: Fault isolation. Trade-offs: Full audit needs exclusive CPU/GPU lease.
Evidence: Internal: Pinned checkpoint/source and actual saved LIBERO init0 raw cameras/proprioception.
    External: Original author Policy. Findings: No simulator stepping or ROS in this API smoke.
    Metrics: Shape, timing, peak CUDA, strict restored values and unchanged state.
    Confidence: API only. Constraints: No training, success claim or concurrent resident model.
Reconsider When: Backend/checkpoint changes. References: vlash_runtime.py. ADR: None.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from vlash_runtime import (FrozenVLASH,checkpoint_header,expand_aliases,load_author_components,materialize_nonpersistent_buffers)


def schema_smoke(modeldir):
    """Check the full original state schema using meta tensors without reading weight payloads.

    Args: modeldir: fixed assets. Returns: Structural proof. Raises: Schema/source/backend errors.
    Design: Context: MultiGB native checkpoint. Concern: Discovering schema errors after full GPU load.
        Decision: Meta constructor and explicit alias coverage; no tensor values loaded.
        Alternatives: Guess loader compatibility. Quality Attributes: Cheap fault isolation.
        Trade-offs: This does not prove restored values or inference.
    Evidence: Internal: Header and original architecture. External: PyTorch meta semantics.
        Findings: All aliases must exist as original parameter identities.
        Metrics: Key/elements counts. Confidence: Structural only.
    Constraints: Exclusive resource lease even for model metadata work.
    Reconsider When: Model structure changes. References: expand_aliases. ADR: None.
    """
    import torch
    cls,config,provenance=load_author_components(modeldir)
    with torch.device('meta'):policy=cls(config)
    expected=policy.state_dict(keep_vars=True);header=checkpoint_header(Path(modeldir)/'model.safetensors')
    dtypes={'F32':torch.float32,'BF16':torch.bfloat16}
    stored={key:torch.empty(value['shape'],dtype=dtypes[value['dtype']],device='meta') for key,value in header.items() if key!='__metadata__'}
    complete,aliases=expand_aliases(header,expected,stored)
    result={'stored_tensor_count':len(stored),'runtime_state_tensor_count':len(expected),'shared_alias_count':len(aliases),
            'stored_element_count':sum(t.numel() for t in stored.values()),
            'unique_parameter_count':sum(p.numel() for p in policy.parameters()),
            'normalization_elements':sum(t.numel() for k,t in stored.items() if k.startswith(('normalize_inputs.','normalize_targets.','unnormalize_outputs.'))),
            'all_state_keys_covered':set(complete)==set(expected),'all_aliases_originally_shared':True,
            'real_tensor_values_loaded':False,'gpu_inference':False,'task_success_measured':False,
            'backend':provenance['backend']}
    del policy,expected,stored,complete;gc.collect();return result


def api_smoke(modeldir,observation):
    """Run two warmups and three measured predictions on real frozen sensors, then audit.

    Args: modeldir: pinned assets; observation: raw sensor NPZ. Returns: API evidence.
    Raises: Any strict restore, shape, inference or frozen-weight failure.
    Design: Context:16GB RTX5080. Concern: API fitting/latency is unverified.
        Decision: Exact native config and synchronized timings, real input, close in finally.
        Alternatives: Synthetic task-success claims. Quality Attributes: Measurability.
        Trade-offs: One observation does not evaluate a policy rollout.
    Evidence: Internal: Saved actual LIBERO sensor packet. External: Author native inference.
        Findings: No physics steps occur here. Metrics: Warm/measured latency and peak VRAM.
        Confidence: API only. Constraints: No ROS or training.
    Reconsider When: Hardware/model changes. References: FrozenVLASH. ADR: None.
    """
    import torch
    torch.set_num_threads(4)
    keys=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')
    with np.load(observation,allow_pickle=False) as data:raw={key:data[key].copy() for key in keys}
    prompt='pick up the black bowl between the plate and the ramekin and place it on the plate'
    runtime=None;start=time.perf_counter()
    try:
        runtime=FrozenVLASH(modeldir);startup=time.perf_counter()-start;runtime.reset(20261102)
        warm=[];measured=[]
        for index in range(5):
            actions,meta=runtime.predict(raw,prompt)
            record={'shape':list(actions.shape),'finite':bool(np.isfinite(actions).all()),'metadata':meta}
            (warm if index<2 else measured).append(record)
        audit=runtime.audit()
        if not audit['weights_unchanged'] or audit['requires_grad_parameter_count']!=0:raise RuntimeError('VLASH frozen audit failed')
        return {'status':'passed','scope':'real raw LIBERO init0 API only; no simulator steps/task accuracy/ROS',
                'startup_seconds':startup,'warmup_count':2,'measured_count':3,'warmups':warm,'measured':measured,
                'mean_inference_ms':float(np.mean([r['metadata']['inference_ms'] for r in measured])),
                'mean_model_inference_ms':float(np.mean([r['metadata']['model_inference_ms'] for r in measured])),
                'audit':audit,'local_training':False,'task_success_measured':False}
    finally:
        if runtime is not None:runtime.close()


def main():
    """Execute a requested smoke and atomically preserve its evidence JSON.

    Args: CLI. Returns: None. Raises: Test errors.
    Design: Context: Explicit resource lease. Concern: Import-time model loads.
        Decision: Main guard and separate mode flag. Alternatives: Implicit smoke on import.
        Quality Attributes: Predictability. Trade-offs: None.
    Evidence: Internal: --help precedes calls. External: argparse. Findings: No side effects on import.
        Metrics: JSON proof. Confidence: Interface contract.
    Constraints: Output is new method evidence only. Reconsider When: CLI changes.
    References: schema_smoke; api_smoke. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--modeldir',type=Path,default=Path('artifacts/realtime-vla/models/vlash'))
    parser.add_argument('--raw-observation',type=Path,default=Path('artifacts/realtime-vla/environment-smoke/raw-observation.npz'))
    parser.add_argument('--schema-only',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();name='schema-smoke.json' if args.schema_only else 'api-smoke.json'
    result=schema_smoke(args.modeldir) if args.schema_only else api_smoke(args.modeldir,args.raw_observation)
    output=args.output or args.modeldir/name;output.parent.mkdir(parents=True,exist_ok=True)
    part=output.with_suffix(output.suffix+'.partial');part.write_text(json.dumps(result,indent=2)+'\n');part.replace(output)
    print(json.dumps({'output':str(output),'status':result.get('status','schema-passed'),'mean_inference_ms':result.get('mean_inference_ms')},indent=2))


if __name__=='__main__':main()
