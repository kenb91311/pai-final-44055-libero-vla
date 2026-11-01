"""Run the official LIBERO SmolVLA expert without training or hidden action buffering.

Args: Fixed local model directory, CUDA device and optional autocast setting.
Returns: Full unnormalized action chunks and measured inference timing.
Raises: ValueError on sensor/schema errors; RuntimeError on unavailable GPU/model dependencies.
Design:
    Context: A real-time scheduler must own chunk consumption independently of model inference.
    Concern: select_action's private queue would obscure generation and action-age measurements.
    Decision: Official predict_action_chunk plus saved pre/postprocessors; return all50actions.
    Alternatives: Generic base model is not LIBERO ready; manual normalization risks mismatch.
    Quality Attributes: Reproducibility, observability. Trade-offs: Fixed released checkpoint.
Evidence:
    Internal: Pinned model config declares2RGB/state8/action7/chunk50.
    External: LeRobot SmolVLA and LiberoProcessorStep reference implementation.
    Findings: Official environment processor rotates raw camera images180degrees.
    Metrics: Chunk wall time, CUDA peak allocation, parameter digest. Confidence: Empirical only.
Constraints: No optimizer, training or privileged object state input.
Reconsider When: Checkpoint or simulator sensor convention changes.
References: artifacts/realtime-vla/models/smolvla/provenance.json.
ADR: None.
"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


class FrozenSmolVLA:
    """Own one frozen official expert and expose explicit chunk-generation calls.

    Args: modeldir: downloaded policy and vlm metadata; device: CUDA; use_amp: bf16 autocast.
    Returns: Inference runtime.
    Raises: RuntimeError for missing GPU; IO/model errors on incomplete assets.
    Design:
        Context: SmolVLA is benchmarked sequentially against a larger policy on16GB.
        Concern: Duplicate base-VLM download or hidden queueing distorts resource measurements.
        Decision: Construct VLM from local config, restore all weights from the complete policy.
        Alternatives: Loading original VLM weights is redundant before restoring the task checkpoint.
        Quality Attributes: Isolation, resource efficiency. Trade-offs: Local metadata is required.
    Evidence:
        Internal: Policy safetensors includes all604,934,176 parameters.
        External: Official SmolVLAConfig.load_vlm_weights=False branch.
        Findings: Model reset clears policy history; scheduler owns executable action queue.
        Metrics: Parameter digest and peak CUDA memory. Confidence: Runtime measured.
    Constraints: Same official normalizers, camera orientation and prompt processing.
    Reconsider When: Model serialization changes.
    References: download_smolvla.py.
    ADR: None.
    """
    def __init__(self, modeldir, device='cuda', use_amp=True):
        """Restore model and processors without fetching duplicate base weights.

        Args: modeldir/device/use_amp: class contract.
        Returns: None. Raises: Model/dependency errors before accepting observations.
        Design:
            Context: Pretrained policy owns normalization and tokenization.
            Concern: Processor tokenizer must resolve the same pinned local VLM metadata.
            Decision: Override only local paths/device and construction-time base-weight loading.
            Alternatives: Hardcoded stats would break reproducibility.
            Quality Attributes: Compatibility. Trade-offs: Library API is pinned by environment.
        Evidence:
            Internal: Official saved processor files include mean/std tensors.
            External: LeRobot make_pre_post_processors.
            Findings: The inference graph and learned policy weights are unchanged.
            Metrics: Startup and parameter identity. Confidence: Verified at runtime.
        Constraints: CUDA required when requested.
        Reconsider When: Processor schema changes.
        References: Model config and provenance.
        ADR: None.
        """
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.processor.env_processor import LiberoProcessorStep
        self.modeldir=Path(modeldir).resolve();self.device=torch.device(device);self.use_amp=bool(use_amp)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        config=SmolVLAConfig.from_pretrained(self.modeldir)
        config.vlm_model_name=str(self.modeldir/'vlm')
        config.load_vlm_weights=False
        config.device=str(self.device)
        self.policy=SmolVLAPolicy.from_pretrained(self.modeldir,config=config).to(self.device).eval()
        for parameter in self.policy.parameters():parameter.requires_grad_(False)
        self.preprocess,self.postprocess=make_pre_post_processors(
            self.policy.config,str(self.modeldir),
            preprocessor_overrides={'device_processor':{'device':str(self.device)},'tokenizer_processor':{'tokenizer_name':str(self.modeldir/'vlm')}})
        self.libero_processor=LiberoProcessorStep()
        self.policy.reset();self.calls=0
        self.restoration=self.verify_checkpoint_restoration()
        self.weights_before=self.weight_digest()
        self.metadata={'model':'HuggingFaceVLA/smolvla_libero','revision':'6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
                       'parameter_count':sum(p.numel() for p in self.policy.parameters()),'chunk_size':int(config.chunk_size),
                       'native_n_action_steps':int(config.n_action_steps),'flow_steps':int(config.num_steps),
                       'amp_dtype':'bfloat16' if self.use_amp and self.device.type=='cuda' else None,
                       'local_training':False,'image_transform':'official LiberoProcessorStep180degree rotation',
                       'action_transform':'official saved postprocessor; no additional gripper inversion','restoration':self.restoration}

    def verify_checkpoint_restoration(self):
        """Compare every stored checkpoint tensor against the reconstructed model.

        Args: None. Returns: Counts of restored tensors and model-only aliases/buffers.
        Raises: ValueError if a checkpoint tensor is missing or changed during restoration.
        Design:
            Context: VLM base weights are deliberately not downloaded separately.
            Concern: Randomly initialized visual or language layers would invalidate inference.
            Decision: Compare all safetensor values, converting model tensors back to stored dtype.
            Alternatives: Checking only key shapes cannot establish value restoration.
            Quality Attributes: Integrity. Trade-offs: Untimed startup copies model tensors to CPU.
        Evidence:
            Internal: Fixed official model.safetensors and reconstructed policy.state_dict.
            External: Safetensors safe_open API.
            Findings: dtype conversion preserves exactly the original BF16 values.
            Metrics: Tensor count and element count. Confidence: Exact value identity.
        Constraints: Run before any inference; no checkpoint mutation.
        Reconsider When: Quantized model representation is introduced.
        References: download_smolvla.py provenance.
        ADR: None.
        """
        from safetensors import safe_open
        state=self.policy.state_dict();checked=0;elements=0
        with safe_open(self.modeldir/'model.safetensors',framework='pt',device='cpu') as source:
            keys=list(source.keys())
            for key in keys:
                expected=source.get_tensor(key)
                if key not in state:raise ValueError(f'checkpoint tensor absent in model: {key}')
                actual=state[key].detach().cpu().to(dtype=expected.dtype)
                if actual.shape!=expected.shape or not torch.equal(actual,expected):raise ValueError(f'checkpoint restoration mismatch: {key}')
                checked+=1;elements+=expected.numel()
        extras=sorted(set(state)-set(keys))
        parameters=dict(self.policy.named_parameters(remove_duplicate=False))
        restored_pointers={state[key].data_ptr() for key in keys}
        for key in extras:
            if key in parameters and state[key].data_ptr() not in restored_pointers:
                raise ValueError(f'model parameter has no checkpoint value or tied alias: {key}')
        return {'all_stored_tensors_match':True,'checked_tensors':checked,'stored_elements':elements,'model_only_keys':extras,'model_only_parameters_are_tied_aliases':True}

    def weight_digest(self):
        """Hash model parameters and buffers for before/after immutability evidence.

        Args: None. Returns: SHA256. Raises: RuntimeError on tensor transfer failure.
        Design:
            Context: Local training is prohibited in this comparison.
            Concern: File identity alone does not prove in-memory inference immutability.
            Decision: Include names/dtypes/shapes and raw tensor bytes.
            Alternatives: Scalar sums can miss compensating parameter changes.
            Quality Attributes: Integrity. Trade-offs: Hashing is outside timed inference.
        Evidence:
            Internal: Frozen eval model state.
            External: SHA256. Findings: Buffers are included alongside parameters.
            Metrics: Exact digest equality. Confidence: High for identity.
        Constraints: No concurrent inference while auditing.
        Reconsider When: Quantized tensor storage changes.
        References: audit.
        ADR: None.
        """
        digest=hashlib.sha256()
        for name,tensor in self.policy.state_dict().items():
            value=tensor.detach().cpu().contiguous()
            digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def reset(self,seed=0):
        """Reset model history and inference RNG at an episode boundary.

        Args: seed: integer seed. Returns: None. Raises: ValueError for invalid seed.
        Design:
            Context: Paired task runs need consistent stochastic flow noise.
            Concern: Queue state must not cross episodes.
            Decision: Reset official policy and both CPU/CUDA Torch generators.
            Alternatives: Resetting only the scheduler leaves model queues intact.
            Quality Attributes: Reproducibility. Trade-offs: Same seed does not force hardware determinism.
        Evidence:
            Internal: SmolVLA policy.reset clears queues.
            External: Torch manual_seed.
            Findings: Scheduler receives full chunks independently.
            Metrics: Episode identity. Confidence: High.
        Constraints: Call between episodes, not during queued chunk consumption.
        Reconsider When: Concurrent episodes are required.
        References: predict.
        ADR: None.
        """
        if type(seed) is not int or seed<0:raise ValueError('seed must be nonnegative integer')
        np.random.seed(seed);torch.manual_seed(seed);self.policy.reset();self.calls=0

    def prepare_observation(self,observation,prompt):
        """Build the exact vision/proprio batch through the official LIBERO processor.

        Args: observation: raw LIBERO sensors; prompt: task instruction.
        Returns: Preprocessed policy batch.
        Raises: ValueError for invalid image/state schema.
        Design:
            Context: Both models see the same raw camera sensors and task instruction.
            Concern: Image flips or quaternion transforms must happen exactly once.
            Decision: Raw HWC pixels to tensor, official LiberoProcessorStep, saved policy preprocess.
            Alternatives: Passing preflipped env images would double-rotate inputs.
            Quality Attributes: Compatibility. Trade-offs: Raw observation contract is strict.
        Evidence:
            Internal: Env adapter explicitly leaves camera images unflipped.
            External: LeRobot env_processor.py.
            Findings: The allowlist excludes object pose and success labels.
            Metrics: Input shapes/finite values. Confidence: High.
        Constraints: RGB uint8 and XYZW quaternion.
        Reconsider When: Camera convention changes.
        References: FrozenLiberoEnv.
        ADR: None.
        """
        data={}
        for source,target in [('agentview_image','observation.images.image'),('robot0_eye_in_hand_image','observation.images.image2')]:
            image=np.asarray(observation[source])
            if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:raise ValueError('expected raw HWC RGB uint8')
            data[target]=torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).unsqueeze(0).float()/255.
        values={}
        for name,size in [('robot0_eef_pos',3),('robot0_eef_quat',4),('robot0_gripper_qpos',2)]:
            value=np.asarray(observation[name],dtype=np.float32)
            if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError('invalid proprioception')
            values[name]=torch.from_numpy(value.copy()).unsqueeze(0)
        data['observation.robot_state']={'eef':{'pos':values['robot0_eef_pos'],'quat':values['robot0_eef_quat']},'gripper':{'qpos':values['robot0_gripper_qpos']}}
        data=self.libero_processor.observation(data)
        data['task']=prompt
        return self.preprocess(data)

    def predict(self,observation,prompt):
        """Generate and unnormalize a full action chunk, measuring actual synchronized latency.

        Args: observation: raw current LIBERO sensors; prompt: instruction.
        Returns: (float32 numpy[T,7] actions, metadata dictionary).
        Raises: ValueError on invalid actions; runtime errors on inference failure.
        Design:
            Context: Real-time execution must measure the complete model-facing request cost.
            Concern: Asynchronous CUDA kernels can under-report latency.
            Decision: Synchronize before/after preprocessing, generation and postprocessing.
            Alternatives: Python launch timing omits GPU work.
            Quality Attributes: Measurement accuracy. Trade-offs: Synchronization removes inter-request overlap.
        Evidence:
            Internal: Official predict_action_chunk generates the full50-step horizon.
            External: CUDA synchronization semantics.
            Findings: Scheduler alone decides how many actions to execute before replacement.
            Metrics: End-to-end latency and CUDA peak memory. Confidence: Empirical.
        Constraints: No physics stepping or sleeping occurs inside inference.
        Reconsider When: Model-side pipelining is benchmarked.
        References: Model metadata.
        ADR: None.
        """
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        batch=self.prepare_observation(observation,prompt)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        model_started=time.perf_counter()
        context=torch.autocast('cuda',dtype=torch.bfloat16) if self.use_amp and self.device.type=='cuda' else nullcontext()
        with torch.inference_mode(),context:
            chunk=self.policy.predict_action_chunk(batch)
            actions=self.postprocess(chunk)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        model_ms=(time.perf_counter()-model_started)*1000
        result=actions.detach().float().cpu().numpy()
        if result.ndim==3 and result.shape[0]==1:result=result[0]
        if result.ndim!=2 or result.shape[-1]!=7 or not np.isfinite(result).all():raise ValueError(f'invalid action chunk {result.shape}')
        self.calls+=1
        return result,{'inference_ms':(time.perf_counter()-started)*1000,'model_inference_ms':model_ms,'chunk_length':len(result),'call':self.calls,
                       'peak_cuda_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
                       'action_min':float(result.min()),'action_max':float(result.max()),'amp_dtype':self.metadata['amp_dtype']}

    def audit(self):
        """Report frozen-weight evidence after completed inference.

        Args: None. Returns: Audit mapping. Raises: RuntimeError on state hashing failure.
        Design:
            Context: Evaluation must not update public checkpoint weights.
            Concern: Grad flags alone are weaker than measured identity.
            Decision: Recompute complete model-state digest.
            Alternatives: Checking checkpoint file only misses in-memory mutations.
            Quality Attributes: Integrity. Trade-offs: Expensive but untimed.
        Evidence:
            Internal: weight_digest includes buffers and parameters.
            External: None. Findings: No optimizer exists in this runtime.
            Metrics: Digest equality. Confidence: High.
        Constraints: Audit between inference calls.
        Reconsider When: State serialization changes.
        References: weight_digest.
        ADR: None.
        """
        after=self.weight_digest()
        return {'weights_before_sha256':self.weights_before,'weights_after_sha256':after,'weights_unchanged':after==self.weights_before,'local_training':False,'calls':self.calls,'peak_cuda_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,**self.metadata}
