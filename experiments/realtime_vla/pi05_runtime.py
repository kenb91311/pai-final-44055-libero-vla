"""Run the fixed official pi0.5 LIBERO flow policy with strict frozen restoration.

Args: Model directory, CUDA device, approved raw LIBERO sensors and language prompt.
Returns: Native50x7 continuous action chunks and measured inference/audit metadata.
Raises: Integrity, tokenizer, state restoration, sensor or inference errors.
Design:
    Context: Add pi0.5 to the existing LIBERO system without changing previous models.
    Concern: Generic base weights, silent random fallback and guessed normalization invalidate comparison.
    Decision: Official task-finetuned checkpoint, exact processor stats and strict all-tensor checks.
    Alternatives: FAST tokens or generic pi05_base are different assets/methods.
    Quality Attributes: Fidelity, inspectability. Trade-offs: Native4.14B runtime parameter structure.
Evidence:
    Internal: Pinned HF revision, Gemma license, MEAN_STD and native50action config.
    External: Official LeRobot PI05Policy and HuggingFace checkpoint card.
    Findings: Official key conversion derives input embedding from stored lm_head weights.
    Metrics: Content equality, unchanged model state, synchronized latency, peak VRAM.
    Confidence: API smoke is distinct from task-success evaluation.
Constraints: No training, quantization, package edits, object truth or FAST action tokenizer.
Reconsider When: Checkpoint, LeRobot revision or processor schemas change.
References: docs/reference/pi05-libero-assets.md. ADR: None.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
from types import SimpleNamespace
import urllib.request
import numpy as np

os.environ.setdefault('HF_MODULES_CACHE',str(Path(__file__).resolve().parents[2]/'.cache/realtime-vla/hf-modules'))
try:
    from .download_pi05 import REPO,REVISION,WEIGHT_SHA256,file_digest
except ImportError:
    from download_pi05 import REPO,REVISION,WEIGHT_SHA256,file_digest
TEXT_URL='https://storage.googleapis.com/big_vision/paligemma_tokenizer.model'
TEXT_SHA='8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'
SNAPSHOT_SHA256={
    'config.json':'2f6d4b96b032593e1de65b391ee7252a70227bff398ef1b94658fea17818142f',
    'policy_preprocessor.json':'3d669ffc18d0364536735d8203076dcc69d3a6523aa5956ccaada3f9e1e6b748',
    'policy_postprocessor.json':'37d719a6584600988e6c343306d2eaee0575109f235ad500313577f73e47e8a1',
    'policy_preprocessor_step_2_normalizer_processor.safetensors':'a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors':'a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9',
}


def prepare_text_tokenizer(modeldir):
    """Build and verify the exact public PaliGemma vocabulary, without FAST components.

    Args: modeldir: fixed pi05 directory. Returns: Local text tokenizer path.
    Raises: Download, digest or vocabulary/encoding mismatch errors.
    Design:
        Context: HF PaliGemma tokenizer access is gated; Google publishes the original SP file.
        Concern: An unrelated tokenizer corrupts state/language conditioning.
        Decision: Public official SP, official Gemma conversion and all-ID/probe verification.
        Alternatives: Secret tokens, vocabulary substitution or loading a FAST action tokenizer.
        Quality Attributes: Identity, reproducibility. Trade-offs: Local format conversion.
    Evidence:
        Internal: Same Google SHA verified for the previous frozen FAST text tokenizer.
        External: big_vision PaliGemma asset and Transformers GemmaConverter.
        Findings: Transformers5.4 extractor needs its original SP handle restored locally.
        Metrics:257152 IDs and six prompt encodings. Confidence: Exact checks before inference.
    Constraints: Conversion shim only; no installed package/model source edits.
    Reconsider When: Tokenizer or Transformers version changes.
    References: tokenizer/text-validation.json. ADR: None.
    """
    import sentencepiece as spm
    from tokenizers import processors
    from transformers import AutoTokenizer,PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import GemmaConverter,GemmaSentencePieceExtractor
    root=Path(modeldir)/'tokenizer';root.mkdir(parents=True,exist_ok=True);original=root/'paligemma_tokenizer.model'
    previous=Path(modeldir).parent/'pi0fast/tokenizer/paligemma_tokenizer.model'
    if not original.exists():
        if previous.is_file() and file_digest(previous)==TEXT_SHA:shutil.copyfile(previous,original)
        else:urllib.request.urlretrieve(TEXT_URL,original)
    if file_digest(original)!=TEXT_SHA:raise ValueError('official PaliGemma SP digest mismatch')
    reference=spm.SentencePieceProcessor(model_file=str(original))
    assert reference.vocab_size()==257152 and (reference.pad_id(),reference.eos_id(),reference.bos_id(),reference.unk_id())==(0,1,2,3)

    class SPCompat(GemmaSentencePieceExtractor):
        """Restore only the missing native SP handle in Transformers5.4 conversion.

        Args: model: original file. Returns: Extractor. Raises: SP/model IO errors.
        Design: Context: Parent extractor changed to protobuf. Concern: self.sp missing.
            Decision: Reuse official extraction with native original SP handle.
            Alternatives: Global package monkeypatch. Quality Attributes: Isolation. Trade-offs: Compatibility shim.
        Evidence: Internal: Exact postconversion IDs/encodings. External: GemmaConverter5.4 source.
            Findings: No ranks, model tensors or inference code are replaced.
        Constraints: Conversion only. Reconsider When: Upstream fixes extractor.
        References: prepare_text_tokenizer. ADR: None.
        """
        def __init__(self,model):
            """Keep official initialization and supply its expected SP member.

            Args: model: file. Returns: None. Raises: File/SP errors.
            Design: Context: One format conversion. Concern: API mismatch.
                Decision: Restore original handle. Alternatives: Package edits.
                Quality Attributes: Fidelity. Trade-offs: None for token IDs.
            Evidence: Internal:257152 ID comparisons. External: Official extractor.
                Findings: Uses exact original file.
            Constraints: No global patch. Reconsider When: Dependency changes.
            References: SPCompat. ADR: None.
            """
            super().__init__(model);self.sp=spm.SentencePieceProcessor(model_file=str(model))

    text=root/'text'
    if not (text/'tokenizer.json').is_file():
        slow=SimpleNamespace(vocab_file=str(original),pad_token='<pad>',eos_token='<eos>',bos_token='<bos>',unk_token='<unk>',add_prefix_space=False)
        converter=GemmaConverter(slow);converter.SpmExtractor=SPCompat;backend=converter.converted()
        backend.post_processor=processors.TemplateProcessing(single='<bos> $A',pair='<bos> $A $B',special_tokens=[('<bos>',2)])
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<bos>',eos_token='<eos>',pad_token='<pad>',unk_token='<unk>',padding_side='right',clean_up_tokenization_spaces=False);tokenizer.save_pretrained(text)
    tokenizer=AutoTokenizer.from_pretrained(text,local_files_only=True)
    assert tokenizer.vocab_size==reference.vocab_size() and (tokenizer.pad_token_id,tokenizer.eos_token_id,tokenizer.bos_token_id,tokenizer.unk_token_id)==(0,1,2,3)
    differences=[]
    for index in range(reference.vocab_size()):
        before=reference.id_to_piece(index);after=tokenizer.convert_ids_to_tokens(index)
        if before!=after:differences.append((index,before,after))
    assert all(before=='<0x09>' and after=='\t' for _,before,after in differences),differences[:5]
    probes=['Action: ','pick up the black bowl','Task: pick up the black bowl, State: 1 2 3 4 5 6 7 8;\nAction: ','Task: pick up the black bowl between the plate and the ramekin and place it on the plate, State: 0 255 127 9 89 200 1 0;\nAction: ','put the red cup on the plate','open the top drawer']
    encoded=[]
    for prompt in probes:
        expected=reference.encode(prompt,out_type=int);actual=tokenizer.encode(prompt,add_special_tokens=False)
        assert actual==expected and tokenizer.encode(prompt,add_special_tokens=True)==[2]+expected
        encoded.append({'text':prompt,'ids':actual})
    (root/'text-validation.json').write_text(json.dumps({'source_url':TEXT_URL,'source_sha256':TEXT_SHA,'vocabulary_size':257152,'all_piece_ids_verified':True,'special_ids':{'pad':0,'eos':1,'bos':2,'unk':3},'official_tab_aliases':differences,'probes':encoded,'license':'upstream PaliGemma/Gemma terms','fast_action_tokenizer_used':False},indent=2)+'\n')
    return text


def model_state_digest(policy):
    """Hash all registered model state outside measured inference windows.

    Args: policy: official torch model. Returns: SHA256. Raises: Tensor transfer errors.
    Design: Context: Frozen evaluation. Concern: Hidden parameter changes.
        Decision: Name/dtype/shape/raw-byte digest before and after.
        Alternatives: Checking requires_grad alone.
        Quality Attributes: Auditability. Trade-offs: MultiGB CPU transfer.
    Evidence: Internal: Persistent state dictionaries. External: PyTorch semantics.
        Findings: BF16 is viewed as bytes before NumPy conversion.
    Constraints: Audit is untimed. Reconsider When: Persistent mutable caches are added.
    References: FrozenPI05.audit. ADR: None.
    """
    import torch
    result=hashlib.sha256()
    for name,tensor in sorted(policy.state_dict().items()):
        result.update(name.encode());result.update(str(tensor.dtype).encode());result.update(str(tuple(tensor.shape)).encode());result.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def restore_checkpoint(policy,checkpoint):
    """Strictly restore official converted keys and compare every resulting tensor value.

    Args: policy: official initialized model; checkpoint: verified safetensors.
    Returns: Restoration proof mapping. Raises: Any key/shape/value/integrity mismatch.
    Design:
        Context: Upstream from_pretrained can catch load failures and return random initialization.
        Concern: Successful construction is not proof that pretrained weights were restored.
        Decision: Official key conversion plus uncaught strict load and exact postcast equality.
        Alternatives: Trust model type, a state hash alone or random fallback.
        Quality Attributes: Fidelity. Trade-offs: CPU memory/transfer during startup.
    Evidence:
        Internal: Stored812 tensors, official derived input embedding copy, all mapped checks.
        External: PI05Policy._convert_openpi_state_dict.
        Findings: Current constructor holds distinct lm_head and input embedding parameters.
        Metrics: Checked tensors/elements and derived keys. Confidence: Exact comparison.
    Constraints: No learned parameter may remain unaccounted for.
    Reconsider When: Official state conversion changes.
    References: restoration.json. ADR: None.
    """
    import torch
    from safetensors.torch import load_file
    actual_sha=file_digest(checkpoint)
    if actual_sha!=WEIGHT_SHA256:raise ValueError('selected pi05 checkpoint SHA mismatch')
    stored=load_file(str(checkpoint),device='cpu');mapped=policy._convert_openpi_state_dict(stored);result=policy.load_state_dict(mapped,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    live=policy.state_dict()
    if set(live)!=set(mapped):raise RuntimeError('unaccounted model state after strict restoration')
    elements=0
    for key,expected in mapped.items():
        actual=live[key].detach().cpu();cast=expected.to(dtype=actual.dtype)
        if actual.shape!=cast.shape or not torch.equal(actual,cast):raise RuntimeError(f'pretrained tensor mismatch: {key}')
        elements+=actual.numel()
    proof={'checkpoint_sha256':actual_sha,'strict':True,'stored_tensor_count':len(stored),'stored_elements':sum(t.numel() for t in stored.values()),'mapped_tensor_count':len(mapped),'checked_tensor_count':len(mapped),'checked_elements':elements,'runtime_parameter_count':sum(p.numel() for p in policy.parameters()),'derived_copy_keys':sorted(set(mapped)-set(stored)),'all_mapped_tensors_equal_after_official_dtype_cast':True,'missing_keys':list(result.missing_keys),'unexpected_keys':list(result.unexpected_keys),'local_training':False}
    del stored,mapped,live;gc.collect()
    return proof


class FrozenPI05:
    """Own one frozen official pi0.5 policy and expose continuous native chunks.

    Args: modeldir: verified official task checkpoint; device: cuda or CPU.
    Returns: Predictor with reset/predict/audit/close. Raises: Restore/runtime errors.
    Design: Context: Scheduler independently advances physics while a GPU child predicts.
        Concern: Internal select_action queues hide the full chunk generation latency.
        Decision: Native predict_action_chunk, official processors and explicit synchronization.
        Alternatives: Time cached single actions. Quality Attributes: Fair measurement.
        Trade-offs: Scheduler separately chooses how much of native50 to consume.
    Evidence: Internal: Fixed checkpoint H50/action7/state8/MEAN_STD.
        External: Official PI05Policy. Findings: Flow head uses no FAST action tokens.
        Metrics: Full input→CPU output time and frozen audit. Confidence: Smoke first, rollout separately.
    Constraints: One GPU model at a time; no RTC/memory/training/quantization changes.
    Reconsider When: New runtime conditions are declared.
    References: pi05-libero-assets.md. ADR: None.
    """
    def __init__(self,modeldir='artifacts/realtime-vla/models/pi05',device='cuda'):
        """Load verified learned weights and the checkpoint's saved pre/postprocessors.

        Args: modeldir/device: see class. Returns: None. Raises: Integrity/import/CUDA errors.
        Design: Context: Existing LeRobot0.6.2/Transformers5.4 environment is retained.
            Concern: Default quantiles or a base model would change the benchmark condition.
            Decision: Preserve saved MEAN_STD, disable compile/checkpointing for eager inference.
            Alternatives: Patch installed dependencies. Quality Attributes: Isolation.
            Trade-offs: Startup includes all tensor value checks.
        Evidence: Internal: Config/state validation. External: Official model config.
            Findings: No optimizer or gradient updates are instantiated.
        Constraints: No previous runtime edits. Reconsider When: Config schema changes.
        References: restoration.json. ADR: None.
        """
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.processor.env_processor import LiberoProcessorStep
        self.torch=torch;self.device=torch.device(device);self.modeldir=Path(modeldir).resolve()
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        for name,expected_digest in SNAPSHOT_SHA256.items():
            if file_digest(self.modeldir/name)!=expected_digest:raise ValueError(f'selected processor/config snapshot mismatch: {name}')
        text=prepare_text_tokenizer(self.modeldir);config=PreTrainedConfig.from_pretrained(self.modeldir)
        assert config.type=='pi05' and config.chunk_size==50 and tuple(config.output_features['action'].shape)==(7,)
        assert {str(k):getattr(v,'value',v) for k,v in config.normalization_mapping.items()}=={'ACTION':'MEAN_STD','STATE':'MEAN_STD','VISUAL':'IDENTITY'}
        assert not config.use_visual_memory and not config.use_proprioceptive_memory and config.rtc_config is None
        config.device=str(self.device);config.compile_model=False;config.gradient_checkpointing=False;config.text_tokenizer_name=str(text.resolve())
        self.policy=PI05Policy(config)
        self.restoration=restore_checkpoint(self.policy,self.modeldir/'model.safetensors')
        (self.modeldir/'restoration.json').write_text(json.dumps(self.restoration,indent=2)+'\n')
        self.policy=self.policy.to(self.device).eval()
        for parameter in self.policy.parameters():parameter.requires_grad_(False)
        self.preprocess,self.postprocess=make_pre_post_processors(config,self.modeldir,preprocessor_overrides={'device_processor':{'device':str(self.device)},'tokenizer_processor':{'tokenizer_name':str(text.resolve())}})
        self.env_processor=LiberoProcessorStep();self.calls=0;self.weights_before=model_state_digest(self.policy);self.reset()
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)

    def reset(self,seed=0):
        """Reset inference buffers and RNG while keeping all learned weights fixed.

        Args: seed: integer. Returns: None. Raises: None.
        Design: Context: Paired rollout boundaries. Concern: Hidden episode carryover.
            Decision: Official reset plus NumPy/torch seed. Alternatives: Reuse action queues.
            Quality Attributes: Reproducibility. Trade-offs: Noise samples advance between requests.
        Evidence: Internal: Flow matching draws fresh Gaussian noise.
            External: PI05Policy.reset. Findings: No optimizer.
        Constraints: Outside active inference. Reconsider When: Concurrent requests introduced.
        References: predict. ADR: None.
        """
        self.seed=int(seed);self.torch.manual_seed(self.seed);np.random.seed(self.seed);self.policy.reset()

    def predict(self,observation,prompt):
        """Generate one50x7 physical chunk from received raw cameras and proprioception.

        Args: observation: raw RGB2+eef/quaternion/gripper; prompt: language string.
        Returns: Float32 NumPy[T=50,7] and measured metadata.
        Raises: Sensor, shape or nonfinite-output errors.
        Design: Context: Shared raw LIBERO wrapper. Concern: Double rotation and object-truth leakage.
            Decision: Exact raw allowlist; official LiberoProcessor rotates once and creates state8.
            Alternatives: Whole simulator state or custom normalization.
            Quality Attributes: Fidelity, observability. Trade-offs: Standard processor copies.
        Evidence: Internal: Checkpoint input/output schema. External: LiberoProcessorStep.
            Findings: Output is native relative pose/gripper, not joint torque.
        Constraints: No clipping/physics stepping here. Reconsider When: Robot action space changes.
        References: Native PI05 predict_action_chunk. ADR: None.
        """
        torch=self.torch;start=time.perf_counter();raw={}
        for camera,key in [('agentview_image','observation.images.image'),('robot0_eye_in_hand_image','observation.images.image2')]:
            image=np.asarray(observation[camera])
            if image.shape!=(256,256,3) or image.dtype!=np.uint8:raise ValueError('expected raw RGB256 uint8')
            raw[key]=torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0).float()/255.
        values={}
        for key,size in [('robot0_eef_pos',3),('robot0_eef_quat',4),('robot0_gripper_qpos',2)]:
            value=np.asarray(observation[key])
            if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError('invalid raw proprioception')
            values[key]=torch.as_tensor(value,dtype=torch.float32).reshape(1,size)
        raw['observation.robot_state']={'eef':{'pos':values['robot0_eef_pos'],'quat':values['robot0_eef_quat']},'gripper':{'qpos':values['robot0_gripper_qpos']}}
        processed=self.env_processor.observation(raw);processed['task']=prompt;batch=self.preprocess(processed)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        model_start=time.perf_counter()
        with torch.inference_mode():chunk=self.postprocess(self.policy.predict_action_chunk(batch))
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        model_ms=(time.perf_counter()-model_start)*1000;actions=chunk.detach().float().cpu().numpy()[0]
        if actions.shape!=(50,7) or not np.isfinite(actions).all():raise RuntimeError('invalid pi05 physical action chunk')
        self.calls+=1
        return actions,{'model':'pi05','repository':REPO,'revision':REVISION,'inference_ms':(time.perf_counter()-start)*1000,'model_inference_ms':model_ms,'chunk_length':50,'native_horizon':50,'native_n_action_steps':int(self.policy.config.n_action_steps),'flow_steps':int(self.policy.config.num_inference_steps),'dtype':str(self.policy.config.dtype),'compile_model':False,'rtc_enabled':False,'call':self.calls,'peak_cuda_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,'action_min':float(actions.min()),'action_max':float(actions.max()),'local_training':False}

    def audit(self):
        """Compare learned state before/after requests and include strict restoration proof.

        Args: None. Returns: Frozen-state and resource audit. Raises: Tensor transfer errors.
        Design: Context: No-training requirement. Concern: Eval mode alone is weak evidence.
            Decision: Full state SHA equality and all-loaded-value verification.
            Alternatives: Model name or requires_grad alone.
            Quality Attributes: Auditability. Trade-offs: Untimed full model transfer.
        Evidence: Internal: Captured startup digest and restoration813tensor proof.
            External: None. Findings: Auxiliary caches are reset outside inference.
        Constraints: Model must still be open. Reconsider When: Persistent state changes.
        References: model_state_digest. ADR: None.
        """
        after=model_state_digest(self.policy)
        return {'model':'pi05','repository':REPO,'revision':REVISION,'weights_before_sha256':self.weights_before,'weights_after_sha256':after,'weights_unchanged':after==self.weights_before,'requires_grad_parameter_count':sum(p.numel() for p in self.policy.parameters() if p.requires_grad),'parameter_count':sum(p.numel() for p in self.policy.parameters()),'peak_cuda_bytes':self.torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,'native_horizon':int(self.policy.config.chunk_size),'flow_steps':int(self.policy.config.num_inference_steps),'processor_snapshot_sha256':SNAPSHOT_SHA256,'local_training':False,'calls':self.calls,'restoration':self.restoration}

    def close(self):
        """Release live tensors and cached CUDA allocation before another model loads.

        Args: None. Returns: None. Raises: None.
        Design: Context: Single16GB GPU comparison. Concern: Residual model memory.
            Decision: Remove all policy/processor references and collect/empty cache.
            Alternatives: Concurrent model load. Quality Attributes: Isolation. Trade-offs: Reload required.
        Evidence: Internal: Root controls model order. External: torch.cuda.empty_cache.
            Findings: Live references must be removed first.
        Constraints: No calls after close. Reconsider When: MultiGPU deployment is used.
        References: FrozenPI05. ADR: None.
        """
        self.policy=None;self.preprocess=None;self.postprocess=None;gc.collect()
        if self.device.type=='cuda':self.torch.cuda.empty_cache()
