"""Run the fixed official LIBERO pi0-FAST checkpoint as a frozen chunk generator.

Args: Model directory, CUDA device, raw LIBERO camera/proprio observations and prompt.
Returns: Unnormalized continuous action chunks and latency / memory metadata.
Raises: RuntimeError for missing assets or invalid token decoding; ValueError for bad sensor inputs.
Design:
    Context: Compare actual SmolVLA and pi0-FAST inference under one control clock.
    Concern: Base models, changed token vocabularies and old normalization regressions invalidate comparison.
    Decision: Official LeRobot pi0_fast checkpoint, native pre/postprocessors and KV-cache sampler.
    Alternatives: openpi JAX old LIBERO requires a z-score compatibility fix and separate dependencies.
    Quality Attributes: Fidelity, provenance, inspectable timing. Trade-offs: 7.73GB checkpoint.
Evidence:
    Internal: Fixed checkpoint processor config specifies MEAN_STD, chunk10, image224 and action7.
    External: https://huggingface.co/lerobot/pi0fast-libero and official LeRobot implementation.
    Findings: PaliGemma SentencePiece vocabulary is publicly available from Google's big_vision bucket.
    Metrics: Generated chunk shape, synchronized inference time and peak CUDA allocation.
    Confidence: Runtime task success must be measured independently.
Constraints: No training; no alternative model substitution; no automatic quantization.
Reconsider When: Checkpoint, tokenizer or LeRobot revision changes.
References: artifacts/realtime-vla/models/pi0fast/provenance.json.
ADR: None.
"""
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.request

import numpy as np

# Transformers copies fixed local AutoProcessor code into its dynamic-module
# cache. Keep that cache inside the authorized workspace rather than $HOME.
os.environ.setdefault('HF_MODULES_CACHE',str(Path(__file__).resolve().parents[2]/'.cache/realtime-vla/hf-modules'))

TEXT_MODEL_URL='https://storage.googleapis.com/big_vision/paligemma_tokenizer.model'
TEXT_MODEL_SHA='8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'
FAST_REPO='jadechoghari/tokenizer-lib-mean'
FAST_REVISION='79ae83e3cbd8786dcb84b628569f8d076ca8151e'


def prepare_tokenizers(model_dir):
    """Restore exact public PaliGemma / FAST vocabularies and test token identity.

    Args: model_dir: Fixed checkpoint directory.
    Returns: Local text and action tokenizer paths.
    Raises: AssertionError on digest/vocabulary/token mismatch; network errors on unavailable assets.
    Design:
        Context: Google's HF model tokenizer URL is gated but official big_vision tokenizer is public.
        Concern: An unrelated tokenizer could silently corrupt FAST autoregressive decoding.
        Decision: Convert the official SP model with Transformers' GemmaConverter and verify IDs.
        Alternatives: Requesting a private token or using a different vocabulary is unnecessary.
        Quality Attributes: Reproducibility, content identity. Trade-offs: One local format conversion.
    Evidence:
        Internal: Fixed GCS file SHA and SentencePiece/fast-tokenizer probe equality.
        External: Google big_vision and official Transformers GemmaConverter.
        Findings: This is the same 257152-entry vocabulary used by openpi, not alternate weights.
        Metrics: Vocabulary size, BOS/EOS/PAD IDs and probe token arrays.
        Confidence: Exact probe and vocabulary identities are checked before model load.
    Constraints: No gated model weights are fetched or redistributed.
    Reconsider When: A checkpoint uses a different text tokenizer.
    References: Model config text_tokenizer_name google/paligemma-3b-pt-224.
    ADR: None.
    """
    import sentencepiece as spm
    from tokenizers import processors
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import GemmaConverter, GemmaSentencePieceExtractor

    class SentencePieceCompatExtractor(GemmaSentencePieceExtractor):
        """Restore the SP handle expected by the official Gemma merge extractor.

        Args: model: Exact original SentencePiece file. Returns: Extractor instance.
        Raises: SentencePiece errors for corrupt files.
        Design:
            Context: Transformers5.4 changed the parent extractor to protobuf-only.
            Concern: Gemma's inherited extract still reads self.sp and raises AttributeError.
            Decision: Add the missing original SP handle while reusing official merge logic.
            Alternatives: Editing installed package or using unrelated tokens.
            Quality Attributes: Fidelity. Trade-offs: Small format-conversion compatibility shim.
        Evidence:
            Internal: Full piece IDs and prompt encodings checked after conversion.
            External: Transformers5.4 GemmaSentencePieceExtractor.extract.
            Findings: No model source or token ranks changed.
            Metrics: 257152 IDs checked. Confidence: Verified at runtime.
        Constraints: Conversion only. Reconsider When: Upstream fixes the extractor.
        References: tokenizer-validation.json. ADR: None.
        """
        def __init__(self,model):
            """Keep official initialization and restore its missing native SP member.

            Args: model: Original file path. Returns: None. Raises: Invalid model errors.
            Design: Context: One-time conversion. Concern: Upstream API mismatch.
                Decision: Restore only missing member. Alternatives: Global monkeypatch.
                Quality Attributes: Isolation. Trade-offs: None for token IDs.
            Evidence: Internal: Exact conversion checks. External: Official extractor source.
                Findings: Native SP handles original pieces. Metrics: ID equality. Confidence: High.
            Constraints: No installed package edits. Reconsider When: Version changes.
            References: SentencePieceCompatExtractor. ADR: None.
            """
            super().__init__(model)
            self.sp=spm.SentencePieceProcessor(model_file=str(model))

    root=Path(model_dir)/'tokenizer';root.mkdir(parents=True,exist_ok=True)
    original=root/'paligemma_tokenizer.model'
    if not original.exists():urllib.request.urlretrieve(TEXT_MODEL_URL,original)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==TEXT_MODEL_SHA
    reference=spm.SentencePieceProcessor(model_file=str(original))
    assert reference.vocab_size()==257152
    assert (reference.pad_id(),reference.eos_id(),reference.bos_id(),reference.unk_id())==(0,1,2,3)
    text_path=root/'text'
    if not (text_path/'tokenizer.json').exists():
        slow=SimpleNamespace(vocab_file=str(original),pad_token='<pad>',eos_token='<eos>',bos_token='<bos>',unk_token='<unk>',add_prefix_space=False)
        converter=GemmaConverter(slow);converter.SpmExtractor=SentencePieceCompatExtractor
        backend=converter.converted()
        backend.post_processor=processors.TemplateProcessing(single='<bos> $A',pair='<bos> $A $B',special_tokens=[('<bos>',2)])
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<bos>',eos_token='<eos>',pad_token='<pad>',unk_token='<unk>',padding_side='right',clean_up_tokenization_spaces=False)
        tokenizer.save_pretrained(text_path)
    tokenizer=AutoTokenizer.from_pretrained(text_path,local_files_only=True)
    assert tokenizer.vocab_size==reference.vocab_size()
    assert (tokenizer.pad_token_id,tokenizer.eos_token_id,tokenizer.bos_token_id,tokenizer.unk_token_id)==(0,1,2,3)
    # IDs used by the autoregressive action head must retain their original piece strings.
    differences=[]
    for index in range(reference.vocab_size()):
        actual=tokenizer.convert_ids_to_tokens(index);expected=reference.id_to_piece(index)
        if actual!=expected:differences.append((index,expected,actual))
    assert all(before=='<0x09>' and after=='\t' for _,before,after in differences),differences[:5]
    probes=['Action: ','pick up the black bowl','Task: pick up the black bowl, State: 1 2 3 4 5 6 7 8;\n','put the red cup on the plate','open the top drawer','Task: move bowl, State: 0 255 127 9 89 200 1 0;\n']
    probe_records=[]
    for prompt in probes:
        expected=reference.encode(prompt,out_type=int)
        actual=tokenizer.encode(prompt,add_special_tokens=False)
        assert actual==expected,(prompt,actual,expected)
        assert tokenizer.encode(prompt,add_special_tokens=True)==[reference.bos_id()]+expected
        probe_records.append({'text':prompt,'ids':actual})
    action_path=root/'action';action_path.mkdir(parents=True,exist_ok=True)
    action_manifest=action_path/'provenance.json'
    if action_manifest.exists():
        pinned=json.loads(action_manifest.read_text())
        assert pinned['repo']==FAST_REPO and pinned['revision']==FAST_REVISION
        listing=[{'type':'file','path':record['path'],'sha256':record['sha256']} for record in pinned['files']]
    else:
        api=f'https://huggingface.co/api/models/{FAST_REPO}/tree/{FAST_REVISION}?recursive=true'
        with urllib.request.urlopen(api,timeout=60) as response:listing=json.load(response)
    records=[]
    for item in listing:
        if item['type']!='file' or item['path']=='.gitattributes':continue
        path=action_path/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(f'https://huggingface.co/{FAST_REPO}/resolve/{FAST_REVISION}/{item["path"]}',timeout=60) as response:path.write_bytes(response.read())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if item.get('sha256'):assert digest==item['sha256'],item['path']
        records.append({'file':item['path'],'sha256':digest})
    evidence={'text_model_url':TEXT_MODEL_URL,'text_model_sha256':TEXT_MODEL_SHA,'vocabulary_size':reference.vocab_size(),'special_ids':{'pad':0,'eos':1,'bos':2,'unk':3},'all_vocabulary_pieces_verified':True,'official_converter_tab_aliases':differences,'probes':probe_records,'bos_behavior':'processor text uses BOS, no EOS; action prefix uses add_special_tokens=False','fast_repository':FAST_REPO,'fast_revision':FAST_REVISION,'fast_files':records,'license':'PaliGemma/Gemma upstream terms; no claim of MIT model license'}
    (root/'tokenizer-validation.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return text_path,action_path


def weight_digest(policy):
    """Hash registered parameters and buffers without modifying inference state.

    Args: policy: loaded official torch module. Returns: SHA256 string.
    Raises: RuntimeError on failed tensor transfer.
    Design:
        Context: The experiment forbids local training.
        Concern: Checking only checkpoint bytes misses in-memory mutation.
        Decision: Hash names, dtype, shape and raw tensor bytes before/after evaluation.
        Alternatives: requires_grad=False alone is structural evidence only.
        Quality Attributes: Auditability. Trade-offs: Several GB copied to CPU during audits.
    Evidence:
        Internal: Official policy state_dict. External: PyTorch state_dict semantics.
        Findings: BF16 is viewed as bytes before numpy conversion.
        Metrics: Exact SHA equality. Confidence: High for persistent registered state.
    Constraints: Audit outside measured inference timing.
    Reconsider When: State cache becomes a persistent buffer.
    References: FrozenPI0Fast.audit.
    ADR: None.
    """
    import torch
    digest=hashlib.sha256()
    for name,tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class DecodeErrorRecorder(logging.Handler):
    """Collect upstream FAST decode warnings so silent zero fallback is observable.

    Args: None. Returns: Logging handler. Raises: None.
    Design:
        Context: Upstream decoder can emit zeros after malformed token sequences.
        Concern: Such chunks must not be mistaken for valid model commands.
        Decision: Retain exact error messages and fail the inference request.
        Alternatives: Silent fallback corrupts accuracy comparison.
        Quality Attributes: Integrity. Trade-offs: Decoder anomalies stop a trial.
    Evidence:
        Internal: LeRobot decode_actions_with_fast logs Error decoding tokens.
        External: Official implementation. Findings: Fallback otherwise returns a numeric zero chunk.
        Metrics: Decode warning count. Confidence: High for matching warning path.
    Constraints: Does not modify generated tokens or decoder arithmetic.
    Reconsider When: Upstream decoder error handling changes.
    References: PI0FastPolicy.decode_actions_with_fast.
    ADR: None.
    """
    def __init__(self):
        """Initialize per-call decoder diagnostic storage.

        Args: None. Returns: None. Raises: None.
        Design:
            Context: One policy object is used serially. Concern: Diagnostics cannot cross calls.
            Decision: Explicit message list. Alternatives: Global log scraping.
            Quality Attributes: Traceability. Trade-offs: Serial inference only.
        Evidence:
            Internal: predict clears messages before inference. External: None.
            Findings: No model behavior is changed. Metrics: Message count. Confidence: High.
        Constraints: Warning level only. Reconsider When: Concurrent prediction is added.
        References: FrozenPI0Fast.predict.
        ADR: None.
        """
        super().__init__(level=logging.WARNING);self.messages=[]

    def emit(self,record):
        """Capture only token-decode errors from the official policy.

        Args: record: logging record. Returns: None. Raises: None.
        Design:
            Context: Other library warnings may be benign. Concern: False inference failure.
            Decision: Match the upstream decoder error prefix.
            Alternatives: Treating every warning as failure is overbroad.
            Quality Attributes: Precision. Trade-offs: Prefix must follow upstream.
        Evidence:
            Internal: decode_actions_with_fast source. External: None.
            Findings: Specific warning means zero fallback. Metrics: Number of anomalies. Confidence: High.
        Constraints: No side effects on logging. Reconsider When: Upstream text changes.
        References: DecodeErrorRecorder.
        ADR: None.
        """
        message=record.getMessage()
        if message.startswith('Error decoding tokens:'):self.messages.append(message)


class FrozenPI0Fast:
    """Own official pi0-FAST inference and produce decoded physical action chunks.

    Args: modeldir: fixed official model; device: CUDA string; compile_model: explicit condition.
    Returns: Predictor instance with last_metadata for every call.
    Raises: RuntimeError when model/tokenizer cannot be loaded or CUDA is unavailable.
    Design:
        Context: Real-time scheduler consumes chunks independently of simulation stepping.
        Concern: Policy-internal select_action queues would hide chunk generation latency.
        Decision: Call official predict_action_chunk once per request and time synchronized CUDA.
        Alternatives: Timing select_action averages cached actions with expensive generation.
        Quality Attributes: Fair timing, fidelity. Trade-offs: Scheduler owns chunk consumption.
    Evidence:
        Internal: Model config chunk10, MEAN_STD and KV cache.
        External: Official LeRobot PI0FastPolicy.
        Findings: No local training or quantization is needed for the comparison condition.
        Metrics: Latency, decoded chunk shape, peak allocated GPU memory.
        Confidence: Task success must be established by rollout.
    Constraints: Model is held separately from SmolVLA on the 16GB GPU.
    Reconsider When: Concurrent model serving is required.
    References: Model provenance and tokenizer-validation.json.
    ADR: None.
    """
    def __init__(self,modeldir='artifacts/realtime-vla/models/pi0fast',device='cuda',compile_model=False):
        """Load frozen networks and fixed official processor statistics.

        Args: modeldir, device, compile_model: See class.
        Returns: None. Raises: ImportError / RuntimeError for incompatible libraries.
        Design:
            Context: Use the same preprocessing as checkpoint evaluation.
            Concern: Gated tokenizer calls and task statistics need deterministic local resolution.
            Decision: Override only tokenizer paths and compile condition, preserve learned weights/stats.
            Alternatives: Ad-hoc normalization would break task compatibility.
            Quality Attributes: Reproducibility. Trade-offs: Public tokenizer conversion step.
        Evidence:
            Internal: Tokenizer identity is checked before policy initialization.
            External: make_pre_post_processors API.
            Findings: State mean/std is8D before the model pads to32.
            Metrics: GPU allocation after load. Confidence: Empirical.
        Constraints: Native BF16/fp32 mixture; no quantization.
        Reconsider When: Processor schema changes.
        References: config.json and policy_preprocessor.json.
        ADR: None.
        """
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        from lerobot.processor.env_processor import LiberoProcessorStep
        self.torch=torch;self.device=torch.device(device);self.modeldir=Path(modeldir).resolve()
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        text_path,action_path=prepare_tokenizers(self.modeldir)
        config=PreTrainedConfig.from_pretrained(self.modeldir)
        assert config.type=='pi0_fast' and config.chunk_size==10 and config.n_action_steps==10
        config.device=str(self.device);config.compile_model=compile_model;config.gradient_checkpointing=False
        config.text_tokenizer_name=str(text_path.resolve());config.action_tokenizer_name=str(action_path.resolve())
        # The upstream from_pretrained catches loading errors and can return random
        # initialization. Reuse its exact remapping with an uncaught strict load.
        from safetensors.torch import load_file
        checkpoint=self.modeldir/'model.safetensors'
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        self.policy=PI0FastPolicy(config)
        stored=load_file(str(checkpoint),device='cpu')
        fixed=self.policy._fix_pytorch_state_dict_keys(stored,config)
        remapped={key if key.startswith('model.') else 'model.'+key:value for key,value in fixed.items()}
        result=self.policy.load_state_dict(remapped,strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        self.checkpoint_tensor_count=len(remapped)
        del stored,fixed,remapped
        self.policy=self.policy.to(self.device).eval()
        for parameter in self.policy.parameters():parameter.requires_grad_(False)
        self.preprocess,self.postprocess=make_pre_post_processors(config,self.modeldir,preprocessor_overrides={'device_processor':{'device':str(self.device)},'tokenizer_processor':{'tokenizer_name':str(text_path.resolve())},'action_tokenizer_processor':{'action_tokenizer_name':str(action_path.resolve()),'paligemma_tokenizer_name':str(text_path.resolve()),'trust_remote_code':True}})
        self.env_processor=LiberoProcessorStep()
        self.decode_errors=DecodeErrorRecorder()
        # Upstream uses logging.warning(...) rather than a module-scoped logger.
        self.decode_logger=logging.getLogger();self.decode_logger.addHandler(self.decode_errors)
        self.last_metadata={};self.calls=0;self.weights_before_sha256=weight_digest(self.policy);self.reset()
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)

    def reset(self,seed=0):
        """Reset policy buffers and sampling seed at episode boundaries.

        Args: seed: integer. Returns: None. Raises: None.
        Design:
            Context: Paired trials must start with equivalent policy state.
            Concern: Cache from a preceding episode would confound outcomes.
            Decision: Official reset and torch seed. Alternatives: Reusing queue state.
            Quality Attributes: Reproducibility. Trade-offs: Model load is not timed as inference.
        Evidence:
            Internal: Chunk sampler uses native greedy temperature0 config.
            External: PI0FastPolicy.reset.
            Findings: No optimization occurs. Metrics: Episode seed. Confidence: High.
        Constraints: One thread per model. Reconsider When: Sampling policy changes.
        References: FrozenPI0Fast.
        ADR: None.
        """
        self.torch.manual_seed(seed);np.random.seed(seed);self.policy.reset()

    def predict(self,observation,prompt):
        """Generate one native action10x7 chunk from raw LIBERO sensors.

        Args: observation: raw two-camera and eef/quaternion/gripper state; prompt: task string.
        Returns: Tuple of numpy float32[T=10,A=7] action chunk and measured metadata.
        Raises: ValueError for malformed sensors; RuntimeError for decode fallback or nonfinite output.
        Design:
            Context: Simulator supplies unflipped raw camera arrays.
            Concern: Duplicate flips, object truth inputs or cached single-action timing.
            Decision: Official LiberoProcessorStep performs180degree flip and state conversion once.
            Alternatives: Own coordinate transforms risk model/environment mismatch.
            Quality Attributes: Fidelity. Trade-offs: Fixed sensor allowlist.
        Evidence:
            Internal: Official processor uses eef pos3+axis-angle3+gripper2.
            External: LeRobot LIBERO evaluation convention.
            Findings: Output normalization is undone by official postprocessor.
            Metrics: Full request latency and peak GPU memory. Confidence: Measured per call.
        Constraints: Only camera/proprio inputs, never object pose or future observations.
        Reconsider When: Environment schema changes.
        References: LiberoProcessorStep and PI0FastPolicy.predict_action_chunk.
        ADR: None.
        """
        torch=self.torch
        start=time.perf_counter()
        raw={}
        for source,target in [('agentview_image','observation.images.image'),('robot0_eye_in_hand_image','observation.images.image2')]:
            image=np.asarray(observation[source])
            if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:raise ValueError('expected raw HWC RGB8 camera')
            raw[target]=torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0).float()/255.
        raw['observation.robot_state']={'eef':{'pos':torch.as_tensor(np.asarray(observation['robot0_eef_pos']),dtype=torch.float32).reshape(1,3),'quat':torch.as_tensor(np.asarray(observation['robot0_eef_quat']),dtype=torch.float32).reshape(1,4)},'gripper':{'qpos':torch.as_tensor(np.asarray(observation['robot0_gripper_qpos']),dtype=torch.float32).reshape(1,2)}}
        processed=self.env_processor.observation(raw);processed['task']=prompt
        batch=self.preprocess(processed)
        self.decode_errors.messages.clear()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        inference_start=time.perf_counter()
        with torch.inference_mode():
            output=self.policy.predict_action_chunk(batch)
            physical=self.postprocess(output)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        inference_seconds=time.perf_counter()-inference_start
        if self.decode_errors.messages:raise RuntimeError('FAST decoder fallback: '+'; '.join(self.decode_errors.messages))
        chunk=physical.detach().float().cpu().numpy()[0]
        if chunk.shape!=(10,7) or not np.isfinite(chunk).all():raise RuntimeError(f'invalid pi0-FAST chunk {chunk.shape}')
        elapsed=time.perf_counter()-start;self.calls+=1
        peak_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None
        self.last_metadata={'model':'pi0-FAST','chunk_size':10,'chunk_length':10,'action_dim':7,'inference_ms':elapsed*1000,'model_inference_ms':inference_seconds*1000,'inference_seconds':inference_seconds,'end_to_end_seconds':elapsed,'call':self.calls,'action_min':float(chunk.min()),'action_max':float(chunk.max()),'kv_cache':bool(self.policy.config.use_kv_cache),'temperature':float(self.policy.config.temperature),'max_decoding_steps':int(self.policy.config.max_decoding_steps),'autoregressive_token_count':None,'compile_model':bool(self.policy.config.compile_model),'dtype':str(self.policy.config.dtype),'peak_cuda_bytes':peak_bytes,'peak_cuda_allocated_bytes':peak_bytes,'decode_errors':list(self.decode_errors.messages),'local_training':False}
        return chunk,self.last_metadata

    def audit(self):
        """Verify frozen model state after a measured rollout.

        Args: None. Returns: Before/after digests, frozen flag and peak GPU allocation.
        Raises: RuntimeError on failed tensor transfer.
        Design:
            Context: Pretrained inference is the controlled comparison condition.
            Concern: Learned state could change in memory without changing downloaded files.
            Decision: Recompute the same state digest outside benchmark timing.
            Alternatives: Trusting eval mode alone is weaker evidence.
            Quality Attributes: Traceability. Trade-offs: CPU transfer overhead.
        Evidence:
            Internal: weights_before_sha256 captured after initialization. External: None.
            Findings: No optimizer is instantiated. Metrics: Exact state hash equality. Confidence: High.
        Constraints: Call before close. Reconsider When: State format changes.
        References: weight_digest.
        ADR: None.
        """
        after=weight_digest(self.policy)
        return {'weights_before_sha256':self.weights_before_sha256,'weights_after_sha256':after,'weights_unchanged':after==self.weights_before_sha256,'peak_cuda_bytes':self.torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,'checkpoint_load_strict':True,'checkpoint_tensor_count':self.checkpoint_tensor_count,'local_training':False}

    def close(self):
        """Release the model and CUDA allocations before the comparison model is loaded.

        Args: None. Returns: None. Raises: None.
        Design:
            Context: One16GB GPU hosts models sequentially.
            Concern: Old tensor references can create a false OOM for the second model.
            Decision: Remove logging hook and model refs, collect and empty CUDA cache.
            Alternatives: Concurrent loading is outside the hardware comparison condition.
            Quality Attributes: Resource isolation. Trade-offs: Reload needed between model families.
        Evidence:
            Internal: Root scheduler controls model execution order.
            External: torch.cuda.empty_cache semantics.
            Findings: Live tensors must be removed before cached allocations are released.
            Metrics: Post-close allocation. Confidence: Runtime measured.
        Constraints: No use after close. Reconsider When: Multi-GPU available.
        References: FrozenPI0Fast.
        ADR: None.
        """
        self.decode_logger.removeHandler(self.decode_errors)
        self.policy=None;self.preprocess=None;self.postprocess=None
        gc.collect()
        if self.device.type=='cuda':self.torch.cuda.empty_cache()
