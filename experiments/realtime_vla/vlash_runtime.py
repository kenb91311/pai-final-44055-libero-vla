"""Run the author's dedicated asynchronous LIBERO VLASH policy with complete restoration.

Args: Fixed local model directory, device, approved raw LIBERO sensors and task prompt.
Returns: Native50x7 continuous actions and latency/integrity metadata.
Raises: Source, backend, tokenizer, checkpoint, schema or finite-output errors.
Design:
    Context: VLASH is a separately trained policy, not an input trick for ordinary pi0.5.
    Concern: Its legacy loader ignores missing keys and its backend differs from current LeRobot.
    Decision: Isolated exact author backend, verified source, metadata aliases and strict restoration.
    Alternatives: Loading VLASH weights through the contemporary pi05 implementation.
    Quality Attributes: Fidelity, isolation. Trade-offs: An additional Python overlay is required.
Evidence:
    Internal: Pinned checkpoint/source manifests and independent frozen-state audits.
    External: mit-han-lab/vlash sim/libero and vlash-pi05-libero-async5 public checkpoint.
    Findings: Checkpoint contains824stored tensors plus255shared aliases, including74normalizer values.
    Metrics: Exact tensor bytes, all key coverage, latency and peak CUDA bytes.
    Confidence: API validation and task evaluation are reported separately.
Constraints: No training, quantization, object truth, silent fallback or old-runtime edits.
Reconsider When: Author publishes weight terms or a different backend/checkpoint pairing.
References: docs/reference/realtime-continuity-methods.md; vlash-source-manifest.json. ADR: None.
"""
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
from importlib.metadata import distribution,version

import numpy as np

try:
    from .libero_env import canonical_state
    from .pi05_runtime import prepare_text_tokenizer,model_state_digest
    from .prepare_vlash_source import prepare_source,SOURCE_REVISION
except ImportError:
    from libero_env import canonical_state
    from pi05_runtime import prepare_text_tokenizer,model_state_digest
    from prepare_vlash_source import prepare_source,SOURCE_REVISION

ROOT=Path(__file__).resolve().parents[2]
REPO='mit-han-lab/vlash-pi05-libero-async5'
REVISION='c85d5962a7b823f9f391f97d60227d6ca8bd7bc7'
WEIGHT_SHA='8dd3bf4766dac481933ced7c8de5fbfc31c486e665cca33e6ddf970792a46a2e'
WEIGHT_BYTES=7472953952
CONFIG_SHA='0458a362b574c2cb088ebf3767c9eacb1f040b9cb680a89b594379a4fe36e5f7'
TRANSFORMERS_REVISION='dcddb970176382c0fcf4521b0c0e6fc15894dfe0'
EXPECTED_VERSIONS={'lerobot':'0.4.1','transformers':'4.53.3','tokenizers':'0.21.4',
                   'huggingface-hub':'0.34.4','torch':'2.7.1+cu128','numpy':'2.2.6'}


def checkpoint_header(checkpoint):
    """Read small safetensors structural metadata without mapping tensor payloads.

    Args: checkpoint: fixed file path. Returns: Header dictionary. Raises: IO/format errors.
    Design: Context: Large weights. Concern: Preflight must avoid multiGB memory pressure.
        Decision: Read only the length-prefixed JSON. Alternatives: load_file during metadata queries.
        Quality Attributes: Low resource use. Trade-offs: Actual values are checked at restoration.
    Evidence: Internal: Native safetensors header. External: Safetensors file format.
        Findings: Alias metadata explains removed tied tensors. Metrics: Header bytes. Confidence: Structural.
    Constraints: No GPU/model construction. Reconsider When: Checkpoint format changes.
    References: restore_native_checkpoint. ADR: None.
    """
    with Path(checkpoint).open('rb') as source:
        size=struct.unpack('<Q',source.read(8))[0]
        if size>16*1024*1024:raise ValueError('unreasonable VLASH safetensors header')
        return json.loads(source.read(size))


def verify_backend():
    """Reject backend substitution before importing the original author's model.

    Args: None. Returns: Package/version provenance. Raises: Version/source errors.
    Design: Context: Legacy patched Gemma AdaRMS APIs. Concern: Same version label can hide different source.
        Decision: Exact versions plus Transformers direct_url revision check.
        Alternatives: Broad dependency ranges or mathematical compatibility shims.
        Quality Attributes: Fidelity. Trade-offs: Dedicated overlay.
    Evidence: Internal: requirements-vlash.lock. External: Author pyproject fixed Git dependency.
        Findings: Existing baseline Transformers5.4 is intentionally unsupported here.
        Metrics: Version equality. Confidence: Backend identity.
    Constraints: Never modify installed packages. Reconsider When: Author changes the backend.
    References: scripts/setup_vlash.sh. ADR: None.
    """
    actual={name:version(name) for name in EXPECTED_VERSIONS}
    if actual!=EXPECTED_VERSIONS:raise RuntimeError(f'Use .venv-vlash: backend mismatch {actual}')
    direct=json.loads(distribution('transformers').read_text('direct_url.json') or '{}')
    if TRANSFORMERS_REVISION not in direct.get('url',''):
        raise RuntimeError('VLASH requires the exact author Transformers Git revision')
    return {'versions':actual,'transformers_git_revision':TRANSFORMERS_REVISION}


def load_author_components(modeldir):
    """Import verified author code and inject only an identical local text tokenizer.

    Args: modeldir: local VLASH model directory. Returns: Policy class, config and provenance.
    Raises: Source/backend/config/tokenizer errors.
    Design: Context: Hardcoded gated tokenizer name in original policy constructor.
        Concern: Secrets or vocabulary substitutions would prevent open reproduction.
        Decision: Exact public Google vocabulary, original converter and module-local factory injection.
        Alternatives: Global Transformers patch or unrelated tokenizer.
        Quality Attributes: Isolation, identity. Trade-offs: Small dependency-injection wrapper.
    Evidence: Internal: Text-validation records all257152 IDs and six prompt encodings.
        External: Public Google PaliGemma SP and original GemmaConverter.
        Findings: No action tokenizer is used; this is continuous flow matching.
        Metrics: Vocabulary/probes. Confidence: Exact verification before inference.
    Constraints: Original model source remains byte-identical. Reconsider When: Tokenizer API changes.
    References: prepare_text_tokenizer; source manifest. ADR: None.
    """
    backend=verify_backend();source=prepare_source(verify_only=True)
    source_dir=ROOT/'.cache/vlash-source'/SOURCE_REVISION
    if str(source_dir) not in sys.path:sys.path.insert(0,str(source_dir))
    from lerobot.configs.policies import PreTrainedConfig
    from transformers import AutoTokenizer
    from vlash.policies.pi05.configuration_pi05 import PI05Config
    import vlash.policies.pi05.modeling_pi05 as native
    # Identical registration to author vlash/configs/__init__.py, without training/hardware imports.
    PreTrainedConfig._choice_registry['pi05']=PI05Config
    text=prepare_text_tokenizer(modeldir)
    tokenizer=AutoTokenizer.from_pretrained(text,local_files_only=True)

    class LocalTextTokenizer:
        """Supply the exact local PaliGemma tokenizer to one author module.

        Args: None. Returns: Factory namespace. Raises: Unsupported tokenizer-name errors.
        Design: Context: Author hardcodes one public gated identifier. Concern: Global patching.
            Decision: Module-local dependency injection. Alternatives: Edit original source.
            Quality Attributes: Isolation. Trade-offs: One wrapper.
        Evidence: Internal: Exact full-vocabulary check. External: Google PaliGemma tokenizer.
            Findings: Model math is untouched. Metrics: None. Confidence: Identity verified.
        Constraints: Only the original identifier is allowed. Reconsider When: Author changes identifiers.
        References: load_author_components. ADR: None.
        """
        @staticmethod
        def from_pretrained(name,**kwargs):
            """Return the verified tokenizer for the author's single supported identifier.

            Args: name: original identifier; kwargs: original unused loading options.
            Returns: Verified local tokenizer. Raises: ValueError for any other name.
            Design: Context: Constructor injection. Concern: Accidental asset substitution.
                Decision: Exact name guard. Alternatives: Unrestricted factory.
                Quality Attributes: Fidelity. Trade-offs: None.
            Evidence: Internal: prepare_text_tokenizer checks. External: Author constructor.
                Findings: No network call. Metrics: None. Confidence: Exact mapping.
            Constraints: No alternate model vocabulary. Reconsider When: Constructor changes.
            References: LocalTextTokenizer. ADR: None.
            """
            if name!='google/paligemma-3b-pt-224':raise ValueError('unexpected author tokenizer identifier')
            return tokenizer

    native.AutoTokenizer=LocalTextTokenizer
    config=PreTrainedConfig.from_pretrained(modeldir)
    if config.type!='pi05' or config.state_cond or config.chunk_size!=50 or config.n_action_steps!=50:
        raise ValueError('unsupported VLASH checkpoint architecture')
    if config.num_inference_steps!=10 or config.dtype!='bfloat16':raise ValueError('unexpected flow/dtype config')
    if tuple(config.output_features['action'].shape)!=(7,):raise ValueError('unexpected VLASH action schema')
    config.compile_model=False;config.gradient_checkpointing=False;config.fuse_qkv=False;config.fuse_gate_up=False
    return native.PI05Policy,config,{'backend':backend,'author_source':source,
                                   'text_tokenizer_validation':json.loads((Path(modeldir)/'tokenizer/text-validation.json').read_text())}


def expand_aliases(header,expected,stored):
    """Require complete schema coverage and original sharing for every removed tied tensor.

    Args: header: safetensors metadata; expected: original keep_vars state; stored: native tensors.
    Returns: Complete state mapping and alias map. Raises: Any missing, unexpected or unshared key.
    Design: Context: Safetensors omits duplicate tied storage. Concern: strict=False masks missing weights.
        Decision: Expand only explicit checkpoint aliases matching original parameter identities.
        Alternatives: Ignore missing keys or guess prefix conversions.
        Quality Attributes: Integrity. Trade-offs: Exact native architecture required.
    Evidence: Internal:824stored tensors/255aliases. External: safetensors save_model metadata convention.
        Findings: No OpenPI conversion is needed for this native VLASH checkpoint.
        Metrics: All key coverage. Confidence: Structural identity.
    Constraints: Every runtime key must be accounted for. Reconsider When: Native sharing changes.
    References: restore_native_checkpoint. ADR: None.
    """
    aliases={};complete=dict(stored)
    for alias,target in header.get('__metadata__',{}).items():
        if alias not in expected or target not in stored:raise ValueError(f'unsupported checkpoint alias {alias}')
        if id(expected[alias])!=id(expected[target]):raise ValueError(f'author architecture does not share {alias}')
        if alias in complete:raise ValueError('alias duplicates stored tensor')
        aliases[alias]=target;complete[alias]=stored[target]
    if set(complete)!=set(expected):
        raise RuntimeError(f'VLASH state coverage mismatch: missing={sorted(set(expected)-set(complete))}; unexpected={sorted(set(complete)-set(expected))}')
    for key,value in complete.items():
        if tuple(value.shape)!=tuple(expected[key].shape):raise ValueError(f'VLASH shape mismatch: {key}')
    return complete,aliases


def bind_aliases(policy,aliases):
    """Restore original Parameter object sharing after zero-allocation assign=True loading.

    Args: policy: loaded native module; aliases: validated explicit target mapping.
    Returns: None. Raises: Nonparameter/nonbuffer or alias target errors.
    Design: Context: Meta construction avoids allocating random multiGB tensors.
        Concern: assign=True can create distinct Parameter objects for aliases.
        Decision: Rebind only aliases already proven shared in the untouched original architecture.
        Alternatives: Duplicate memory or rely on incomplete restoration.
        Quality Attributes: Memory efficiency, fidelity. Trade-offs: Explicit alias traversal.
    Evidence: Internal: Preload keep_vars identity checks. External: PyTorch assign semantics.
        Findings: Checkpoint aliases are not independently learned values.
        Metrics: Exact object identities after binding. Confidence: Checked at restoration.
    Constraints: No new sharing may be invented. Reconsider When: Module parameter registration changes.
    References: expand_aliases. ADR: None.
    """
    for alias,target in aliases.items():
        target_module,target_name=target.rsplit('.',1);alias_module,alias_name=alias.rsplit('.',1)
        source=policy.get_submodule(target_module);dest=policy.get_submodule(alias_module)
        if target_name in source._parameters and alias_name in dest._parameters:
            dest._parameters[alias_name]=source._parameters[target_name]
        elif target_name in source._buffers and alias_name in dest._buffers:
            dest._buffers[alias_name]=source._buffers[target_name]
        else:raise RuntimeError(f'unsupported alias binding: {alias}')


def materialize_nonpersistent_buffers(policy):
    """Recreate only deterministic author/HF caches omitted from the checkpoint.

    Args: policy: meta-constructed original module with assigned persistent state.
    Returns: Per-buffer provenance. Raises: Any unknown or remaining meta buffer.
    Design: Context: Meta construction also creates nonpersistent RoPE/position caches on meta.
        Concern: They have no checkpoint values and cannot be copied out of meta.
        Decision: Invoke original cache constructors and the original positional arange expression.
        Alternatives: Random allocation, guessed cache values or allocating a complete random policy.
        Quality Attributes: Fidelity, memory efficiency. Trade-offs: Small deterministic CPU cache construction.
    Evidence: Internal: Fixed author rope.py and patched Transformers Gemma/SigLIP source.
        External: PyTorch nonpersistent buffers are intentionally absent from state_dict.
        Findings: These caches contain no learned parameters; all learned state remains strictly restored.
        Metrics: Enumerated buffer names/constructors. Confidence: Unknown buffers fail closed.
    Constraints: No learned values are generated here. Reconsider When: Upstream adds another cache type.
    References: restore_native_checkpoint. ADR: None.
    """
    import torch
    from vlash.layers.rope import RotaryEmbedding
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
    from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings
    records=[]
    for name,module in policy.named_modules():
        pending={key:value for key,value in module._buffers.items() if value is not None and value.is_meta}
        if not pending:continue
        if isinstance(module,RotaryEmbedding) and set(pending)=={'cos_sin_cache'}:
            text=policy.config.vlm_config.text_config
            fresh=RotaryEmbedding(module.head_size,module.head_size,pending['cos_sin_cache'].shape[0],text.rope_theta)
            module._buffers['cos_sin_cache']=fresh.cos_sin_cache
            method='original vlash.layers.rope.RotaryEmbedding constructor'
        elif isinstance(module,GemmaRotaryEmbedding) and set(pending)=={'inv_freq'}:
            fresh=GemmaRotaryEmbedding(module.config,device='cpu')
            module._buffers['inv_freq']=fresh.inv_freq
            module.original_inv_freq=module.inv_freq
            method='original patched Transformers GemmaRotaryEmbedding constructor'
        elif isinstance(module,SiglipVisionEmbeddings) and set(pending)=={'position_ids'}:
            module._buffers['position_ids']=torch.arange(module.num_positions).expand((1,-1))
            method='original SiglipVisionEmbeddings positional arange expression'
        else:raise RuntimeError(f'unsupported nonpersistent meta cache: {name} {sorted(pending)}')
        for key,old in pending.items():
            value=module._buffers[key]
            if key not in module._non_persistent_buffers_set or value.is_meta or value.shape!=old.shape or value.dtype!=old.dtype:
                raise RuntimeError('deterministic cache materialization changed its schema')
            records.append({'buffer':f'{name}.{key}','shape':list(value.shape),'dtype':str(value.dtype),'method':method,'learned':False})
    if any(buffer.is_meta for buffer in policy.buffers()):raise RuntimeError('unmaterialized meta buffer remains')
    return records


def restore_native_checkpoint(policy,checkpoint,device):
    """Load every native tensor, preserve original ties and compare exact postcast bytes.

    Args: policy: meta native model; checkpoint: fixed safetensors; device: target device.
    Returns: Full restoration proof. Raises: Digest, schema, alias, value or finite-stat errors.
    Design: Context: No local training or random fallback. Concern: Upstream loader tolerates missing keys.
        Decision: SHA verification, strict complete load, alias rebinding and all-value comparison.
        Alternatives: Author permissive from_pretrained path.
        Quality Attributes: Inspectability, memory efficiency. Trade-offs: Untimed full tensor IO.
    Evidence: Internal: Pinned LFS SHA,824stored values,255original aliases.
        External: Original author architecture and safetensors metadata.
        Findings:74extra values are normalization Parameters rather than new learned layers.
        Metrics: All keys/elements verified, no meta leftovers. Confidence: Exact bytes after native dtype cast.
    Constraints: Must run outside concurrent real-time experiments. Reconsider When: Checkpoint changes.
    References: restoration.json. ADR: None.
    """
    import torch
    from safetensors.torch import load_file
    checkpoint=Path(checkpoint)
    if checkpoint.stat().st_size!=WEIGHT_BYTES:raise ValueError('VLASH checkpoint byte count mismatch')
    digest=hashlib.sha256()
    with checkpoint.open('rb') as source:
        while block:=source.read(8*1024*1024):digest.update(block)
    if digest.hexdigest()!=WEIGHT_SHA:raise ValueError('VLASH checkpoint SHA mismatch')
    header=checkpoint_header(checkpoint);stored=load_file(str(checkpoint),device='cpu')
    expected=policy.state_dict(keep_vars=True);complete,aliases=expand_aliases(header,expected,stored)
    casts={key:str(expected[key].dtype) for key,value in complete.items() if value.dtype!=expected[key].dtype}
    mapped={key:value.to(dtype=expected[key].dtype) for key,value in complete.items()}
    incompatible=policy.load_state_dict(mapped,strict=True,assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:raise RuntimeError('strict VLASH restore failed')
    bind_aliases(policy,aliases)
    deterministic_caches=materialize_nonpersistent_buffers(policy)
    del expected
    policy.to(device)
    live=policy.state_dict(keep_vars=True);checked_elements=0
    for key,reference in mapped.items():
        value=live[key]
        if value.is_meta:raise RuntimeError(f'uninitialized VLASH tensor: {key}')
        actual=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        expected_bytes=reference.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(actual,expected_bytes):raise RuntimeError(f'VLASH tensor byte mismatch: {key}')
        checked_elements+=reference.numel()
    for alias,target in aliases.items():
        if id(live[alias])!=id(live[target]):raise RuntimeError('original VLASH tie was not preserved')
    used_stats=[key for key in stored if (key.startswith('normalize_inputs.buffer_observation_state.') or
                key.startswith('unnormalize_outputs.buffer_action.') or key.startswith('normalize_targets.buffer_action.'))]
    for key in used_stats:
        if not torch.isfinite(stored[key]).all():raise RuntimeError(f'nonfinite used normalization statistic: {key}')
    stored_elements=sum(value.numel() for value in stored.values())
    proof={'repository':REPO,'revision':REVISION,'weight_sha256':WEIGHT_SHA,'weight_bytes':WEIGHT_BYTES,
           'stored_tensor_count':len(stored),'stored_element_count':stored_elements,
           'runtime_state_tensor_count':len(live),'checked_tensor_count':len(mapped),
           'checked_state_elements_including_aliases':checked_elements,'shared_alias_count':len(aliases),
           'normalization_element_count':sum(value.numel() for key,value in stored.items() if key.startswith(('normalize_inputs.','normalize_targets.','unnormalize_outputs.'))),
           'unique_parameter_count':sum(parameter.numel() for parameter in policy.parameters()),
           'parameter_bytes':sum(parameter.numel()*parameter.element_size() for parameter in policy.parameters()),
           'native_dtype_casts':casts,'missing_keys':[],'unexpected_keys':[],
           'all_tensor_values_byte_equal':True,'all_aliases_preserved':True,'meta_tensors_remaining':0,
           'nonpersistent_deterministic_buffers':deterministic_caches,
           'source_revision':SOURCE_REVISION,'local_training':False,'strict':True}
    del mapped,complete,stored,live;gc.collect()
    return proof


class FrozenVLASH:
    """Expose one verified author VLASH model to the shared raw-sensor worker API.

    Args: modeldir: pinned weights; device: CUDA/CPU device.
    Returns: Runtime object. Raises: Integrity/backend/device/restoration errors.
    Design: Context: Dedicated asynchronous pretrained policy. Concern: Baseline environment must stay intact.
        Decision: New overlay and exact original policy with strict native restoration.
        Alternatives: Alter ordinary pi05 runtime. Quality Attributes: Isolation. Trade-offs: Startup audit IO.
    Evidence: Internal: Source and model manifests. External: Author LIBERO release.
        Findings: Native H50/flow10; state goes into discretized language prompt.
        Metrics: Latency, frozen-state SHA, VRAM. Confidence: Per-run audits required.
    Constraints: Single request at a time, no optimizer or simulator truth.
    Reconsider When: Concurrent inference is introduced.
    References: predict; audit. ADR: None.
    """
    def __init__(self,modeldir,device='cuda'):
        """Construct without random tensor allocation and account for every restored value.

        Args: modeldir: fixed directory; device: target torch device. Returns: None.
        Raises: Source/config/CUDA/restoration errors.
        Design: Context:16GB GPU and shared32GB host. Concern: Random CPU construction doubles RAM.
            Decision: Original constructor on meta followed by strict assigned weights and original ties.
            Alternatives: Allocating full random initialization. Quality Attributes: Resource isolation. Trade-offs: Alias audit.
        Evidence: Internal: Strict all-tensor proof. External: torch meta/assign APIs.
            Findings: No learned parameter remains on meta. Metrics: Restoration counts. Confidence: Exact checks.
        Constraints: Initialization outside timed rollout. Reconsider When: Author constructor changes.
        References: restore_native_checkpoint. ADR: None.
        """
        import torch
        self.torch=torch;self.device=torch.device(device);self.modeldir=Path(modeldir).resolve()
        if hashlib.sha256((self.modeldir/'config.json').read_bytes()).hexdigest()!=CONFIG_SHA:
            raise ValueError('VLASH config digest mismatch')
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        policy_class,config,self.provenance=load_author_components(self.modeldir)
        config.device=str(self.device)
        with torch.device('meta'):self.policy=policy_class(config)
        self.restoration=restore_native_checkpoint(self.policy,self.modeldir/'model.safetensors',self.device)
        (self.modeldir/'restoration.json').write_text(json.dumps(self.restoration,indent=2)+'\n')
        self.policy.eval()
        for parameter in self.policy.parameters():parameter.requires_grad_(False)
        self.weights_before=model_state_digest(self.policy);self.calls=0;self.reset()
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)

    def reset(self,seed=0):
        """Reset native inference queue and RNG without touching any learned state.

        Args: seed: integer. Returns: None. Raises: Seed errors.
        Design: Context: Paired episodes. Concern: Hidden queue/noise carryover.
            Decision: Author reset plus torch/NumPy seed. Alternatives: Shared queues across episodes.
            Quality Attributes: Reproducibility. Trade-offs: Noise advances per request.
        Evidence: Internal: Native sample_actions resets KV cache per call.
            External: Author reset. Findings: No optimizer. Metrics: None. Confidence: API contract.
        Constraints: No active inference. Reconsider When: Multiple requests run concurrently.
        References: predict. ADR: None.
        """
        self.seed=int(seed);self.torch.manual_seed(self.seed);np.random.seed(self.seed);self.policy.reset()

    def predict(self,observation,prompt):
        """Generate native actions from raw RGB/proprioception with one canonical image rotation.

        Args: observation: raw two cameras and measured robot sensors; prompt: task string.
        Returns: Float32 actions[50,7] and timed metadata. Raises: Sensor/output shape or finite errors.
        Design: Context: Shared LIBERO environment. Concern: Extra truth leakage or double rotation.
            Decision: Sensor allowlist, author LeRobot180degree rotation, canonical state8 and native preprocessing.
            Alternatives: Passing simulator dictionaries or replacing author normalization.
            Quality Attributes: Fidelity. Trade-offs: CPU-to-GPU preprocessing included in wall latency.
        Evidence: Internal: Frozen checkpoint schema. External: Author LeRobot0.4.1 LiberoEnv._format_raw_obs.
            Findings: Additional ee/joint/gripper stats are retained but their unused feature keys are omitted.
            Metrics: Native horizon50, flow10, synchronized model/wall time.
            Confidence: API smoke does not establish task success.
        Constraints: No action clipping, physics stepping or ground-truth metrics here.
        Reconsider When: Robot/camera coordinate conventions change.
        References: Native PI05Policy.predict_action_chunk. ADR: None.
        """
        torch=self.torch;start=time.perf_counter();batch={}
        for sensor,key in [('agentview_image','observation.images.image'),('robot0_eye_in_hand_image','observation.images.wrist_image')]:
            image=np.asarray(observation[sensor])
            if image.shape!=(256,256,3) or image.dtype!=np.uint8:raise ValueError('expected raw256RGB uint8 camera')
            rotated=image[::-1,::-1].copy()
            batch[key]=torch.from_numpy(rotated).permute(2,0,1).unsqueeze(0).to(self.device,dtype=torch.float32)/255.
        state=canonical_state({key:observation[key] for key in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')})
        batch['observation.state']=torch.as_tensor(state,device=self.device,dtype=torch.float32).unsqueeze(0)
        batch['task']=[str(prompt)]
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        model_start=time.perf_counter()
        with torch.inference_mode():chunk=self.policy.predict_action_chunk(batch)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        model_ms=(time.perf_counter()-model_start)*1000
        actions=chunk.detach().float().cpu().numpy()[0]
        if actions.shape!=(50,7) or not np.isfinite(actions).all():raise RuntimeError('invalid VLASH native action chunk')
        self.calls+=1
        return actions,{'model':'vlash','repository':REPO,'revision':REVISION,
                        'source_revision':SOURCE_REVISION,'inference_ms':(time.perf_counter()-start)*1000,
                        'model_inference_ms':model_ms,'chunk_length':50,'native_horizon':50,
                        'native_n_action_steps':50,'flow_steps':10,'dtype':self.policy.config.dtype,
                        'compile_model':False,'fuse_qkv':False,'fuse_gate_up':False,'call':self.calls,
                        'peak_cuda_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
                        'action_min':float(actions.min()),'action_max':float(actions.max()),
                        'local_training':False,'raw_sensor_allowlist':['agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}

    def audit(self):
        """Hash every registered state value before/after and report restoration/resource evidence.

        Args: None. Returns: Audit mapping. Raises: Tensor transfer errors.
        Design: Context: Frozen inference. Concern: requires_grad alone misses state changes.
            Decision: Full byte digest outside timed inference. Alternatives: Model name alone.
            Quality Attributes: Auditability. Trade-offs: Untimed multiGB transfer.
        Evidence: Internal: Startup digest and exact restoration. External: Torch state_dict.
            Findings:74normalizer Parameters are included. Metrics: Equal SHA, zero grad count, peak bytes.
            Confidence: Exact persistent-state comparison.
        Constraints: Must run during exclusive resource lease. Reconsider When: Persistent state changes.
        References: model_state_digest. ADR: None.
        """
        after=model_state_digest(self.policy)
        return {'model':'vlash','repository':REPO,'revision':REVISION,'source_revision':SOURCE_REVISION,
                'weights_before_sha256':self.weights_before,'weights_after_sha256':after,
                'weights_unchanged':after==self.weights_before,
                'requires_grad_parameter_count':sum(p.numel() for p in self.policy.parameters() if p.requires_grad),
                'parameter_count':sum(p.numel() for p in self.policy.parameters()),
                'peak_cuda_bytes':self.torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
                'calls':self.calls,'native_horizon':50,'flow_steps':10,'local_training':False,
                'restoration':self.restoration,'provenance':self.provenance}

    def close(self):
        """Release model references and cached CUDA allocations before another policy loads.

        Args: None. Returns: None. Raises: None.
        Design: Context: One16GB device. Concern: Residual models create cross-method interference.
            Decision: Remove references, collect and empty CUDA cache. Alternatives: Concurrent loads.
            Quality Attributes: Isolation. Trade-offs: Reload required.
        Evidence: Internal: Resource lease protocol. External: torch.cuda.empty_cache.
            Findings: Live tensors must be removed first. Metrics: None. Confidence: API semantics.
        Constraints: No calls after close. Reconsider When: MultiGPU deployment is used.
        References: FrozenVLASH. ADR: None.
        """
        self.policy=None;gc.collect()
        if self.device.type=='cuda':self.torch.cuda.empty_cache()
