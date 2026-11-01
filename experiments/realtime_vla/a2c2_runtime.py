# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Portions adapted from k1000dai/a2c2-libero, commit75fa9d202fb74150bdbc54ab182c2773f3ed5600.
# Licensed under Apache License2.0: https://www.apache.org/licenses/LICENSE-2.0
# Original notices and full source are retained by download_a2c2.py in assets/upstream/LICENSE.
"""Run a frozen author SmolVLA and A2C2 residual candidate with explicit per-tick correction.

Args: Owned pinned asset directory, device and raw LIBERO sensor observations.
Returns: Full base action chunks, immutable source context and corrected single actions.
Raises: Integrity/schema errors; no random-weight or partial-load fallback.
Design: Context: Additional training is excluded from the realtime study.
    Concern: Head features and normalizers must match the author base, not a different expert.
    Decision: Restore exact author base and matching September head architecture in existing env.
    Alternatives: Current main head has incompatible learned parameters.
    Quality Attributes: Reproducibility, transparency. Trade-offs: Pair SHA linkage is inferred.
Evidence: Internal: Fixed HF weights and source. External: arxiv.org/abs/2509.23224.
    Findings: Frozen correction is distinct from base finetuning.
    Metrics: All stored tensors restored, inference latency. Confidence: Identity until task tests.
Constraints: No optimizer, object ground truth, package mutation or global monkeypatch.
Reconsider When: Matching author pair is explicitly published.
References: docs/reference/realtime-continuity-methods.md. ADR: None.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from safetensors.torch import load_file

if __package__:
    from .download_a2c2 import BASE_REPO, BASE_REV, HEAD_REPO, HEAD_REV, UPSTREAM_REV, DIRECT_DECODE_UPSTREAM_REV
else:
    from download_a2c2 import BASE_REPO, BASE_REV, HEAD_REPO, HEAD_REV, UPSTREAM_REV, DIRECT_DECODE_UPSTREAM_REV

ResidualTransformerConfig = SimpleNamespace

# Source-derived functions below preserve the pinned author graph and state_dict names.
def sinusoidal_position_embedding_2d(height: int, width: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Apply the checkpoint-compatible author operation without changing its tensor math.

Args: As annotated; tensors follow the author model contract. Returns: Author tensor result.
Raises: Torch shape/device errors or explicit upstream schema errors.
Design: Context: Inference reproduction. Concern: Architecture drift.
    Decision: Preserve fixed author implementation. Alternatives: Reimplement or approximate.
    Quality Attributes: Fidelity. Trade-offs: Fixed legacy behavior.
Evidence: Internal: SHA-pinned source and strict weights. External: Author75fa9d2.
    Findings: No training operation. Metrics: Restoration/parity. Confidence: High identity.
Constraints: Source-derived math. Reconsider When: Model revision changes.
References: docs/reference/realtime-continuity-methods.md. ADR: None."""
    if dim % 2 != 0:
        raise ValueError(f'Embedding dimension {dim} must be even for 2D encoding.')

    def _positional_encoding(length: int, channels: int) -> Tensor:
        """Apply the checkpoint-compatible author operation without changing its tensor math.

Args: As annotated; tensors follow the author model contract. Returns: Author tensor result.
Raises: Torch shape/device errors or explicit upstream schema errors.
Design: Context: Inference reproduction. Concern: Architecture drift.
    Decision: Preserve fixed author implementation. Alternatives: Reimplement or approximate.
    Quality Attributes: Fidelity. Trade-offs: Fixed legacy behavior.
Evidence: Internal: SHA-pinned source and strict weights. External: Author75fa9d2.
    Findings: No training operation. Metrics: Restoration/parity. Confidence: High identity.
Constraints: Source-derived math. Reconsider When: Model revision changes.
References: docs/reference/realtime-continuity-methods.md. ADR: None."""
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2, device=device, dtype=dtype) * (-math.log(10000.0) / channels))
        pe = torch.zeros(length, channels, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    half_dim = dim // 2
    height_encoding = _positional_encoding(height, half_dim)
    width_encoding = _positional_encoding(width, half_dim)
    pos = torch.zeros(height, width, dim, device=device, dtype=dtype)
    pos[:, :, :half_dim] = height_encoding[:, None, :]
    pos[:, :, half_dim:] = width_encoding[None, :, :]
    return pos.view(height * width, dim)

class AuthorResidualTransformer(nn.Module):
    """Port the author's checkpoint-compatible residual architecture without changing math.

Args: config: checkpoint feature and architecture metadata. Returns: Torch module.
Raises: Upstream input/schema errors.
Design: Context: September head differs from latest main. Concern: Partial restoration.
    Decision: Static source port from author commit75fa9d2. Alternatives: Current main head.
    Quality Attributes: Fidelity. Trade-offs: Legacy architecture is pinned.
Evidence: Internal:200 stored tensors. External: k1000dai/a2c2-libero.
    Findings: action_head, not a learned CLS/residual MLP. Metrics: Strict tensor restoration.
    Confidence: High identity, task competence unmeasured.
Constraints: Frozen inference only. Reconsider When: Another checkpoint is selected.
References: download_a2c2.py. ADR: None."""

    def __init__(self, config: ResidualTransformerConfig) -> None:
        """Apply the checkpoint-compatible author operation without changing its tensor math.

Args: As annotated; tensors follow the author model contract. Returns: Author tensor result.
Raises: Torch shape/device errors or explicit upstream schema errors.
Design: Context: Inference reproduction. Concern: Architecture drift.
    Decision: Preserve fixed author implementation. Alternatives: Reimplement or approximate.
    Quality Attributes: Fidelity. Trade-offs: Fixed legacy behavior.
Evidence: Internal: SHA-pinned source and strict weights. External: Author75fa9d2.
    Findings: No training operation. Metrics: Restoration/parity. Confidence: High identity.
Constraints: Source-derived math. Reconsider When: Model revision changes.
References: docs/reference/realtime-continuity-methods.md. ADR: None."""
        super().__init__()
        self.config = config
        self.dim_model = config.dim_model
        self._spatial_pos_cache: dict[tuple[int, int], dict[tuple[str, torch.dtype], Tensor]] = {}
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights, norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation])
            self.image_out_dim = backbone_model.fc.in_features
            self.image_encoder = IntermediateLayerGetter(backbone_model, return_layers={'layer4': 'feature_map'})
            self.image_proj = nn.Conv2d(self.image_out_dim, self.dim_model, kernel_size=1)
            if config.freeze_vision_backbone:
                self.image_encoder.eval()
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
        else:
            self.image_encoder = None
            self.image_proj = None
        if self.config.robot_state_feature is not None:
            self.state_proj = nn.Linear(self.config.robot_state_feature.shape[0], self.dim_model)
        else:
            self.state_proj = None
        if self.config.env_state_feature is not None:
            self.env_state_proj = nn.Linear(self.config.env_state_feature.shape[0], self.dim_model)
        else:
            self.env_state_proj = None
        self.action_proj = nn.Linear(self.config.action_feature.shape[0], self.dim_model)
        self.time_proj = nn.Linear(2, self.dim_model)
        vlm_feature = self.config.input_features.get('vlm_hidden')
        if vlm_feature is not None:
            self.vlm_hidden_proj = nn.Linear(vlm_feature.shape[0], self.dim_model)
        else:
            self.vlm_hidden_proj = None
        self.task_proj = nn.Linear(1, self.dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_model, nhead=self.config.n_heads, dim_feedforward=self.config.dim_feedforward, dropout=self.config.dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.n_encoder_layers)
        self.out_norm = nn.LayerNorm(self.dim_model)
        self.action_head = nn.Linear(self.dim_model, self.config.action_feature.shape[0])

    def _positional_encoding(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Apply the checkpoint-compatible author operation without changing its tensor math.

Args: As annotated; tensors follow the author model contract. Returns: Author tensor result.
Raises: Torch shape/device errors or explicit upstream schema errors.
Design: Context: Inference reproduction. Concern: Architecture drift.
    Decision: Preserve fixed author implementation. Alternatives: Reimplement or approximate.
    Quality Attributes: Fidelity. Trade-offs: Fixed legacy behavior.
Evidence: Internal: SHA-pinned source and strict weights. External: Author75fa9d2.
    Findings: No training operation. Metrics: Restoration/parity. Confidence: High identity.
Constraints: Source-derived math. Reconsider When: Model revision changes.
References: docs/reference/realtime-continuity-methods.md. ADR: None."""
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / self.dim_model))
        pe = torch.zeros(seq_len, self.dim_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, batch: Dict[str, Tensor], base_action: Tensor, base_action_chunk: Tensor | None=None) -> Tensor:
        """Apply the checkpoint-compatible author operation without changing its tensor math.

Args: As annotated; tensors follow the author model contract. Returns: Author tensor result.
Raises: Torch shape/device errors or explicit upstream schema errors.
Design: Context: Inference reproduction. Concern: Architecture drift.
    Decision: Preserve fixed author implementation. Alternatives: Reimplement or approximate.
    Quality Attributes: Fidelity. Trade-offs: Fixed legacy behavior.
Evidence: Internal: SHA-pinned source and strict weights. External: Author75fa9d2.
    Findings: No training operation. Metrics: Restoration/parity. Confidence: High identity.
Constraints: Source-derived math. Reconsider When: Model revision changes.
References: docs/reference/realtime-continuity-methods.md. ADR: None."""
        tokens: list[Tensor] = []
        token_lengths: list[int] = []
        batch_size = base_action.shape[0]
        dtype = base_action.dtype
        device = base_action.device
        base_action_token = self.action_proj(base_action).unsqueeze(1)
        tokens.append(base_action_token)
        token_lengths.append(base_action_token.shape[1])
        chunk_tokens = None
        if base_action_chunk is None:
            base_action_chunk = batch.get('base_action_chunk')
        if base_action_chunk is not None:
            chunk = base_action_chunk.to(device=device, dtype=dtype)
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(1)
            chunk_tokens = self.action_proj(chunk)
        time_feature = batch.get('time_feature')
        if time_feature is None:
            time_feature = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        else:
            time_feature = time_feature.to(device=device, dtype=dtype)
        time_token = self.time_proj(time_feature).unsqueeze(1)
        tokens.append(time_token)
        token_lengths.append(time_token.shape[1])
        if chunk_tokens is not None:
            tokens.append(chunk_tokens)
            token_lengths.append(chunk_tokens.shape[1])
        if self.state_proj is not None and 'observation.state' in batch:
            state_token = self.state_proj(batch['observation.state'].to(dtype)).unsqueeze(1)
            tokens.append(state_token)
            token_lengths.append(state_token.shape[1])
        if self.env_state_proj is not None and 'observation.environment_state' in batch:
            env_token = self.env_state_proj(batch['observation.environment_state'].to(dtype)).unsqueeze(1)
            tokens.append(env_token)
            token_lengths.append(env_token.shape[1])
        image_tokens_concat: Tensor | None = None
        if self.image_encoder is not None:
            image_tokens = []
            for key in self.config.image_features:
                img = batch[key].to(device=device, dtype=torch.float32)
                features_dict = self.image_encoder(img)
                feature_map = features_dict.get('feature_map')
                if feature_map is None:
                    raise KeyError("Expected vision backbone to return key 'feature_map'.")
                feature_map = feature_map.to(device=device, dtype=torch.float32)
                projected = self.image_proj(feature_map).to(device=device, dtype=dtype)
                bsz, _, height, width = projected.shape
                pos_embed = self._get_spatial_pos_embed(height, width, device=device, dtype=dtype)
                pos_embed = pos_embed.unsqueeze(0).expand(bsz, -1, -1)
                projected = projected.flatten(2).transpose(1, 2)
                image_tokens.append(projected + pos_embed)
            if image_tokens:
                image_tokens_concat = torch.cat(image_tokens, dim=1)
        if self.vlm_hidden_proj is not None and 'vlm_hidden' in batch:
            hidden_vec = batch['vlm_hidden'].to(device=device, dtype=dtype)
            vlm_token = self.vlm_hidden_proj(hidden_vec).unsqueeze(1)
            tokens.append(vlm_token)
            token_lengths.append(vlm_token.shape[1])
        tasks = batch.get('task')
        if tasks is not None:
            if isinstance(tasks, str):
                tasks = [tasks] * batch_size
            task_values = []
            for task in tasks:
                digest = hashlib.sha1(task.encode('utf-8')).digest()
                value = int.from_bytes(digest[:4], 'little') / float(4294967295)
                task_values.append(value)
            task_tensor = torch.tensor(task_values, device=device, dtype=dtype).unsqueeze(1)
        else:
            task_tensor = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        task_token = self.task_proj(task_tensor).unsqueeze(1)
        tokens.append(task_token)
        token_lengths.append(task_token.shape[1])
        non_image_token_total = sum(token_lengths)
        if non_image_token_total > 0:
            pos = self._positional_encoding(non_image_token_total, device, dtype=dtype)
            base_tokens = torch.cat(tokens, dim=1) + pos[:, :non_image_token_total]
        else:
            base_tokens = torch.empty(batch_size, 0, self.dim_model, device=device, dtype=dtype)
        if image_tokens_concat is not None:
            x = torch.cat([base_tokens, image_tokens_concat], dim=1)
        else:
            x = base_tokens
        x = self.encoder(x)
        cls_state = self.out_norm(x[:, 0])
        action_norm = self.action_head(cls_state)
        return action_norm

    def _get_spatial_pos_embed(self, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Apply the checkpoint-compatible author operation without changing its tensor math.

Args: As annotated; tensors follow the author model contract. Returns: Author tensor result.
Raises: Torch shape/device errors or explicit upstream schema errors.
Design: Context: Inference reproduction. Concern: Architecture drift.
    Decision: Preserve fixed author implementation. Alternatives: Reimplement or approximate.
    Quality Attributes: Fidelity. Trade-offs: Fixed legacy behavior.
Evidence: Internal: SHA-pinned source and strict weights. External: Author75fa9d2.
    Findings: No training operation. Metrics: Restoration/parity. Confidence: High identity.
Constraints: Source-derived math. Reconsider When: Model revision changes.
References: docs/reference/realtime-continuity-methods.md. ADR: None."""
        cache_key = (height, width)
        per_shape = self._spatial_pos_cache.setdefault(cache_key, {})
        device_key = (str(device), dtype)
        cached = per_shape.get(device_key)
        if cached is not None:
            return cached
        base_key = ('cpu', torch.float32)
        base = per_shape.get(base_key)
        if base is None:
            base = sinusoidal_position_embedding_2d(height, width, self.dim_model, device=torch.device('cpu'), dtype=torch.float32)
            per_shape[base_key] = base
        converted = base.to(device=device, dtype=dtype)
        per_shape[device_key] = converted
        return converted


def _restore_model(module, stored):
    """Restore all model tensors, rejecting missing nonalias parameters and extra keys.

    Args: module: constructed Torch model; stored: its exact state_dict subset.
    Returns: Exact restoration audit. Raises: ValueError for any unexplained tensor.
    Design: Context: Legacy and current LeRobot have different normalizer wrappers.
        Concern: strict=False alone could leave random parameters unnoticed.
        Decision: Verify every stored value and permit only tied aliases for missing keys.
        Alternatives: Shape-only checks. Quality Attributes: Integrity. Trade-offs: CPU IO.
    Evidence: Internal: Pinned safetensors. External: Torch state_dict.
        Findings: Aliases may be omitted by serialization. Metrics: Tensor/element count.
        Confidence: High value identity.
    Constraints: No partial fallback. Reconsider When: Quantization is used.
    References: FrozenSmolVLA restoration pattern. ADR: None.
    """
    result = module.load_state_dict(stored, strict=False, assign=True)
    if result.unexpected_keys:
        raise ValueError(f'unrecognized checkpoint model keys: {result.unexpected_keys}')
    state = module.state_dict()
    pointers = {state[k].data_ptr() for k in stored}
    parameters = dict(module.named_parameters(remove_duplicate=False))
    for key in result.missing_keys:
        if key in parameters and state[key].data_ptr() not in pointers:
            raise ValueError(f'model parameter missing from checkpoint: {key}')
    for key, expected in stored.items():
        actual = state[key].detach().cpu()
        if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise ValueError(f'checkpoint restoration mismatch: {key}')
    return {'all_stored_tensors_match':True, 'tensors':len(stored),
            'stored_elements':sum(x.numel() for x in stored.values()),
            'missing_keys_are_tied_aliases':list(result.missing_keys)}


class LegacyMeanStd:
    """Apply the checkpoint's legacy mean/std buffers without newer processor defaults.

    Args: stored: checkpoint mapping; device: compute device.
    Returns: Frozen normalization helper. Raises: ValueError for unexpected stored tensors.
    Design: Context: Author checkpoint embeds stats rather than separate processors.
        Concern: State, images, VLM features and actions have distinct learned statistics.
        Decision: Retain every exact stored normalization tensor and use legacy equations.
        Alternatives: Recompute statistics. Quality Attributes: Fidelity. Trade-offs: Legacy names.
    Evidence: Internal: Checkpoint buffers. External: Author policies/normalize.py.
        Findings: Normalization uses std+1e-8; inverse uses std directly.
        Metrics: Tensor count. Confidence: High identity.
    Constraints: No statistics fitting. Reconsider When: Different normalization mode is used.
    References: download_a2c2.py. ADR: None.
    """
    def __init__(self, stored, device):
        """Retain validated finite statistics on the selected device.

        Args: stored/device: class contract. Returns: None. Raises: ValueError for invalid stats.
        Design: Context: Published legacy weights. Concern: Infinite defaults.
            Decision: Require finite mean/std keys. Alternatives: Silent identity.
            Quality Attributes: Reliability. Trade-offs: Reject incompatible files.
        Evidence: Internal: Fixed stats. External: Author normalizer.
            Findings: Negative std invalid. Metrics: Counts. Confidence: High.
        Constraints: Frozen. Reconsider When: Mode changes. References: Class. ADR: None.
        """
        self.tensors = {}
        for key, value in stored.items():
            if not key.startswith(('normalize_inputs.', 'normalize_targets.', 'unnormalize_outputs.')):
                raise ValueError(f'unexpected checkpoint tensor: {key}')
            if not key.endswith(('.mean', '.std')) or not torch.isfinite(value).all():
                raise ValueError(f'unsupported or invalid normalization tensor: {key}')
            if key.endswith('.std') and (value < 0).any():
                raise ValueError(f'negative normalization std: {key}')
            self.tensors[key] = value.detach().to(device).clone()

    def transform(self, value, key, prefix, inverse=False):
        """Apply one legacy channel's exact mean/std equation.

        Args: value/key/prefix: channel tensor and checkpoint key; inverse: undo scale.
        Returns: Transformed tensor. Raises: KeyError for missing stats.
        Design: Context: Head actions and chunks share output stats.
            Concern: Mixing model-space and physical-space actions.
            Decision: Explicit prefixes. Alternatives: Generic library defaults.
            Quality Attributes: Fidelity. Trade-offs: Only released MEAN_STD supported.
        Evidence: Internal: Stored tensor identity. External: Author source.
            Findings: Torch broadcasts image/channel stats. Metrics: None. Confidence: High.
        Constraints: No mutation. Reconsider When: Mode changes. References: Class. ADR: None.
        """
        name = f'{prefix}.buffer_{key.replace(".", "_")}'
        mean, std = self.tensors[name+'.mean'], self.tensors[name+'.std']
        return value*std+mean if inverse else (value-mean)/(std+1e-8)


def decode_head_output(head_output, base_normalized, normalizer, decode_mode):
    """Interpret the same trained head tensor as a normalized action or action residual.

    Args: head_output/base_normalized: equal-shaped action tensors; normalizer: saved inverse;
        decode_mode: direct or residual.
    Returns: Physical native action tensor. Raises: ValueError for invalid mode/shape.
    Design: Context: Author75fa uses L1 residual; f7b17 uses direct normalized-action MSE.
        Concern: Identical state_dict shapes do not identify output semantics.
        Decision: Explicit declared mode, same frozen graph and checkpoint normalizers.
        Alternatives: Infer semantics from shape or silently add the base action.
        Quality Attributes: Fidelity, reproducibility. Trade-offs: Checkpoint-code linkage unresolved.
    Evidence: Internal: Recorded author commit diff and fixed MSE checkpoint identity.
        External: github.com/k1000dai/a2c2-libero/commit/f7b17e25871ecdf20cbd5a56bcffad31d3a7289a.
        Findings: The author changed loss and inverse decoding without changing head parameters.
        Metrics: Exact algebra tests. Confidence: Code semantics, not training-link certainty.
    Constraints: No weight/statistics changes or fitting.
    Reconsider When: Authors publish the checkpoint training revision.
    References: docs/reference/realtime-continuity-methods.md. ADR: None.
    """
    if decode_mode not in ('residual', 'direct') or head_output.shape != base_normalized.shape:
        raise ValueError('invalid head decode mode or tensor shape')
    normalized = head_output if decode_mode == 'direct' else base_normalized+head_output
    return normalizer.transform(normalized, 'action', 'unnormalize_outputs', inverse=True)


@dataclass(frozen=True)
class A2C2ChunkContext:
    """Carry one base generation's actions and source VLM feature into later corrections.

    Args: actions: full unnormalized [50,7]; vlm_hidden: source [1,960]; generation: id.
    Returns: Frozen record. Raises: None.
    Design: Context: Head consumes stale base context and latest observations together.
        Concern: A newer chunk's feature must not be attached to an older action.
        Decision: Explicit generation context. Alternatives: Mutable shared last feature.
        Quality Attributes: Temporal integrity. Trade-offs: Small context transfer.
    Evidence: Internal: Author eval stores feature with each chunk entry.
        External: A2C2 paper. Findings: Index refers to original full chunk.
        Metrics: Generation/index. Confidence: High schema.
    Constraints: Preserve source index. Reconsider When: Streaming multiple bases.
    References: FrozenA2C2Runtime.correct. ADR: None.
    """
    actions: np.ndarray
    vlm_hidden: np.ndarray
    generation: int


class FrozenA2C2Runtime:
    """Own the exact author base and a source-compatible frozen residual head.

    Args: modeldir: downloaded pair; device: cpu/cuda; use_amp: base autocast switch.
        load_base/load_head: select process-owned components; at least one is required.
    Returns: Runtime with separate predict and correct operations.
    Raises: ValueError for schema/restoration failure; dependency/device errors.
    Design: Context: Head must run independently of expensive chunk generation.
        Concern: The official prior expert is a different model.
        Decision: Explicit author identities, old stats, source prefix feature and chunk index.
        Alternatives: Fine-tune or substitute a handcrafted controller.
        Quality Attributes: Fidelity, observability. Trade-offs: Candidate pair linkage.
    Evidence: Internal:506 base and200 head tensors. External: Author75fa9d2.
        Findings: Latest main's architecture mismatches the September head.
        Metrics: Restoration, latencies, weight immutability. Confidence: Runtime dependent.
    Constraints: No optimizer, private object state or hidden action buffer.
    Reconsider When: Head/base revisions change.
    References: docs/reference/realtime-continuity-methods.md. ADR: None.
    """
    def __init__(self, modeldir='artifacts/realtime-vla/models/a2c2', device='cuda', use_amp=False,
                 load_base=True, load_head=True, decode_mode='residual'):
        """Build base/head in the existing environment and restore every tensor exactly.

        Args: modeldir/device/use_amp/load_base/load_head plus explicit decode_mode. Returns: None.
        Raises: ValueError for any unexplained checkpoint parameter or statistic.
        Design: Context: Legacy normalizers were moved out of modern policy classes.
            Concern: Partial pretrained loading can hide incompatibility.
            Decision: Split model/stats, strict identity-check both; no base VLM download.
            Alternatives: Install the old full package stack.
            Quality Attributes: Compatibility. Trade-offs: Explicit adapter.
        Evidence: Internal: Pinned weights. External: Author config.
            Findings: Both architectures are available with existing Torch/LeRobot primitives.
            Metrics: Exact tensor counts. Confidence: High identity.
        Constraints: CPU restoration precedes GPU use. Reconsider When: Package ABI changes.
        References: _restore_model. ADR: None.
        """
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor.env_processor import LiberoProcessorStep
        from transformers import AutoTokenizer
        if type(load_base) is not bool or type(load_head) is not bool or not (load_base or load_head):
            raise ValueError('load_base/load_head must select at least one component')
        if decode_mode not in ('residual', 'direct'):
            raise ValueError('decode_mode must be residual or direct')
        self.decode_mode = decode_mode
        self.modeldir = Path(modeldir).resolve()
        self.device = torch.device(device)
        self.use_amp = bool(use_amp)
        self.load_base, self.load_head = load_base, load_head
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        self.policy = self.head = None
        self.base_normalizer = self.head_normalizer = None
        self.base_restoration = self.head_restoration = None
        self.tokenizer = None
        if load_base:
            config = SmolVLAConfig.from_pretrained(self.modeldir/'base')
            config.vlm_model_name = str(self.modeldir/'base'/'vlm')
            config.load_vlm_weights = False
            config.device = 'cpu'
            self.policy = SmolVLAPolicy(config)
            base = load_file(self.modeldir/'base'/'model.safetensors', device='cpu')
            base_model = {k:v for k,v in base.items() if k.startswith('model.')}
            self.base_restoration = _restore_model(self.policy, base_model)
            self.base_normalizer = LegacyMeanStd({k:v for k,v in base.items() if not k.startswith('model.')}, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.modeldir/'base'/'vlm', local_files_only=True)
            self.policy.to(self.device).eval()
            self._original_prefix_forward = self.policy.model.vlm_with_expert.forward
            self.policy.model.vlm_with_expert.forward = self._capture_prefix
            del base, base_model
        if load_head:
            head_config = json.loads((self.modeldir/'head'/'config.json').read_text())
            features = {k:SimpleNamespace(**v) for k,v in head_config['input_features'].items()}
            head_config.update(image_features={k:v for k,v in features.items() if '.images.' in k},
                               input_features=features, robot_state_feature=features['observation.state'],
                               env_state_feature=None,
                               action_feature=SimpleNamespace(**head_config['output_features']['action']),
                               pretrained_backbone_weights=None)
            self.head = AuthorResidualTransformer(SimpleNamespace(**head_config))
            head = load_file(self.modeldir/'head'/'model.safetensors', device='cpu')
            self.head_restoration = _restore_model(self.head, {k[6:]:v for k,v in head.items() if k.startswith('model.')})
            self.head_normalizer = LegacyMeanStd({k:v for k,v in head.items() if not k.startswith('model.')}, self.device)
            self.head.to(self.device).eval()
            del head
        for module in (self.policy, self.head):
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.processor = LiberoProcessorStep()
        self.last_context = None
        self.calls = 0
        self.corrections = 0
        self._captured_hidden = None
        self.metadata = {
            'method':'A2C2 candidate reproduction', 'base_model':BASE_REPO, 'base_revision':BASE_REV,
            'head_model':HEAD_REPO, 'head_revision':HEAD_REV, 'upstream_revision':UPSTREAM_REV,
            'base_parameter_count':sum(x.numel() for x in self.policy.parameters()) if self.policy is not None else 0,
            'head_parameter_count':sum(x.numel() for x in self.head.parameters()) if self.head is not None else 0,
            'load_base':load_base, 'load_head':load_head,
            'chunk_size':50, 'action_dim':7, 'hidden_dim':960,
            'local_training':False, 'base_amp':self.use_amp, 'head_amp':False,
            'decode_mode':decode_mode,
            'head_decode_reference_revision':DIRECT_DECODE_UPSTREAM_REV if decode_mode == 'direct' else UPSTREAM_REV,
            'head_output_semantics':('normalized_action' if decode_mode == 'direct' else 'normalized_action_residual') if load_head else None,
            'base_restoration':self.base_restoration, 'head_restoration':self.head_restoration,
            'pair_guarantee':'Author base ID is explicit in source; exact head/base SHA linkage not stated.',
        }
        self.weights_before = self.weight_digest()

    def _capture_prefix(self, *args, **kwargs):
        """Observe the prefix forward result while returning its original output unchanged.

        Args: Original VLM forward args. Returns: Original VLM result.
        Raises: RuntimeError if the expected source feature is unavailable.
        Design: Context: Current Smol does not expose the author's vlm_hidden attribute.
            Concern: Recomputing a second vision pass changes latency and features.
            Decision: Wrap only this runtime instance and capture prefix first-token960.
            Alternatives: Global patch or separate encoder pass.
            Quality Attributes: Isolation, fidelity. Trade-offs: Small feature clone.
        Evidence: Internal: Author sample_actions L893-912. External: Author75fa9d2.
            Findings: Source is final prefix output[:,0], not a pooled vision feature.
            Metrics: Shape[1,960]. Confidence: High extraction identity.
        Constraints: No model output mutation. Reconsider When: VLM return schema changes.
        References: predict. ADR: None.
        """
        result = self._original_prefix_forward(*args, **kwargs)
        embeds = kwargs.get('inputs_embeds')
        if embeds is not None and embeds[0] is not None and embeds[1] is None:
            outputs = result[0]
            hidden = next((x for x in outputs if x is not None), None) if isinstance(outputs, (list, tuple)) else outputs
            if hidden is None or hidden.shape[-1] != 960:
                raise RuntimeError('source VLM prefix feature is not hidden960')
            self._captured_hidden = hidden[:, 0].contiguous().detach().float()
        return result

    def prepare_observation(self, observation, prompt):
        """Allowlist raw LIBERO sensors and apply the author's one-time camera rotation.

        Args: observation: raw HWC RGB and proprioception; prompt: task string.
        Returns: Unnormalized device tensors and task. Raises: ValueError for invalid sensors.
        Design: Context: Both base/head require the same sensor orientation.
            Concern: Privileged object pose or double rotation would invalidate evaluation.
            Decision: Official LIBERO processor with image/wrist_image names.
            Alternatives: Arbitrary observations dict passthrough.
            Quality Attributes: Input integrity. Trade-offs: Fixed schema.
        Evidence: Internal: Author eval256RGB/state8. External: LeRobot LiberoProcessorStep.
            Findings: State is EEF3+axisangle3+gripper2. Metrics: Shapes. Confidence: High.
        Constraints: No object truth. Reconsider When: Sensor schema changes.
        References: Existing FrozenSmolVLA. ADR: None.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('task prompt must be nonempty text')
        batch = {}
        for source, target in [('agentview_image', 'observation.images.image'),
                               ('robot0_eye_in_hand_image', 'observation.images.wrist_image')]:
            image = np.asarray(observation[source])
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError('expected raw256x256 RGB uint8')
            batch[target] = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)/255.
        values = {}
        for name, size in [('robot0_eef_pos', 3), ('robot0_eef_quat', 4), ('robot0_gripper_qpos', 2)]:
            value = np.asarray(observation[name], dtype=np.float32)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f'invalid proprioception field {name}')
            values[name] = torch.from_numpy(value.copy()).unsqueeze(0).to(self.device)
        batch['observation.robot_state'] = {
            'eef':{'pos':values['robot0_eef_pos'], 'quat':values['robot0_eef_quat']},
            'gripper':{'qpos':values['robot0_gripper_qpos']},
        }
        batch = self.processor.observation(batch)
        batch['task'] = prompt
        return batch

    def predict(self, observation, prompt):
        """Generate all50 author base actions and retain their exact source VLM context.

        Args: observation/prompt: raw current sensors and task.
        Returns: actions[50,7], JSON metadata; last_context stores actions/hidden/generation.
        Raises: ValueError for invalid output. RuntimeError for feature/schema failure.
        Design: Context: Scheduler owns execution and retains the returned source context.
            Concern: A head needs feature provenance at each later action index.
            Decision: Full base generation with legacy tokenizer, stats and source feature.
            Alternatives: select_action hides buffering.
            Quality Attributes: Temporal auditability. Trade-offs: Full50generation.
        Evidence: Internal: Author equations and50step config. External: Source75fa9d2.
            Findings: This is the author's450M base, not previous604M expert.
            Metrics: End-to-end model ms. Confidence: Measured runtime.
        Constraints: Frozen. Reconsider When: Source generation API changes.
        References: A2C2ChunkContext. ADR: None.
        """
        if self.policy is None:
            raise RuntimeError('predict requires load_base=True')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        batch = self.prepare_observation(observation, prompt)
        batch['observation.state'] = self.base_normalizer.transform(batch['observation.state'], 'observation.state', 'normalize_inputs')
        text = prompt if prompt.endswith('\n') else prompt+'\n'
        tokens = self.tokenizer([text], padding=self.policy.config.pad_language_to, padding_side='right',
                                max_length=self.policy.config.tokenizer_max_length, return_tensors='pt')
        batch['observation.language.tokens'] = tokens['input_ids'].to(self.device)
        batch['observation.language.attention_mask'] = tokens['attention_mask'].to(self.device, dtype=torch.bool)
        self._captured_hidden = None
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.use_amp and self.device.type == 'cuda' else nullcontext()
        with torch.inference_mode(), amp:
            normalized = self.policy.predict_action_chunk(batch)
            actions = self.base_normalizer.transform(normalized, 'action', 'unnormalize_outputs', inverse=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        result = actions.detach().float().cpu().numpy()[0]
        if result.shape != (50, 7) or not np.isfinite(result).all() or self._captured_hidden is None:
            raise ValueError('invalid base actions or missing source context')
        self.calls += 1
        hidden = self._captured_hidden.detach().cpu().numpy().copy()
        result = np.array(result, copy=True)
        result.flags.writeable = False
        hidden.flags.writeable = False
        self.last_context = A2C2ChunkContext(result, hidden, self.calls)
        return result.copy(), {'inference_ms':(time.perf_counter()-started)*1000,
                               'chunk_length':50, 'generation':self.calls,
                               'method':'author SmolVLA base for A2C2', 'local_training':False}

    def correct(self, observation, prompt, context, action_index):
        """Use latest sensors to correct one source action using its own chunk and feature.

        Args: observation/prompt: current sensors/task; context: source record; action_index:0..49.
        Returns: corrected float32[7], timing/index metadata.
        Raises: ValueError for invalid context/index; Torch errors for incompatible features.
        Design: Context: Heavy base inference and low-latency head have independent schedules.
            Concern: Current chunk context cannot replace an older executed action's context.
            Decision: Explicit context and original index; author sin/cos phase denominator49.
            Alternatives: Chunk-relative index after slicing corrupts positional input.
            Quality Attributes: Fidelity, temporal integrity. Trade-offs: Head overhead per tick.
        Evidence: Internal: Author eval phase and residual equation.
            External: Author75fa/f7b17. Findings: Declared residual/direct decode precedes inverse stats.
            Metrics: Head request ms. Confidence: Runtime measured.
        Constraints: No object truth or optimizer. Reconsider When: Head input schema changes.
        References: Author evaluation_libero.py. ADR: None.
        """
        if self.head is None:
            raise RuntimeError('correct requires load_head=True')
        if not isinstance(context, A2C2ChunkContext) or type(action_index) is not int or not 0 <= action_index < 50:
            raise ValueError('correction requires source context and original action index0..49')
        if context.actions.shape != (50, 7) or context.vlm_hidden.shape != (1, 960):
            raise ValueError('invalid source context shapes')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        batch = self.prepare_observation(observation, prompt)
        batch['vlm_hidden'] = torch.tensor(context.vlm_hidden, dtype=torch.float32, device=self.device)
        for key in ('observation.images.image', 'observation.images.wrist_image', 'observation.state', 'vlm_hidden'):
            batch[key] = self.head_normalizer.transform(batch[key], key, 'normalize_inputs')
        chunk = torch.tensor(context.actions, dtype=torch.float32, device=self.device).unsqueeze(0)
        base = chunk[:, action_index]
        base_normalized = self.head_normalizer.transform(base, 'action', 'normalize_targets')
        batch['base_action_chunk'] = self.head_normalizer.transform(chunk, 'action', 'normalize_targets')
        phase = 2*math.pi*float(action_index % 50)/49
        batch['time_feature'] = torch.tensor([[math.sin(phase), math.cos(phase)]], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            head_output = self.head(batch, base_normalized, batch['base_action_chunk'])
            action = decode_head_output(head_output, base_normalized, self.head_normalizer, self.decode_mode)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        output = action.detach().float().cpu().numpy()[0]
        if output.shape != (7,) or not np.isfinite(output).all():
            raise ValueError('invalid corrected action')
        self.corrections += 1
        return output, {'correction_ms':(time.perf_counter()-started)*1000, 'source_generation':context.generation,
                        'source_action_index':action_index, 'decode_mode':self.decode_mode,
                        'correction_l2':float(np.linalg.norm(output-context.actions[action_index]))}

    def reset(self, seed=0):
        """Clear generation context and reset random generators between episodes.

        Args: seed: nonnegative integer. Returns: None. Raises: ValueError for invalid seed.
        Design: Context: Paired trials. Concern: Context leaking across episodes.
            Decision: Reset Torch/NumPy and explicit source context.
            Alternatives: Only reset physics. Quality Attributes: Reproducibility. Trade-offs: None.
        Evidence: Internal: No private executable queue. External: Torch seed.
            Findings: Head is stateless. Metrics: Counts. Confidence: High.
        Constraints: Episode boundaries only. Reconsider When: Concurrent episodes.
        References: predict/correct. ADR: None.
        """
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be nonnegative integer')
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.policy is not None:
            self.policy.reset()
        self.calls = self.corrections = 0
        self.last_context = self._captured_hidden = None

    def weight_digest(self):
        """Hash both frozen models and all normalization tensors for immutability checks.

        Args: None. Returns: SHA256. Raises: IO/Torch errors.
        Design: Context: Additional learning prohibited. Concern: Buffer mutation.
            Decision: Include raw names/shapes/dtypes/bytes outside timing.
            Alternatives: Weightfile identity only. Quality Attributes: Integrity. Trade-offs: IO cost.
        Evidence: Internal: All state tensors. External: SHA256.
            Findings: Includes normalization. Metrics: Digest equality. Confidence: High identity.
        Constraints: No concurrent inference during audit. Reconsider When: Quantized state.
        References: audit. ADR: None.
        """
        digest = hashlib.sha256()
        groups = []
        if self.policy is not None:
            groups.extend([('base', self.policy.state_dict()), ('base_stats', self.base_normalizer.tensors)])
        if self.head is not None:
            groups.extend([('head', self.head.state_dict()), ('head_stats', self.head_normalizer.tensors)])
        for group, tensors in groups:
            for key, tensor in tensors.items():
                value = tensor.detach().cpu().contiguous()
                digest.update(f'{group}:{key}:{value.dtype}:{tuple(value.shape)}'.encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def audit(self):
        """Return identities and before/after model immutability without changing weights.

        Args: None. Returns: JSON mapping. Raises: Torch hashing failures.
        Design: Context: Frozen inference reporting. Concern: Unsupported training claims.
            Decision: Exact digest equality. Alternatives: No evidence.
            Quality Attributes: Auditability. Trade-offs: Outside timed control.
        Evidence: Internal: Two models and embedded stats. External: None.
            Findings: No optimizer. Metrics: Digest. Confidence: High identity.
        Constraints: No concurrent model calls. Reconsider When: State schema changes.
        References: weight_digest. ADR: None.
        """
        after = self.weight_digest()
        parameters = [p for module in (self.policy, self.head) if module is not None for p in module.parameters()]
        return {**self.metadata, 'weights_before':self.weights_before, 'weights_after':after,
                'weights_unchanged':after == self.weights_before, 'generation_calls':self.calls,
                'correction_calls':self.corrections,
                'requires_grad_parameter_count':sum(p.requires_grad for p in parameters),
                'parameters_with_grad':sum(p.grad is not None for p in parameters)}


def main():
    """Validate CPU restoration/schema or run explicit dummy-sensor inference on a device.

    Args: CLI modeldir/device/infer/output. Returns: None and validation JSON.
    Raises: Runtime/schema errors. Design: Context: GPU use requires scheduling by caller.
        Concern: Hidden inference during schema validation. Decision: --infer is explicit.
        Alternatives: Constructor starts GPU work. Quality Attributes: Predictability. Trade-offs: None.
    Evidence: Internal: Schema and digest audit. External: Author source.
        Findings: Dummy observations cannot establish task success.
        Metrics: Restoration and finite shapes. Confidence: Smoke only.
    Constraints: No simulator/task-success claim. Reconsider When: CLI contract changes.
    References: FrozenA2C2Runtime. ADR: None.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--modeldir', type=Path, default=Path('artifacts/realtime-vla/models/a2c2'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runtime = FrozenA2C2Runtime(args.modeldir, device=args.device)
    observation = {'agentview_image':np.zeros((256, 256, 3), dtype=np.uint8),
                   'robot0_eye_in_hand_image':np.zeros((256, 256, 3), dtype=np.uint8),
                   'robot0_eef_pos':np.array([0., 0., .5], dtype=np.float32),
                   'robot0_eef_quat':np.array([0., 0., 0., 1.], dtype=np.float32),
                   'robot0_gripper_qpos':np.array([.02, -.02], dtype=np.float32)}
    batch = runtime.prepare_observation(observation, 'pick up the black bowl and place it on the plate')
    report = {'status':'cpu_schema_passed', 'task_success_evaluated':False,
              'sensor_shapes':{k:list(v.shape) for k,v in batch.items() if isinstance(v, torch.Tensor)},
              'metadata':runtime.metadata}
    if args.infer:
        actions, timing = runtime.predict(observation, 'pick up the black bowl and place it on the plate')
        corrected, correction = runtime.correct(observation, 'pick up the black bowl and place it on the plate', runtime.last_context, 0)
        report.update(status='dummy_sensor_inference_passed', actions_shape=list(actions.shape),
                      corrected_shape=list(corrected.shape), generation=timing, correction=correction)
    report['audit'] = runtime.audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
