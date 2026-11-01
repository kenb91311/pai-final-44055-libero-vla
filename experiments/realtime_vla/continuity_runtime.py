# RTC guidance equations and processor structure adapted from LeRobot.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Modification: explicitly differentiate the denoiser input before velocity evaluation.
"""Expose frozen flow policies and explicitly timed RTC chunk continuation.

Args: Raw LIBERO sensors, a fixed policy family, and an observed-tick action prefix.
Returns: Paired physical and normalized H50 action arrays plus timing metadata.
Raises: ValueError for ambiguous time/action schemas; runtime errors for model failures.
Design:
    Context: New chunks must connect to actions executed while inference is running.
    Concern: Physical actions are not normalized model latents; FIFO retiming hides delay.
    Decision: Keep both spaces, retain native H50, and align actions to the sensor tick.
    Alternatives: Averaging chunks is not RTC; truncating to E10 can expire every action.
    Quality Attributes: Fidelity, auditability. Trade-offs: Full VJP adds compute/memory.
Evidence:
    Internal: CPU alignment, normalization and analytical VJP tests.
    External: RTC arXiv2506.07339 and pinned LeRobot RTCProcessor implementation.
    Findings: Stock LeRobot and complete denoiser VJP differ and are named separately.
    Metrics: Full inference time, model time, prefix length and peak CUDA allocation.
    Confidence: Unit contracts; GPU/task evidence must be recorded by the evaluator.
Constraints: No optimizer, training, installed-source edits or privileged object inputs.
Reconsider When: Action convention, checkpoint processors or LeRobot RTC API changes.
References: docs/reference/realtime-continuity-methods.md. ADR: None.
"""
from contextlib import nullcontext
import gc
import time

import numpy as np
import torch
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def action_array(value, *, allow_empty=True):
    """Validate finite unbatched seven-coordinate actions without clipping them.

    Args: value: array-like[T,7]; allow_empty: permit a known zero-length prefix.
    Returns: Owned float32 array. Raises: ValueError for shape/nonfinite values.
    Design: Context: OSC inputs. Concern: Silent padding invents future commands.
        Decision: Exact shape/finite checks. Alternatives: Broadcasting rejected.
        Quality Attributes: Integrity. Trade-offs: One small CPU copy.
    Evidence: Internal: Boundary tests. External: LIBERO action7 contract.
        Findings: Empty prefixes must retain their second dimension.
        Metrics: Shape/finite assertions. Confidence: High.
    Constraints: No units conversion. Reconsider When: Another robot action space is used.
    References: build_aligned_prefix. ADR: None.
    """
    result=np.asarray(value,dtype=np.float32)
    if result.ndim!=2 or result.shape[1]!=7 or not np.isfinite(result).all():
        raise ValueError('expected finite unbatched actions[T,7]')
    if not allow_empty and len(result)==0:raise ValueError('empty action chunk')
    return result.copy()


def build_aligned_prefix(observation_tick,request_tick,history_actions,future_actions,*,horizon=50):
    """Join actual past actions to the still-planned future from the image's tick.

    Args: observation_tick/request_tick: nonnegative control indices; history_actions:
        mapping[tick] to the actual physical action, including holds; future_actions:
        array[T,7] whose first action is scheduled at request_tick; horizon: maximum H.
    Returns: Mapping with actions, origin_tick, valid_steps, history_steps and future_steps.
    Raises: ValueError for missing past actions, reversed ticks or invalid arrays.
    Design: Context: Asynchronous cameras return older snapshots.
        Concern: A prefix starting at request time is phase-shifted from image conditioning.
        Decision: Include every executed action from capture to request, then known future.
        Alternatives: Fabricated zeros or dropping sensor lag hide uncertainty.
        Quality Attributes: Temporal correctness. Trade-offs: Caller retains short action history.
    Evidence: Internal: Tests cover camera lag, holds and incomplete history.
        External: RTC old-prefix time origin. Findings: Unknown future is never padded.
        Metrics: Valid known prefix length. Confidence: High for timestamp contract.
    Constraints: History records executed values, not original unclipped proposals.
    Reconsider When: Control periods vary enough to require continuous-time interpolation.
    References: drop_elapsed_prefix. ADR: None.
    """
    if any(type(v) is not int or v<0 for v in (observation_tick,request_tick)):
        raise ValueError('ticks must be nonnegative integers')
    if request_tick<observation_tick:raise ValueError('observation cannot be from the future')
    if type(horizon) is not int or horizon<1:raise ValueError('horizon must be positive')
    past=[]
    for tick in range(observation_tick,min(request_tick,observation_tick+horizon)):
        if tick not in history_actions:raise ValueError(f'missing executed history at tick {tick}')
        value=np.asarray(history_actions[tick],dtype=np.float32)
        if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('invalid executed action')
        past.append(value.copy())
    future=action_array(future_actions)
    history=np.stack(past) if past else np.empty((0,7),dtype=np.float32)
    remaining=max(0,horizon-len(history));future=future[:remaining]
    actions=np.concatenate((history,future),axis=0)
    return {'actions':actions,'origin_tick':observation_tick,'valid_steps':len(actions),
            'history_steps':len(history),'future_steps':len(future)}


def drop_elapsed_prefix(chunk,observation_tick,arrival_tick):
    """Drop already elapsed indices identically in physical and normalized chunk space.

    Args: chunk: mapping containing actions/normalized arrays with the same H;
        observation_tick/arrival_tick: chunk origin and control index before next action.
    Returns: New mapping with suffix arrays, dropped_steps, elapsed_steps and origin_tick.
    Raises: ValueError for inconsistent arrays or backwards time.
    Design: Context: Physics progresses during generation.
        Concern: Retiming late actions or truncating E before dropping changes the policy.
        Decision: Keep native H until arrival, then remove elapsed control indices.
        Alternatives: FIFO time shifting is a different baseline.
        Quality Attributes: Reproducibility. Trade-offs: A wholly expired result is empty.
    Evidence: Internal: Delay > E and delay >= H tests. External: RTC action queue semantics.
        Findings: Both model and actuator spaces must lose exactly the same prefix.
        Metrics: Actual elapsed and expired counts. Confidence: High.
    Constraints: No invented replacement when every action expired.
    Reconsider When: Policy predicts absolute-time splines instead of sampled actions.
    References: build_aligned_prefix. ADR: None.
    """
    if any(type(v) is not int or v<0 for v in (observation_tick,arrival_tick)) or arrival_tick<observation_tick:
        raise ValueError('invalid chunk origin/arrival ticks')
    actions=action_array(chunk['actions']);normalized=action_array(chunk['normalized'])
    if actions.shape!=normalized.shape:raise ValueError('physical and normalized chunk shapes differ')
    elapsed=arrival_tick-observation_tick;dropped=min(elapsed,len(actions))
    return {**chunk,'actions':actions[dropped:],'normalized':normalized[dropped:],
            'dropped_steps':dropped,'elapsed_steps':elapsed,'origin_tick':arrival_tick}


class FullVJPRTCProcessor(RTCProcessor):
    """Evaluate the RTC denoiser Jacobian before applying prefix guidance.

    Args: RTCConfig as in the upstream parent. Returns: Guided processor instance.
    Raises: Upstream config errors; invalid prefix shapes fail explicitly.
    Design: Context: Pinned LeRobot sets requires_grad after computing its velocity.
        Concern: That path treats velocity as constant during the VJP.
        Decision: Mark latent differentiable before denoising; keep all weights frozen.
        Alternatives: Stock path remains a separately named experimental condition.
        Quality Attributes: Method fidelity. Trade-offs: Backpropagation costs memory/time.
    Evidence: Internal: Analytical linear-denoiser test. External: Author JAX jax.vjp denoiser.
        Findings: In reverse time, the endpoint is x-time*v(x).
        Metrics: Input derivative only. Confidence: High for the implemented equation.
    Constraints: Inference guidance is not gradient-based weight training.
    Reconsider When: Upstream RTC changes its differentiation order.
    References: https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
    ADR: None.
    """
    def denoise_step(self,x_t,prev_chunk_left_over,inference_delay,time,
                     original_denoise_step_partial,execution_horizon=None):
        """Compute prefix-weighted complete VJP using the upstream reverse-time convention.

        Args: Same contract as RTCProcessor.denoise_step, including padded model action space.
        Returns: Detached velocity tensor with the original latent shape.
        Raises: ValueError for prefix dimensions; RuntimeError inside inference_mode.
        Design: Context: Full VJP comparison. Concern: Stale graphs or fake future prefix.
            Decision: Detached leaf, fresh local graph and zero weights after known prefix.
            Alternatives: Differentiating the loss through weights is unnecessary.
            Quality Attributes: Bounded graph lifetime. Trade-offs: One backward per flow step.
        Evidence: Internal: Frozen-linear parameter and no-grad tests.
            External: RTC paper guidance and LeRobot scaling formula.
            Findings: Prefix padding only fills inaccessible model dimensions, not known time.
            Metrics: VJP equality. Confidence: High for analytical cases.
        Constraints: Parent processor configuration is kept; no installed code patch.
        Reconsider When: Attention schedule or guidance definition changes.
        References: RTCProcessor.denoise_step. ADR: None.
        """
        if prev_chunk_left_over is None:return original_denoise_step_partial(x_t)
        if torch.is_inference_mode_enabled():raise RuntimeError('RTC VJP requires no_grad, not inference_mode')
        squeezed=x_t.ndim==2
        if x_t.ndim not in (2,3):raise ValueError('expected latent[T,A] or [B,T,A]')
        latent=x_t.detach().clone()
        if squeezed:latent=latent.unsqueeze(0)
        previous=prev_chunk_left_over.detach()
        if previous.ndim==2:previous=previous.unsqueeze(0)
        if previous.ndim!=3 or previous.shape[0]!=latent.shape[0] or previous.shape[1]>latent.shape[1] or previous.shape[2]>latent.shape[2]:
            raise ValueError('previous prefix incompatible with current latent')
        known=previous.shape[1]
        end=min(known,self.rtc_config.execution_horizon if execution_horizon is None else execution_horizon)
        padded=torch.zeros_like(latent)
        padded[:,:known,:previous.shape[2]]=previous.to(device=latent.device,dtype=latent.dtype)
        weights=self.get_prefix_weights(inference_delay,end,latent.shape[1]).to(device=latent.device,dtype=latent.dtype)[None,:,None]
        with torch.enable_grad():
            latent.requires_grad_(True)
            velocity=original_denoise_step_partial(latent)
            endpoint=latent-time*velocity
            error=(padded-endpoint)*weights
            correction=torch.autograd.grad(endpoint,latent,error.detach(),retain_graph=False,create_graph=False)[0]
        # This is LeRobot's reverse-time guidance scaling, evaluated on the latent device.
        tau=torch.as_tensor(1-time,device=latent.device,dtype=torch.float32)
        cap=torch.as_tensor(self.rtc_config.max_guidance_weight,device=latent.device,dtype=torch.float32)
        complement=1-tau
        inverse_r2=(complement.square()+tau.square())/complement.square()
        coefficient=torch.nan_to_num(complement/tau,posinf=float(self.rtc_config.max_guidance_weight))
        guidance=torch.minimum(torch.nan_to_num(coefficient*inverse_r2,posinf=float(self.rtc_config.max_guidance_weight)),cap)
        result=(velocity-guidance*correction).detach()
        self.track(time=time,x1_t=endpoint.detach(),correction=correction.detach(),err=error.detach(),
                   weights=weights,guidance_weight=guidance,inference_delay=inference_delay,execution_horizon=end)
        return result.squeeze(0) if squeezed else result


class FrozenContinuityPolicy:
    """Wrap the existing verified frozen runtimes with explicit normalized chunk access.

    Args: family: smolvla or pi05; modeldir; variant: off/rtc-stock/rtc-vjp;
        device; execution_horizon: known-prefix weighting end; max_guidance_weight.
    Returns: Runtime with predict/reset/audit/close and normalize_actions methods.
    Raises: Configuration, restoration and inference errors.
    Design: Context: Same learned policy must be compared with and without RTC.
        Concern: Existing inference_mode wrappers disable latent differentiation.
        Decision: Reuse restoration/processors and call official model under no_grad.
        Alternatives: Modify original baseline code or checkpoint weights rejected.
        Quality Attributes: Isolation. Trade-offs: Separate new experimental runtime.
    Evidence: Internal: Frozen adapter state audits. External: LeRobot guided RTC API.
        Findings: Both selected flow policies expose normalized cropped H50x7 output.
        Metrics: Method-specific timing and memory. Confidence: GPU smoke required separately.
    Constraints: Single GPU model at a time; no new state modalities.
    Reconsider When: Models use another chunk/action schema.
    References: smolvla_runtime.py; pi05_runtime.py. ADR: None.
    """
    def __init__(self,family,modeldir,*,variant='off',device='cuda',execution_horizon=30,max_guidance_weight=10.):
        """Load one exact family runtime, then attach only the selected RTC processor.

        Args: Class configuration. Returns: None. Raises: ValueError on unsupported choice.
        Design: Context: Preserve verified checkpoint loading.
            Concern: Processor changes must not alter learned tensors.
            Decision: Configure RTC after frozen restoration; install an instance-local processor.
            Alternatives: Global monkeypatching rejected. Quality Attributes: Traceability.
            Trade-offs: Untimed loading/hashing cost is retained.
        Evidence: Internal: Policy has zero trainable parameters.
            External: Official init_rtc_processor updates inner model pointer.
            Findings: Both policy and inner model share the same config.
            Metrics: Restoration audit. Confidence: API-specific.
        Constraints: No trained-RTC mode on ordinary checkpoints.
        Reconsider When: Model initialization no longer exposes RTC API.
        References: FrozenContinuityPolicy. ADR: None.
        """
        if family not in ('smolvla','pi05') or variant not in ('off','rtc-stock','rtc-vjp'):
            raise ValueError('unsupported frozen flow family or RTC variant')
        if type(execution_horizon) is not int or not 0<=execution_horizon<=50:raise ValueError('invalid execution_horizon')
        if family=='smolvla':
            try:from .smolvla_runtime import FrozenSmolVLA
            except ImportError:from smolvla_runtime import FrozenSmolVLA
            self.runtime=FrozenSmolVLA(modeldir,device=device)
        else:
            try:from .pi05_runtime import FrozenPI05
            except ImportError:from pi05_runtime import FrozenPI05
            self.runtime=FrozenPI05(modeldir,device=device)
        self.family=family;self.variant=variant;self.execution_horizon=execution_horizon
        self.policy=self.runtime.policy;self.device=self.runtime.device;self.calls=0
        if any(p.requires_grad for p in self.policy.parameters()):raise RuntimeError('all policy weights must remain frozen')
        self.policy.config.rtc_config=RTCConfig(enabled=variant!='off',mode='guided',
            execution_horizon=execution_horizon,max_guidance_weight=float(max_guidance_weight))
        self.policy.init_rtc_processor()
        if variant=='rtc-vjp':
            processor=FullVJPRTCProcessor(self.policy.config.rtc_config)
            self.policy.rtc_processor=processor;self.policy.model.rtc_processor=processor
        if int(self.policy.config.chunk_size)!=50:raise ValueError('this experiment requires native H50')
        from lerobot.processor.normalize_processor import NormalizerProcessorStep
        normalizers=[step for step in self.runtime.preprocess.steps if isinstance(step,NormalizerProcessorStep)]
        if len(normalizers)!=1:raise ValueError('expected one saved normalizer step')
        self.action_normalizer=normalizers[0]
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)

    def normalize_actions(self,actions):
        """Normalize actual physical actions using only the checkpoint's saved action normalizer.

        Args: actions: finite[T,7], including actual clipped or hold commands.
        Returns: Owned float32 normalized[T,7] array. Raises: Invalid action/stat errors.
        Design: Context: Execution history has no original model latent for holds.
            Concern: Physical zero is generally not zero after normalization.
            Decision: Apply official normalizer directly to an action-only transition.
            Alternatives: Hand-coded mean/std or padded zero latents rejected.
            Quality Attributes: Fidelity. Trade-offs: Small CPU/device copy outside main model.
        Evidence: Internal: Official roundtrip and nonzero-mean test.
            External: NormalizerProcessorStep transition API.
            Findings: No state tokenizer or image processor is needed for action normalization.
            Metrics: Numerical roundtrip tolerance. Confidence: High.
        Constraints: No additional clipping or gripper inversion.
        Reconsider When: Action preprocess includes another transform.
        References: Saved policy_preprocessor.json. ADR: None.
        """
        from lerobot.lerobot_types import TransitionKey
        value=action_array(actions)
        if len(value)==0:return value
        with torch.no_grad():
            tensor=torch.as_tensor(value,device=self.device,dtype=torch.float32)
            normalized=self.action_normalizer({TransitionKey.ACTION:tensor})[TransitionKey.ACTION]
        return normalized.detach().float().cpu().numpy().copy()

    def prepare_observation(self,observation,prompt):
        """Reuse exact frozen-family raw LIBERO sensor preprocessing.

        Args: observation: raw RGB2 and proprioception; prompt: language instruction.
        Returns: Saved-processor batch. Raises: Invalid sensor shapes or finite values.
        Design: Context: PI05 baseline prepares its inputs inline.
            Concern: Reusing predict would invoke inference_mode and lose normalized actions.
            Decision: Reproduce its raw conversion then call the same saved processors.
            Alternatives: Custom state/quaternion transforms rejected.
            Quality Attributes: Compatibility. Trade-offs: Small family-specific glue.
        Evidence: Internal: Same allowlist and env processor as original runtimes.
            External: LiberoProcessorStep rotates images and creates state8.
            Findings: No object pose, success or evaluator metrics are forwarded.
            Metrics: Raw RGB256 and proprio dimensions. Confidence: Shape checks plus paired tests.
        Constraints: Exactly one official image rotation.
        Reconsider When: Original frozen runtime exposes a shared preparation API.
        References: FrozenPI05.predict. ADR: None.
        """
        if self.family=='smolvla':return self.runtime.prepare_observation(observation,prompt)
        raw={}
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
        processed=self.runtime.env_processor.observation(raw);processed['task']=prompt
        return self.runtime.preprocess(processed)

    def predict(self,observation,prompt,*,prev_normalized=None,prev_actions=None,inference_delay=0,execution_horizon=None):
        """Generate native normalized and physical chunks without blocking physical control.

        Args: Sensors/prompt; either normalized old prefix or physical actual/planned prefix;
            inference_delay: estimated capture-to-arrival ticks; execution_horizon: prefix end.
        Returns: Dict actions[50,7], normalized[50,7], metadata with timings and prefix coverage.
        Raises: ValueError for ambiguous/oversized prefix; RuntimeError in inference_mode.
        Design: Context: New independent scheduler aligns the result on actual arrival.
            Concern: Committed prefix must share the observation's origin and action space.
            Decision: Optional saved-normalizer conversion, no_grad outer context, detach outputs.
            Alternatives: Baseline predict hides normalized chunk and forbids RTC gradients.
            Quality Attributes: Inspectability. Trade-offs: Guidance adds latency to be measured.
        Evidence: Internal: CPU adapter tests and pending GPU smoke.
            External: Official predict_action_chunk RTC kwargs.
            Findings: H50 is retained before scheduler delay removal.
            Metrics: Total/model ms, known prefix, estimated delay and peak CUDA bytes.
            Confidence: API contract, not task-success evidence.
        Constraints: No optimizer or model-weight gradient accumulation.
        Reconsider When: More than one request is allowed in flight.
        References: build_aligned_prefix; drop_elapsed_prefix. ADR: None.
        """
        if torch.is_inference_mode_enabled():raise RuntimeError('continuity runtime requires no_grad, not inference_mode')
        if prev_normalized is not None and prev_actions is not None:raise ValueError('provide only one prefix action space')
        if type(inference_delay) is not int or not 0<=inference_delay<=50:raise ValueError('inference_delay must be in [0,50]')
        if execution_horizon is None:execution_horizon=self.execution_horizon
        if type(execution_horizon) is not int or not 0<=execution_horizon<=50:raise ValueError('invalid prefix weighting horizon')
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter();previous=None
        if prev_actions is not None:prev_normalized=self.normalize_actions(prev_actions)
        if prev_normalized is not None:
            array=action_array(prev_normalized)
            if len(array)>50:raise ValueError('prefix longer than native H50')
            if len(array):previous=torch.as_tensor(array,device=self.device,dtype=torch.float32).unsqueeze(0)
        valid_steps=0 if previous is None else previous.shape[1]
        end=min(valid_steps,execution_horizon)
        kwargs={} if self.variant=='off' else {'prev_chunk_left_over':previous,'inference_delay':inference_delay,'execution_horizon':end}
        batch=self.prepare_observation(observation,prompt)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        model_started=time.perf_counter()
        amp=(torch.autocast('cuda',dtype=torch.bfloat16) if self.family=='smolvla' and self.device.type=='cuda' and self.runtime.use_amp else nullcontext())
        with torch.no_grad(),amp:
            normalized=self.policy.predict_action_chunk(batch,**kwargs).detach()
            actions=self.runtime.postprocess(normalized.clone()).detach()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        model_ms=(time.perf_counter()-model_started)*1000
        normal=action_array(normalized.float().cpu().numpy()[0],allow_empty=False)
        physical=action_array(actions.float().cpu().numpy()[0],allow_empty=False)
        if normal.shape!=(50,7) or physical.shape!=(50,7):raise RuntimeError('native normalized/physical chunk must be H50x7')
        self.calls+=1;self.runtime.calls+=1
        metadata={'model':self.family,'variant':self.variant,'inference_ms':(time.perf_counter()-started)*1000,
            'model_inference_ms':model_ms,'chunk_length':50,'native_horizon':50,'prefix_valid_steps':valid_steps,
            'execution_horizon':end,'inference_delay':inference_delay,'prefix_covers_estimated_delay':valid_steps>=inference_delay,
            'rtc_enabled':self.variant!='off','full_denoiser_vjp':self.variant=='rtc-vjp','local_training':False,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
            'normalized_action_space':'checkpoint saved ACTION normalization; cropped7; internal unused dimensions padded by RTC',
            'call':self.calls,'action_min':float(physical.min()),'action_max':float(physical.max())}
        return {'actions':physical,'normalized':normal,'metadata':metadata}

    def reset(self,seed=0):
        """Reset only family inference state and RNG between episodes.

        Args: seed: nonnegative integer. Returns: None. Raises: Family reset errors.
        Design: Context: Paired conditions. Concern: Hidden state carryover.
            Decision: Delegate verified reset. Alternatives: Reuse history rejected.
            Quality Attributes: Reproducibility. Trade-offs: No cross-episode memory.
        Evidence: Internal: Original runtime reset. External: Policy.reset.
            Findings: Weights remain fixed. Metrics: Seed. Confidence: API contract.
        Constraints: No concurrent prediction. Reconsider When: Parallel episodes are added.
        References: FrozenSmolVLA.reset; FrozenPI05.reset. ADR: None.
        """
        self.runtime.reset(seed);self.calls=0

    def audit(self):
        """Include base complete-state identity and RTC-specific frozen-gradient checks.

        Args: None. Returns: Audit mapping. Raises: Family hashing errors.
        Design: Context: Latent differentiation must never become training.
            Concern: Gradients on parameters could reveal accidental weight participation.
            Decision: Check requires_grad and accumulated grads in addition to byte digests.
            Alternatives: Eval mode alone rejected. Quality Attributes: Integrity.
            Trade-offs: Base hashing remains untimed and expensive.
        Evidence: Internal: Full frozen-model audits. External: PyTorch autograd.
            Findings: Only latent leaves are differentiable. Metrics: Parameter grad count.
            Confidence: High for measured instance.
        Constraints: Audit after requests complete. Reconsider When: Intentional adaptation is added.
        References: FullVJPRTCProcessor. ADR: None.
        """
        return {**self.runtime.audit(),'variant':self.variant,'continuity_calls':self.calls,
            'requires_grad_parameter_count':sum(p.numel() for p in self.policy.parameters() if p.requires_grad),
            'parameters_with_grad':sum(p.grad is not None for p in self.policy.parameters()),'local_training':False}

    def close(self):
        """Release this one owned runtime and CUDA cache before the next condition.

        Args: None. Returns: None. Raises: None.
        Design: Context: Single16GB GPU. Concern: A wrapper retains extra policy references.
            Decision: Drop wrapper references then delegate close/collect.
            Alternatives: Concurrent model residency rejected. Quality Attributes: Isolation.
            Trade-offs: Reload for next family.
        Evidence: Internal: Parent schedules GPU ownership. External: CUDA allocation semantics.
            Findings: empty_cache cannot free live tensors. Metrics: None. Confidence: API contract.
        Constraints: No calls after close. Reconsider When: MultiGPU serving is used.
        References: FrozenPI05.close. ADR: None.
        """
        self.policy=None;self.action_normalizer=None
        if hasattr(self.runtime,'close'):self.runtime.close()
        self.runtime=None;gc.collect()
        if self.device.type=='cuda':torch.cuda.empty_cache()
