"""Verify RTC time alignment, physical normalization and input-only Jacobians on CPU.

Args: pytest. Returns: Contract assertions. Raises: AssertionError on regressions.
Design: Context: GPU rollouts cannot isolate scheduling and autograd edge cases.
    Concern: Incorrect prefix phase can be mistaken for learned policy failure.
    Decision: Analytical examples with no pretrained model or GPU allocation.
    Alternatives: Replaying only successful task videos misses boundary conditions.
    Quality Attributes: Correctness. Trade-offs: These tests do not measure task competence.
Evidence: Internal: New continuity runtime. External: RTC endpoint VJP and LeRobot normalizers.
    Findings: Camera lag, whole-chunk expiration and nonzero normalized holds matter.
    Metrics: Exact arrays and numerical Jacobians. Confidence: High for these contracts.
Constraints: No simulator/model downloads or CUDA. Reconsider When: Action/time schema changes.
References: experiments/realtime_vla/continuity_runtime.py. ADR: None.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType,NormalizationMode,PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.processor.normalize_processor import NormalizerProcessorStep,UnnormalizerProcessorStep

from experiments.realtime_vla.continuity_runtime import (
    FrozenContinuityPolicy,FullVJPRTCProcessor,action_array,build_aligned_prefix,drop_elapsed_prefix,
)


def test_capture_tick_prefix_contains_actual_holds_then_planned_future():
    """Keep camera lag and already-executed holds in the exact observed time frame.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Request at12 uses image9. Concern: Three actions already happened.
        Decision: Prefix starts9 and includes actual hold at10.
        Alternatives: Starting at12 rejected. Quality Attributes: Temporal correctness.
        Trade-offs: Finite known future only.
    Evidence: Internal: Distinct action values at each tick. External: RTC time origin.
        Findings: Past and future must not overlap. Metrics: Array equality. Confidence: High.
    Constraints: Physical7D. Reconsider When: Fractional control times are used.
    References: build_aligned_prefix. ADR: None.
    """
    history={9:np.array([.2,0,0,0,0,0,1]),10:np.array([0,0,0,0,0,0,1]),11:np.array([.3,0,0,0,0,0,-1])}
    future=np.arange(35,dtype=np.float32).reshape(5,7)
    prefix=build_aligned_prefix(9,12,history,future,horizon=50)
    np.testing.assert_array_equal(prefix['actions'][:3],np.stack(list(history.values())).astype(np.float32))
    np.testing.assert_array_equal(prefix['actions'][3:],future)
    assert (prefix['origin_tick'],prefix['valid_steps'],prefix['history_steps'],prefix['future_steps'])==(9,8,3,5)


def test_unknown_history_is_rejected_and_unknown_future_is_not_padded():
    """Fail on missing actual history and retain a short known prefix without invented zeros.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Queue may be empty. Concern: Padding falsely conditions future hold.
        Decision: Require executed history but allow known future length0.
        Alternatives: Fake future commands rejected. Quality Attributes: Integrity.
        Trade-offs: Caller may have less conditioning coverage.
    Evidence: Internal: Gap at tick3. External: None. Findings: Valid length is explicit.
        Metrics: Error and prefix shape. Confidence: High.
    Constraints: Missing history cannot silently disappear.
    Reconsider When: Sensor observations have no discrete origin.
    References: build_aligned_prefix. ADR: None.
    """
    with pytest.raises(ValueError,match='missing executed history'):
        build_aligned_prefix(2,4,{2:np.zeros(7)},np.empty((0,7)))
    prefix=build_aligned_prefix(2,3,{2:np.zeros(7)},np.empty((0,7)))
    assert prefix['actions'].shape==(1,7) and prefix['future_steps']==0


def test_delay_drop_uses_native_horizon_before_any_execution_prefix_limit():
    """Keep38 actions of H50 after12 elapsed ticks rather than expiring an E10 slice.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Generation can exceed ten control periods.
        Concern: Early E10 truncation discards useful generated future.
        Decision: Remove delay from native paired chunks first.
        Alternatives: FIFO retiming rejected. Quality Attributes: Correctness.
        Trade-offs:38 rather than50 future actions survive.
    Evidence: Internal: Indexed physical/normalized arrays. External: RTC queue semantics.
        Findings: Both spaces remain paired. Metrics:38 actions. Confidence: High.
    Constraints: Actual arrival tick. Reconsider When: Native horizon changes.
    References: drop_elapsed_prefix. ADR: None.
    """
    physical=np.arange(350,dtype=np.float32).reshape(50,7);normalized=physical/100-2
    suffix=drop_elapsed_prefix({'actions':physical,'normalized':normalized},10,22)
    assert suffix['actions'].shape==(38,7) and suffix['dropped_steps']==12 and suffix['origin_tick']==22
    np.testing.assert_array_equal(suffix['actions'],physical[12:])
    np.testing.assert_array_equal(suffix['normalized'],normalized[12:])


def test_wholly_expired_chunk_has_no_retimed_action():
    """Retain explicit expiration when arrival is later than the whole H50 prediction.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Inference can stall. Concern: Reusing stale motion hides missed timing.
        Decision: Empty suffix. Alternatives: Renaming stale actions as fresh rejected.
        Quality Attributes: Auditability. Trade-offs: Scheduler must issue a hold.
    Evidence: Internal:55tick delay. External: None. Findings:50actions expire.
        Metrics: Empty shapes. Confidence: High.
    Constraints: Paired normalized/physical shapes.
    Reconsider When: Absolute targets support valid indefinite reuse.
    References: drop_elapsed_prefix. ADR: None.
    """
    suffix=drop_elapsed_prefix({'actions':np.zeros((50,7)),'normalized':np.ones((50,7))},0,55)
    assert suffix['actions'].shape==suffix['normalized'].shape==(0,7)
    assert suffix['dropped_steps']==50 and suffix['elapsed_steps']==55


def test_physical_hold_uses_checkpoint_action_normalizer_and_roundtrips():
    """Distinguish physical zero motion from normalized zero with nonzero action means.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Executed holds have no generated latent.
        Concern: Zero-filled normalized prefix means a nonzero physical action.
        Decision: Apply official action-only processor and invert with official unnormalizer.
        Alternatives: Handcrafted scaling rejected. Quality Attributes: Compatibility.
        Trade-offs: Numerical tolerance for processor epsilon.
    Evidence: Internal: Seven unequal means/stds. External: LeRobot processor API.
        Findings: Hold normalized coordinates are not zero. Metrics:1e-6 tolerance.
        Confidence: High for same saved-processor mechanism.
    Constraints: Test statistics are synthetic, not task-success data.
    Reconsider When: Other action preprocess transforms are introduced.
    References: FrozenContinuityPolicy.normalize_actions. ADR: None.
    """
    features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))}
    stats={'action':{'mean':np.linspace(.1,.7,7).tolist(),'std':np.linspace(.2,.8,7).tolist()}}
    args={'features':features,'norm_map':{FeatureType.ACTION:NormalizationMode.MEAN_STD},'stats':stats,'device':'cpu'}
    normalizer=NormalizerProcessorStep(**args);unnormalizer=UnnormalizerProcessorStep(**args)
    runtime=FrozenContinuityPolicy.__new__(FrozenContinuityPolicy)
    runtime.device=torch.device('cpu');runtime.action_normalizer=normalizer
    physical=np.array([[0,0,0,0,0,0,1],[.1,.2,.3,0,0,0,-1]],dtype=np.float32)
    normalized=runtime.normalize_actions(physical)
    assert not np.allclose(normalized[0,:6],0)
    restored=unnormalizer({TransitionKey.ACTION:torch.from_numpy(normalized)})[TransitionKey.ACTION].numpy()
    np.testing.assert_allclose(restored,physical,atol=1e-6)


def test_complete_denoiser_vjp_differs_from_stock_constant_velocity_path():
    """Verify the endpoint derivative vanishes for f(x)=2x at reverse time0.5.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Endpoint x-.5*2x is identically0.
        Concern: Setting requires_grad after f(x) retains an incorrect identity VJP.
        Decision: Compare labeled stock and complete-VJP processors analytically.
        Alternatives: Treating both implementations as identical rejected.
        Quality Attributes: Method fidelity. Trade-offs: Does not establish task advantage.
    Evidence: Internal: Frozen diagonal linear layer. External: Author jax.vjp construction.
        Findings: Complete velocity stays2; stock changes constrained prefix.
        Metrics: Exact allclose and no parameter grads. Confidence: High.
    Constraints: Model weights frozen. Reconsider When: Upstream VJP ordering changes.
    References: FullVJPRTCProcessor. ADR: None.
    """
    layer=torch.nn.Linear(7,7,bias=False)
    with torch.no_grad():layer.weight.copy_(2*torch.eye(7))
    layer.requires_grad_(False)
    config=RTCConfig(execution_horizon=2)
    x=torch.ones(1,4,7);previous=torch.ones(1,4,7)
    with torch.no_grad():
        full=FullVJPRTCProcessor(config).denoise_step(x,previous,1,.5,layer,execution_horizon=2)
        stock=RTCProcessor(config).denoise_step(x,previous,1,.5,layer,execution_horizon=2)
    torch.testing.assert_close(full,torch.full_like(x,2))
    assert not torch.allclose(full,stock)
    assert layer.weight.grad is None and not layer.weight.requires_grad and not full.requires_grad


def test_vjp_pads_only_unused_model_dimensions_and_respects_known_time():
    """Handle an action7 prefix in an internal action32 flow without extending time guidance.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: VLA pads seven robot actions to32 latent coordinates.
        Concern: Internal shape padding can be confused with fake future actions.
        Decision: Prefix2steps only; later timesteps keep base velocity unchanged.
        Alternatives: Padding known time weights to50 rejected.
        Quality Attributes: Schema fidelity. Trade-offs: Extra internal latent coordinates exist.
    Evidence: Internal: Identity denoiser and short7D prefix. External: LeRobot padded latent.
        Findings: Unknown later time has zero guidance. Metrics: Exact suffix velocity.
        Confidence: High for padding/weight mechanics.
    Constraints: No external seven-to32 action transform.
    Reconsider When: Model uses learned action projection before RTC.
    References: FullVJPRTCProcessor.denoise_step. ADR: None.
    """
    x=torch.ones(1,4,32);previous=torch.full((1,2,7),.3)
    with torch.no_grad():
        result=FullVJPRTCProcessor(RTCConfig(execution_horizon=4)).denoise_step(x,previous,1,.5,lambda v:v,execution_horizon=4)
    torch.testing.assert_close(result[:,2:],x[:,2:])
    assert result.shape==x.shape and not result.requires_grad


def test_inference_mode_rejected_before_silent_missing_latent_derivative():
    """Fail explicitly when an outer inference-mode scope would disable RTC action gradients.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Original frozen predictors use inference_mode.
        Concern: enable_grad cannot override that mode.
        Decision: Reject the incompatible context with actionable error.
        Alternatives: Silent identity-only correction rejected.
        Quality Attributes: Reliability. Trade-offs: Caller must use no_grad.
    Evidence: Internal: CPU inference_mode scope. External: PyTorch autograd semantics.
        Findings: No model required to reveal the incompatibility.
        Metrics: Error. Confidence: High.
    Constraints: Only guided calls need latent gradients.
    Reconsider When: Forward-mode/functional VJP replaces autograd.
    References: FullVJPRTCProcessor.denoise_step. ADR: None.
    """
    with torch.inference_mode(),pytest.raises(RuntimeError,match='no_grad'):
        FullVJPRTCProcessor(RTCConfig()).denoise_step(torch.ones(1,4,7),torch.ones(1,4,7),1,.5,lambda x:x)


@pytest.mark.parametrize('bad',[np.zeros((3,8)),np.full((3,7),np.nan),np.zeros(7)])
def test_bad_actions_rejected(bad):
    """Reject unsafe shape/finite values before prefix construction or actuator access.

    Args: bad: malformed chunk. Returns: None. Raises: AssertionError.
    Design: Context: Allspaces share strict action7 schema.
        Concern: Broadcasting hides mistakes. Decision: Explicit validation.
        Alternatives: Clipping does not fix NaNs. Quality Attributes: Integrity.
        Trade-offs: Caller sees an error rather than a guessed command.
    Evidence: Internal: Three invalid cases. External: None.
        Findings: Validation occurs before inference. Metrics: Errors. Confidence: High.
    Constraints: Finite[T,7]. Reconsider When: Action schema changes.
    References: action_array. ADR: None.
    """
    with pytest.raises(ValueError):action_array(bad)
