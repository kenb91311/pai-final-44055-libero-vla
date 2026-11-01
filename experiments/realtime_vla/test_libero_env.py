"""Test canonical sensing, safe relative holds and pre-actuation validation.

Args: pytest discovery. Returns: Assertions. Raises: AssertionError on contract mismatch.
Design:
    Context: Async scheduling can starve or deliver malformed chunks.
    Concern: A fallback must not repeat motion or silently corrupt state features.
    Decision: Unit tests isolate action and sensing semantics without graphics startup.
    Alternatives: Rollout videos cannot establish rejection before actuation.
    Quality Attributes: Reliability. Trade-offs: Full environment tested separately.
Evidence:
    Internal: libero_env.py contract. External: LIBERO relative control conventions.
    Findings: Zero relative pose plus retained gripper is a neutral fallback.
    Metrics: Exact shape/state checks. Confidence: Unit-test scope only.
Constraints: No policy model or asset download.
Reconsider When: Controller representation changes.
References: libero_env.py.
ADR: None.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('libero_env',Path(__file__).with_name('libero_env.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_canonical_state_and_identity_rotation():
    """Produce the shared eight-value state without object truth.

    Args: None. Returns: None. Raises: AssertionError.
    Design:
        Context: Both policies expect one state schema. Concern: Quaternion shape drift.
        Decision: Test identity and90degree rotation. Alternatives: Shape-only tests miss units.
        Quality Attributes: Compatibility. Trade-offs: Floating-point tolerance.
    Evidence:
        Internal: Official LeRobot axis-angle convention. External: None.
        Findings: Identity rotation becomes zero. Metrics: Numerical equality. Confidence: High.
    Constraints: XYZW quaternion.
    Reconsider When: Policy schema changes.
    References: canonical_state.
    ADR: None.
    """
    raw={'robot0_eef_pos':[.1,.2,.3],'robot0_eef_quat':[0,0,0,1],'robot0_gripper_qpos':[.02,-.02]}
    np.testing.assert_allclose(module.canonical_state(raw),[.1,.2,.3,0,0,0,.02,-.02],atol=1e-7)
    raw['robot0_eef_quat']=[0,0,np.sqrt(.5),np.sqrt(.5)]
    np.testing.assert_allclose(module.canonical_state(raw)[3:6],[0,0,np.pi/2],atol=1e-6)


@pytest.mark.parametrize('action',[[0]*6,[0]*8,[0,0,0,0,0,0,float('nan')],[0,0,0,0,0,0,float('inf')]])
def test_invalid_action_never_advances_physics(action):
    """Reject invalid action shapes and values before touching the simulator.

    Args: action: invalid vector. Returns: None. Raises: AssertionError.
    Design:
        Context: Real-time queue input may be malformed. Concern: Invalid actuation.
        Decision: No env attribute exists, so any premature stepping fails.
        Alternatives: Mocking only the validator misses method ordering.
        Quality Attributes: Integrity. Trade-offs: No dynamics in unit test.
    Evidence:
        Internal: step validates before env.step. External: None.
        Findings: Rejection precedes mutation. Metrics: Expected exception. Confidence: High.
    Constraints: Testing initialized-not-terminal state.
    Reconsider When: Action representation changes.
    References: FrozenLiberoEnv.step.
    ADR: None.
    """
    env=object.__new__(module.FrozenLiberoEnv);env.done=False
    with pytest.raises(ValueError):env.step(action)


def test_hold_zeroes_delta_and_keeps_gripper():
    """A starvation hold must not replay the last displacement.

    Args: None. Returns: None. Raises: AssertionError.
    Design:
        Context: Relative commands differ from absolute targets.
        Concern: Repetition creates uncontrolled drift.
        Decision: Assert six zeros and retained gripper sign.
        Alternatives: Full last-action repetition is incorrect.
        Quality Attributes: Predictability. Trade-offs: Controller still holds under dynamics.
    Evidence:
        Internal: hold_action. External: None.
        Findings: No motion increment persists. Metrics: Exact equality. Confidence: High.
    Constraints: Simulation fallback only.
    Reconsider When: Absolute controller used.
    References: libero_env.py.
    ADR: None.
    """
    env=object.__new__(module.FrozenLiberoEnv);env.last_action=np.array([.3,.1,-.2,.1,0,0,.7])
    np.testing.assert_allclose(env.hold_action(),[0,0,0,0,0,0,.7])


def test_native_controller_clip_is_common_to_models():
    """Preserve canonical bounds for finite unbounded flow-policy outputs.

    Args: None. Returns: None. Raises: AssertionError.
    Design:
        Context: Both policies may predict finite tails beyond nominal action bounds.
        Concern: Rejecting one model changes task/controller semantics.
        Decision: Common native clipping while parent logs raw chunk values.
        Alternatives: Model-specific rejection confounds comparisons.
        Quality Attributes: Fairness. Trade-offs: Raw proposals differ from applied commands.
    Evidence:
        Internal: LIBERO OSC input range. External: controller.scale_action.
        Findings: Both models use the same physical actuator limits.
        Metrics: Exact clipped command. Confidence: High.
    Constraints: Nonfinite inputs still fail before physics.
    Reconsider When: Controller limits change.
    References: validate_action.
    ADR: None.
    """
    np.testing.assert_array_equal(module.validate_action([2,-3,0,0,0,0,.5]),[1,-1,0,0,0,0,.5])
