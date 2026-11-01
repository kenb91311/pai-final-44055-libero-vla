"""Test causal state-origin scheduling and SI-unit forecasting without model or simulator loads.

Args: Pytest. Returns: Deterministic assertions. Raises: Test failures.
Design: Context: VLASH inputs may represent past, present or predicted robot state.
    Concern: Timing/units errors can masquerade as model failure.
    Decision: Analytic forecast and queue activation cases independent of task success.
    Alternatives: Only inspect end-to-end videos. Quality Attributes: Fault isolation. Trade-offs: No contact dynamics.
Evidence: Internal: vlash_timing.py. External: Native OSC scales and critical damping formula.
    Findings: Early results must retain the old queue until target time. Metrics: Exact mechanics.
    Confidence: Unit contracts only.
Constraints: No torch/GPU/model/ROS imports. Reconsider When: Controller or scheduler changes.
References: vlash_timing.py. ADR: None.
"""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from experiments.realtime_vla.vlash_timing import (
    StateOriginQueue,critical_step,forecast_robot_sensors,quaternion_exp,quaternion_log,quaternion_product,
)
from experiments.realtime_vla.vlash_worker import VLASHWorker


def test_critical_step_has_physical_displacement_and_velocity():
    """Compare native-scale prediction against the closed-form step response.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Normalized command1 means a .05m target. Concern: Treating it as .05m actual motion.
        Decision: Independent analytic response. Alternatives: Mirror implementation loops.
        Quality Attributes: Unit correctness. Trade-offs: Free dynamics only.
    Evidence: Internal: kp150,dt.05. External: Critical damping response.
        Findings: Actual predicted displacement is smaller than target. Metrics: Formula equality.
        Confidence: Analytic. Constraints: Zero initial velocity. Reconsider When: Gains change.
    References: critical_step. ADR: None.
    """
    dt=.05;omega=math.sqrt(150);target=np.array([.05,0,0]);delta,velocity=critical_step(target,np.zeros(3),dt)
    np.testing.assert_allclose(delta,target*(1-(1+omega*dt)*math.exp(-omega*dt)),atol=1e-14)
    np.testing.assert_allclose(velocity,target*omega**2*dt*math.exp(-omega*dt),atol=1e-14)
    assert 0<delta[0]<.05


def test_zero_time_keeps_measured_state():
    """Require zero-horizon integration to preserve position and velocity.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Forecast boundary. Concern: Applying command immediately as state.
        Decision: Zero dt invariant. Alternatives: Unscaled action addition.
        Quality Attributes: Physical consistency. Trade-offs: None.
    Evidence: Internal: Closed-form equation. External: Initial-value dynamics.
        Findings: Delta0/velocity unchanged. Metrics: Equality. Confidence: Analytic.
    Constraints: Finite vectors. Reconsider When: Formula changes. References: critical_step. ADR: None.
    """
    velocity=np.array([1.,2.,3.]);displacement,terminal=critical_step(np.array([.1,.2,.3]),velocity,0.)
    np.testing.assert_allclose(displacement,0);np.testing.assert_allclose(terminal,velocity)


def test_so3_round_trip_and_world_left_composition():
    """Check quaternion normalization, sign equivalence and noncommuting left rotation.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Native orientation scale is axis-angle radians. Concern: Euler/vector addition.
        Decision: Group composition and shortest-log invariants. Alternatives: Raw quaternion differencing.
        Quality Attributes: Geometric correctness. Trade-offs: Avoid pi branch.
    Evidence: Internal: Quaternion helpers. External: SO3 algebra.
        Findings: Opposite signs describe the same orientation. Metrics: Round-trip error.
        Confidence: Analytic. Constraints: Small rotations. Reconsider When: Convention changes.
    References: quaternion_exp. ADR: None.
    """
    vector=np.array([.1,-.2,.3]);q=quaternion_exp(vector)
    np.testing.assert_allclose(quaternion_log(q),vector,atol=1e-12)
    np.testing.assert_allclose(quaternion_log(-q),vector,atol=1e-12)
    x=quaternion_exp([.3,0,0]);z=quaternion_exp([0,0,.2])
    assert not np.allclose(quaternion_product(z,x),quaternion_product(x,z))


def test_forecast_uses_only_robot_sensors_and_holds_fingers():
    """Exercise native positional/rotational scaling and causal measured velocity.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Predicted proprioception. Concern: Object truth or incompatible action/state dimensions.
        Decision: Three measured robot fields and known prefix. Alternatives: Simulator lookahead.
        Quality Attributes: Causality. Trade-offs: Fingers held under contact uncertainty.
    Evidence: Internal: One native command and two sensor samples. External: OSC scale convention.
        Findings: Finite3+4+2 output, unchanged fingers. Metrics: Horizon and units.
        Confidence: Contract. Constraints: No contact model. Reconsider When: Forecast changes.
    References: forecast_robot_sensors. ADR: None.
    """
    old={'robot0_eef_pos':np.array([0.,0,0]),'robot0_eef_quat':np.array([0.,0,0,1]),'robot0_gripper_qpos':np.array([.02,-.02])}
    now=dict(old,robot0_eef_pos=np.array([.001,0,0]))
    actions=np.zeros((1,7));actions[0,[0,5]]=1
    predicted,meta=forecast_robot_sensors({'tick':1,'raw':now},{'tick':0,'raw':old},actions,1)
    assert set(predicted)==set(old) and predicted['robot0_eef_pos'][0]>.001
    np.testing.assert_allclose(predicted['robot0_gripper_qpos'],old['robot0_gripper_qpos'])
    np.testing.assert_allclose(np.linalg.norm(predicted['robot0_eef_quat']),1,atol=1e-7)
    assert meta['linear_velocity_m_s']==[.02,0.,0.] and meta['horizon_seconds']==.05
    assert 0<quaternion_log(predicted['robot0_eef_quat'])[2]<.5


def test_predicted_quaternion_matches_native_sensor_hemisphere_across_pi():
    """Preserve native axis-angle token convention when a forecast crosses pi.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: LIBERO sensors use robosuite mat2quat with nonnegative quaternion scalar.
        Concern: Equivalent q/-q can encode different axis-angle state vectors.
        Decision: Test a measured pose just below pi and positive rotation crossing that boundary.
        Alternatives: Normalize quaternion length only. Quality Attributes: Model-input fidelity.
        Trade-offs: The native axis-angle branch discontinuity remains, as in measured sensing.
    Evidence: Internal: Installed robosuite transform_utils.mat2quat lines351-355.
        External: Quaternion double cover. Findings: Physical orientation is unchanged by sign choice.
        Metrics: Scalar sign and equivalent predicted rotation. Confidence: Analytic.
    Constraints: No model/task feedback. Reconsider When: Native sensor convention changes.
    References: forecast_robot_sensors. ADR: None.
    """
    measured={'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':quaternion_exp([math.pi-.01,0,0]),
              'robot0_gripper_qpos':np.zeros(2)}
    actions=np.zeros((1,7));actions[0,3]=1
    predicted,_=forecast_robot_sensors({'tick':0,'raw':measured},None,actions,1)
    q=predicted['robot0_eef_quat']
    assert q[3]>=0 and q[0]<0
    delta,_=critical_step(np.array([.5,0,0]),np.zeros(3),.05)
    expected=quaternion_product(quaternion_exp(delta),measured['robot0_eef_quat'])
    np.testing.assert_allclose(q,-expected,atol=1e-7)


def test_early_chunk_waits_then_replaces_at_its_state_origin():
    """Continue old controls while a future-state result waits in the pending slot.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Overestimated inference latency. Concern: Running future actions early.
        Decision: Pending target-tick gate. Alternatives: Block physics or rebase origin.
        Quality Attributes: Causality. Trade-offs: Old queue continues.
    Evidence: Internal: Distinct old/new commands. External: State/action offset semantics.
        Findings: New chunk index0 starts at target. Metrics: Activation tick. Confidence: Deterministic.
    Constraints: One pending. Reconsider When: Scheduler changes. References: StateOriginQueue. ADR: None.
    """
    queue=StateOriginQueue();old=np.zeros((50,7));old[:,0]=.1;new=old.copy();new[:,0]=.8
    message={'actions':old,'request_id':'old','state_origin_tick':0,'observation_tick':0,'observation_time':1.,
             'measured_proprio_tick':0,'state_condition':'current'}
    queue.accept(message,0)
    result=queue.accept(dict(message,actions=new,request_id='future',state_origin_tick=4,state_condition='predicted'),2)
    assert result['deferred'] and queue.advance(3) is None
    assert queue.pop()['action'][0]==.1
    activated=queue.advance(4);item=queue.pop()
    assert activated['expired_prefix_actions']==0 and item['request_id']=='future' and item['chunk_index']==0
    assert item['image_capture_tick']==0 and item['state_origin_tick']==4 and item['state_is_prediction']


@pytest.mark.parametrize('origin,expected_drop',[(2,5),(5,2),(7,0)])
def test_late_expiration_uses_state_origin_not_image_tick(origin,expected_drop):
    """Drop different prefixes for the same stale image according to represented state time.

    Args: origin/expected_drop: parameterized state time. Returns: None. Raises: AssertionError.
    Design: Context: Coherent/current/predicted comparisons. Concern: All chunks indexed at image capture.
        Decision: Same image2/arrival7 with changing state origin. Alternatives: One source timestamp.
        Quality Attributes: Temporal correctness. Trade-offs: None.
    Evidence: Internal: Explicit queue metadata. External: VLASH state/target alignment.
        Findings: First action index equals arrival-origin. Metrics: Dropped prefix. Confidence: Deterministic.
    Constraints: Fixed H50. Reconsider When: Alignment changes. References: StateOriginQueue.accept. ADR: None.
    """
    queue=StateOriginQueue();message={'actions':np.zeros((50,7)),'request_id':'x','state_origin_tick':origin,
                                    'observation_tick':2,'observation_time':1.,'measured_proprio_tick':2,'state_condition':'current'}
    event=queue.accept(message,7);item=queue.pop()
    assert event['expired_prefix_actions']==expected_drop and item['chunk_index']==expected_drop
    assert item['image_capture_tick']==2


def test_worker_request_drops_truth_and_labels_prediction_separately():
    """Verify model packet allowlist and prediction provenance without spawning a process.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Mixed input ages. Concern: Truth leakage or future state called measured.
        Decision: Capture/query/measured/target ticks independently in a mocked send.
        Alternatives: Forward entire simulator dictionary. Quality Attributes: Auditability. Trade-offs: No transport here.
    Evidence: Internal: VLASHWorker request builder. External: Shared five-sensor schema.
        Findings: Object/success fields excluded. Metrics: Exact allowlist. Confidence: Deterministic.
    Constraints: No GPU/ROS. Reconsider When: Request schema changes. References: VLASHWorker.predict. ADR: None.
    """
    worker=VLASHWorker.__new__(VLASHWorker);worker.send=Mock(return_value='request')
    observation={'agentview_image':np.zeros((2,2,3),np.uint8),'robot0_eye_in_hand_image':np.ones((2,2,3),np.uint8),
                 'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0,0,0,1]),'robot0_gripper_qpos':np.zeros(2),
                 'prompt':'task','metrics':{'secret_object_truth':1},'success':True}
    worker.predict(observation,2,1.,request_tick=5,measured_proprio_tick=5,measured_proprio_time=1.2,
                   state_origin_tick=12,state_condition='predicted',predictor={'horizon_ticks':7})
    packet=worker.send.call_args.args[0]
    assert set(packet['observation'])=={'agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'}
    assert (packet['observation_tick'],packet['measured_proprio_tick'],packet['state_origin_tick'])==(2,5,12)
