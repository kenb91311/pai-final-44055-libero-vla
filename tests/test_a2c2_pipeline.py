"""Check A2C2 next-tick source/deadline gates and nonblocking physical scheduling on CPU.

Args: pytest. Returns: Assertions. Raises: AssertionError on causal/provenance regressions.
Design: Context: GPU trials must not be used to debug control protocol mechanics.
    Concern: Late relative commands or hidden RPC waits corrupt realtime evaluation.
    Decision: Deterministic fake clock/transport and actual queue/executor code.
    Alternatives: Test only matching shapes. Quality Attributes: Reliability. Trade-offs: No task claim.
Evidence: Internal: v1 TCP timeout despite DDS receipt. External: A2C2 context contract.
    Findings: Corrected values require exact source and target. Metrics: Assertions.
    Confidence: Contract level.
Constraints: No GPU, network, model restoration or simulation.
Reconsider When: Pipeline protocol changes. References: evaluate_a2c2.py. ADR: None.
"""
from copy import deepcopy
import json
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from experiments.realtime_vla.a2c2_worker import ActuatorWorker
from experiments.realtime_vla.evaluate_a2c2 import choose_received_action, collect_audits, run_episode


def gate_fixture():
    """Build one source item and its valid received next-tick correction.

    Args: None. Returns: Item/context/packet. Raises: None.
    Design: Context: Target3/sourceA/index3. Concern: Mutation between cases.
        Decision: Fresh dictionaries. Alternatives: Shared mutable state.
        Quality Attributes: Isolation. Trade-offs: None.
    Evidence: Internal: Executor schema. External: None. Findings: Deadline10.
        Metrics: None. Confidence: Fixture. Constraints: CPU only.
    Reconsider When: Schema changes. References: choose_received_action. ADR: None.
    """
    item = {'action':np.zeros(7), 'held':False, 'request_id':'source-A', 'chunk_index':3}
    context = SimpleNamespace(generation=2)
    packet = {'kind':'action', 'target_tick':3, 'target_deadline':10., 'finished_time':9.,
              'source_request_id':'source-A', 'source_generation':2, 'original_action_index':3,
              'proposal_kind':'correction', 'received_action':np.full(7, .2)}
    return item, context, packet


@pytest.mark.parametrize('field,value', [('target_tick', 2), ('source_request_id', 'source-B'),
                                         ('source_generation', 3), ('original_action_index', 4),
                                         ('finished_time', 10.001)])
def test_reject_temporal_mismatch(field, value):
    """Reject each independently stale provenance component instead of repeating a correction.

    Args: Mismatched field/value. Returns: None. Raises: AssertionError.
    Design: Context: New source can arrive after submission. Concern: Wrong command.
        Decision: Require base fallback. Alternatives: Last correction.
        Quality Attributes: Integrity. Trade-offs: None.
    Evidence: Internal: Production gate. External: None. Findings: Five separate components.
        Metrics: Assertions. Confidence: Contract. Constraints: CPU only.
    Reconsider When: Protocol changes. References: choose_received_action. ADR: None.
    """
    item, context, packet = gate_fixture()
    packet[field] = value
    action, origin, reason = choose_received_action(item, context, packet, 3)
    np.testing.assert_array_equal(action, item['action'])
    assert origin == 'base_ros_chunk' and reason


def test_received_values_and_executor_hold():
    """Use the received correction only when valid and label empty-queue holds as derived.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Holds are a controller rule. Concern: False DDS claims.
        Decision: Separate origin labels and exact returned values.
        Alternatives: Reuse a motion delta. Quality Attributes: Traceability. Trade-offs: None.
    Evidence: Internal: Gate code. External: None. Findings: Hold never adopts packet delta.
        Metrics: Assertions. Confidence: Contract. Constraints: CPU only.
    Reconsider When: Hold rule changes. References: choose_received_action. ADR: None.
    """
    item, context, packet = gate_fixture()
    action, origin, reason = choose_received_action(item, context, packet, 3)
    np.testing.assert_array_equal(action, packet['received_action'])
    assert origin == 'corrected_ros_packet' and reason is None
    item.update(held=True, action=np.array([0.]*6+[.7]))
    action, origin, _ = choose_received_action(item, None, packet, 3)
    np.testing.assert_array_equal(action, item['action'])
    assert origin == 'derived_controller_hold'


def test_actuator_has_one_inflight_and_timeout_is_data():
    """Reject backlog and release its slot when a timeout result is polled.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: A slow request must not accumulate old deltas.
        Concern: Queue backlog. Decision: Actual bounded send/poll without spawning.
        Alternatives: Unlimited queue. Quality Attributes: Boundedness. Trade-offs: No ROS.
    Evidence: Internal: Production lifecycle. External: queue.
        Findings: Timeout is explicit data. Metrics: Assertions. Confidence: Contract.
    Constraints: CPU/no network. Reconsider When: Lifecycle changes.
    References: ActuatorWorker. ADR: None.
    """
    actor = object.__new__(ActuatorWorker)
    actor.input, actor.output = Queue(maxsize=1), Queue(maxsize=1)
    actor.counter, actor.inflight, actor.prefix = 0, False, 'unit'
    kwargs = dict(source_request_id='source', source_generation=1, original_action_index=1,
                  target_tick=1, target_deadline=1., proposal_kind='correction')
    actor.submit(np.zeros(7), **kwargs)
    with pytest.raises(RuntimeError):
        actor.submit(np.zeros(7), **kwargs)
    actor.input.get_nowait()
    actor.output.put({'kind':'timeout', 'request_id':'unit-query-1'})
    assert actor.poll()['kind'] == 'timeout' and not actor.inflight
    actor.submit(np.zeros(7), **kwargs)


class FakeActuator:
    """Return two valid packets around one deliberately delayed response using a fake clock.

    Args: mutable clock. Returns: CPU-only fake transport.
    Raises: AssertionError if an operation would block control.
    Design: Context: Real executor calls. Concern: Hidden waits.
        Decision: Poll only, target2 delayed until tick3. Alternatives: Sleep tests.
        Quality Attributes: Determinism. Trade-offs: Synthetic.
    Evidence: Internal: Protocol. External: None. Findings: Received gripper differs from proposal.
        Metrics: Call records. Confidence: Fixture. Constraints: No IO.
    Reconsider When: API changes. References: run_episode. ADR: None.
    """
    def __init__(self, clock):
        """Initialize the bounded fake mailbox.

        Args: clock list. Returns: None. Raises: None.
        Design: Context: Fake time. Concern: State leak. Decision: Per-test owner.
            Alternatives: Global state. Quality Attributes: Isolation. Trade-offs: None.
        Evidence: Internal: Test. External: None. Findings: Empty mailbox.
            Metrics: Count. Confidence: Fixture. Constraints: CPU only.
        Reconsider When: API changes. References: Class. ADR: None.
        """
        self.clock, self.inflight, self.pending, self.count = clock, False, None, 0

    def submit(self, action, **identity):
        """Keep one proposal and make only target2 arrive after its deadline.

        Args: action/identity: production fields. Returns: Request ID. Raises: Busy assertion.
        Design: Context: Delayed response. Concern: Old command applied later.
            Decision: Deliberate source-preserving delay. Alternatives: Random timing.
            Quality Attributes: Determinism. Trade-offs: Synthetic.
        Evidence: Internal: Clock. External: None. Findings: Arrival index differs from target.
            Metrics: Count. Confidence: Fixture. Constraints: No IO.
        Reconsider When: API changes. References: Class. ADR: None.
        """
        assert not self.inflight
        self.count += 1
        request = f'actor-{self.count}'
        received = np.asarray(action).copy()
        received[6] = .25
        finished = identity['target_deadline']+.02 if identity['target_tick'] == 2 else self.clock[0]+.005
        self.pending = dict(identity, kind='action', request_id=request, proposed_action=np.asarray(action),
                            received_action=received, started_time=self.clock[0], finished_time=finished,
                            action_transport={'rpc_wall_seconds':.005})
        self.inflight = True
        return request

    def poll(self):
        """Return a ready packet without advancing fake time.

        Args: None. Returns: Packet/None. Raises: None.
        Design: Context: Nonblocking loop. Concern: Waiting. Decision: Clock comparison only.
            Alternatives: Sleep. Quality Attributes: Timeliness. Trade-offs: Synthetic.
        Evidence: Internal: Clock. External: None. Findings: One slot cleared.
            Metrics: None. Confidence: Fixture. Constraints: No IO.
        Reconsider When: API changes. References: Class. ADR: None.
        """
        if self.pending['finished_time'] > self.clock[0]:
            return None
        self.inflight = False
        return self.pending


def test_pipeline_physics_does_not_wait_for_delayed_transport(tmp_path):
    """Advance all ticks, use actual received values and fall back during one delayed packet.

    Args: tmp_path: isolated output. Returns: None. Raises: AssertionError.
    Design: Context: CPU end-to-end scheduler. Concern: v1 control-blocking regression.
        Decision: Exact fake clock and production run_episode.
        Alternatives: GPU task as unit test. Quality Attributes: Reliability. Trade-offs: Synthetic.
    Evidence: Internal: Target1/4 ready, target2 late. External: None.
        Findings: Only bootstrap waits for base and only startup probes ROS synchronously.
        Metrics: Origin/index/call sequence. Confidence: Contract.
    Constraints: No GPU/model/network. Reconsider When: Pipeline changes.
    References: run_episode. ADR: None.
    """
    import experiments.realtime_vla.evaluate_a2c2 as evaluation
    clock, applied, indices = [0.], [], []
    raw = {'agentview_image':np.zeros((256, 256, 3), np.uint8), 'robot0_eye_in_hand_image':np.zeros((256, 256, 3), np.uint8),
           'robot0_eef_pos':np.zeros(3), 'robot0_eef_quat':np.array([0, 0, 0, 1]), 'robot0_gripper_qpos':np.zeros(2),
           'prompt':'task', 'tick':0, 'capture_time':0., 'episode_id':'author-smol-a2c2-init0'}
    initial = {'metrics':{'goal_xy_error_m':1.}, 'success':False, 'done':False}
    env, camera = MagicMock(), MagicMock()
    env.metadata, camera.metadata = {}, {}
    env.reset.return_value = initial
    env.snapshot.return_value = {}
    env.step.side_effect = lambda action:(applied.append(np.asarray(action).copy()), clock.__setitem__(0, clock[0]+.01), dict(initial, done=len(applied) == 5))[-1]
    camera.wait.return_value, camera.poll.return_value = raw, None
    actions = np.full((50, 7), .1, np.float32)
    context = SimpleNamespace(actions=actions, vlm_hidden=np.zeros((1, 960), np.float32), generation=1)
    packet = {'kind':'chunk', 'request_id':'base', 'actions':actions, 'context':context, 'metadata':{'peak_cuda_bytes':0},
              'observation_tick':0, 'observation_time':0., 'started_time':0., 'finished_time':0.}
    worker = SimpleNamespace(warmed=True, warmup_records=[], inflight=False,
                             reset=MagicMock(), predict=MagicMock(), wait=MagicMock(return_value=packet))
    head = SimpleNamespace(device=SimpleNamespace(type='cpu'), reset=MagicMock())
    head.correct = MagicMock(side_effect=lambda sensor, prompt, source, index:(indices.append(index), clock.__setitem__(0, clock[0]+.01), (np.array([2., -2., 0., 0., 0., 0., .9]), {'correction_ms':10.}))[-1])
    actor = FakeActuator(clock)
    probe = MagicMock()
    probe.actions.return_value = (np.zeros((1, 7)), {'verified':True})
    args = SimpleNamespace(control_hz=20, wall_seconds=.25, seed=20261102, variant='a2c2', video=False,
                           ros_port=8780, action_timeout=.5, prefetch_steps=30, max_sensor_age=.5, camera_hz=10.)
    with patch.object(evaluation.time, 'monotonic', side_effect=lambda:clock[0]), \
         patch.object(evaluation.time, 'sleep', side_effect=lambda seconds:clock.__setitem__(0, clock[0]+seconds)), \
         patch.object(evaluation, 'RosRoundTrip', return_value=probe):
        result = run_episode(env, camera, worker, head, actor, args, 0, tmp_path/'episode')
    assert result['status'] == 'completed' and len(applied) == 5
    assert worker.wait.call_count == 1 and probe.actions.call_count == 1
    assert indices == [1, 2, 4]
    assert result['action_origin_counts'] == {'base_ros_chunk':3, 'base_action_ros_packet':0,
                                             'corrected_ros_packet':2, 'derived_controller_hold':0}
    assert applied[1][6] == .25 and applied[4][6] == .25
    assert applied[2][6] == pytest.approx(.1)
    np.testing.assert_array_equal(applied[1][:2], [1., -1.])
    packets = json.loads((tmp_path/'episode'/'actuator-packets.json').read_text())
    assert [p['used'] for p in packets] == [True, False, True]
    assert packets[1]['rejection_reason'] == 'wrong_target_tick'


def test_error_audit_collects_available_weights(tmp_path):
    """Retain both after digests even if one owner cannot be audited after a failure.

    Args: tmp_path. Returns: None. Raises: AssertionError.
    Design: Context: Failure path. Concern: Missing after evidence.
        Decision: Independent attempts. Alternatives: Abort after first error.
        Quality Attributes: Traceability. Trade-offs: None.
    Evidence: Internal: collect_audits. External: None. Findings: No fabricated base audit.
        Metrics: Files/errors. Confidence: Contract. Constraints: No GPU.
    Reconsider When: Owner lifecycle changes. References: collect_audits. ADR: None.
    """
    worker = SimpleNamespace(audit=MagicMock(side_effect=RuntimeError('base unavailable')))
    head = SimpleNamespace(audit=MagicMock(return_value={'weights_unchanged':True}))
    base_audit, head_audit, errors = collect_audits(worker, head, tmp_path)
    assert base_audit is None and head_audit['weights_unchanged']
    assert 'base' in errors and (tmp_path/'head-model-audit.json').is_file()


def test_direct_and_residual_use_identical_inverse_statistics():
    """Separate output semantics while preserving the saved nontrivial action normalizer.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Same head graph can predict an action or an action residual.
        Concern: Shape-compatible wrong decode can silently double base commands.
        Decision: Check both exact equations using nonzero mean and nonunit std.
        Alternatives: Unit/zero stats would hide decoding mistakes.
        Quality Attributes: Semantic fidelity. Trade-offs: No trained model required.
    Evidence: Internal: Author75fa/f7b17 diff. External: None.
        Findings: Direct excludes base addition before inverse stats.
        Metrics: Tensor closeness and unchanged stats. Confidence: Algebraic.
    Constraints: CPU only, no model/GPU fitting.
    Reconsider When: Output semantics change. References: decode_head_output. ADR: None.
    """
    import torch
    from experiments.realtime_vla.a2c2_runtime import LegacyMeanStd, decode_head_output
    mean = torch.arange(7, dtype=torch.float32)-3
    std = torch.arange(7, dtype=torch.float32)+2
    head, base = torch.full((1, 7), .25), torch.full((1, 7), 2.)
    normalizer = LegacyMeanStd({'unnormalize_outputs.buffer_action.mean':mean,
                                'unnormalize_outputs.buffer_action.std':std}, 'cpu')
    direct = decode_head_output(head, base, normalizer, 'direct')
    residual = decode_head_output(head, base, normalizer, 'residual')
    torch.testing.assert_close(direct, head*std+mean)
    torch.testing.assert_close(residual, (base+head)*std+mean)
    torch.testing.assert_close(residual-direct, base*std)
    torch.testing.assert_close(normalizer.tensors['unnormalize_outputs.buffer_action.mean'], mean)
    torch.testing.assert_close(normalizer.tensors['unnormalize_outputs.buffer_action.std'], std)
