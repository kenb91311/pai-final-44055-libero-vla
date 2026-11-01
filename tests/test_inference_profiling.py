"""Verify profiling reversibility, interval accounting and pairing without GPU initialization.

Args: Pytest CPU process. Returns: Mechanical assertions. Raises: AssertionError on contract violations.
Design: Context: Native benchmark validity depends on the measurement harness.
    Concern: Hooks, nesting or RNG mistakes can invent apparent speed differences.
    Decision: Analytic event clocks and deterministic fake predictions test protocol independently.
    Alternatives: Only inspect real GPU timing output. Quality Attributes: Fault isolation.
    Trade-offs: Tests establish measurement mechanics, not real model performance.
Evidence: Internal: profile_inference.py and profiling_hooks.py. External: None.
    Findings: No Torch/model/simulator imports are necessary. Metrics: Exact partitions/counts/pairing.
    Confidence: CPU protocol evidence.
Constraints: No GPU calls, pretrained assets or model library source modifications.
Reconsider When: Multi-stream/concurrent inference is supported.
References: Profile report schema_version1. ADR: None.
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.realtime_vla.profile_inference import (
    compare_conditions, load_observations, make_schedule, run_condition,
)
from experiments.realtime_vla.profiling_hooks import IntervalRecorder, NativeInstrumentation


class ManualClock:
    """Expose an analytic monotonic clock shared by fake host and CUDA event timers.

    Args: None. Returns: Mutable seconds clock. Raises: None.
    Design: Context: Exact nested accounting. Concern: Wall-clock jitter obscures arithmetic errors.
        Decision: Test-controlled time. Alternatives: Sleep. Quality Attributes: Determinism.
        Trade-offs: No GPU behavior coverage.
    Evidence: Internal: test_nested_intervals. External: None. Findings: Expected sums are analytic.
        Metrics: Milliseconds. Confidence: Exact.
    Constraints: Test only. Reconsider When: Backend contract changes. References: FakeCUDA. ADR: None.
    """
    def __init__(self):
        """Start the analytic clock at zero.

        Args: None. Returns: None. Raises: None.
        Design: Context: Test setup. Concern: Initial offsets. Decision: Zero. Alternatives: Real clock.
            Quality Attributes: Determinism. Trade-offs: None.
        Evidence: Internal: Nested timer test. External: None. Findings: Exact origin. Metrics: Seconds0.
            Confidence: Exact. Constraints: Test only. Reconsider When: None. References: ManualClock. ADR: None.
        """
        self.value = 0.

    def __call__(self):
        """Read controlled seconds without advancing time.

        Args: None. Returns: Float seconds. Raises: None.
        Design: Context: Analytic timer. Concern: Hidden time increments. Decision: Pure read.
            Alternatives: Auto-increment. Quality Attributes: Clarity. Trade-offs: Explicit test assignments.
        Evidence: Internal: Nested intervals. External: None. Findings: No jitter. Metrics: Seconds.
            Confidence: Exact. Constraints: Test only. Reconsider When: None. References: ManualClock. ADR: None.
        """
        return self.value


class FakeEvent:
    """Implement the event record/elapsed contract using the analytic clock.

    Args: clock: ManualClock. Returns: Fake CUDA event. Raises: Unrecorded event attribute errors.
    Design: Context: No GPU unit tests. Concern: Accidental synchronize calls.
        Decision: Minimal asynchronous-looking event interface. Alternatives: Real CUDA.
        Quality Attributes: Portability. Trade-offs: No event resolution quantization.
    Evidence: Internal: IntervalRecorder API. External: None. Findings: Event ordering matches host ordering.
        Metrics: Elapsed milliseconds. Confidence: Exact arithmetic.
    Constraints: CPU only. Reconsider When: Timing interface changes. References: FakeCUDA. ADR: None.
    """
    def __init__(self, clock):
        """Retain the test clock until event recording.

        Args: clock: ManualClock. Returns: None. Raises: None.
        Design: Context: Fake event. Concern: Premature timestamps. Decision: Record later.
            Alternatives: Constructor timestamp. Quality Attributes: Contract fidelity. Trade-offs: None.
        Evidence: Internal: measure. External: None. Findings: Lazy record. Metrics: None. Confidence: Exact.
        Constraints: Test only. Reconsider When: None. References: FakeEvent. ADR: None.
        """
        self.clock = clock

    def record(self, stream):
        """Store a controlled event timestamp without waiting.

        Args: stream: fake stream. Returns: None. Raises: None.
        Design: Context: Asynchronous event contract. Concern: Hidden synchronization.
            Decision: Timestamp only. Alternatives: Wait. Quality Attributes: Test isolation. Trade-offs: No GPU.
        Evidence: Internal: measure. External: None. Findings: No waits. Metrics: Seconds. Confidence: Exact.
        Constraints: Test only. Reconsider When: None. References: FakeEvent. ADR: None.
        """
        self.timestamp = self.clock()

    def elapsed_time(self, other):
        """Calculate the independent analytic elapsed interval.

        Args: other: recorded end event. Returns: Milliseconds. Raises: Missing timestamp errors.
        Design: Context: Accounting oracle. Concern: Mirroring recorder subtraction.
            Decision: Direct timestamp difference. Alternatives: Reuse production accounting.
            Quality Attributes: Independence. Trade-offs: Exact resolution only.
        Evidence: Internal: Analytic test. External: None. Findings: Millisecond conversion. Metrics: Elapsed.
            Confidence: Exact. Constraints: Test only. Reconsider When: Units change. References: finish. ADR: None.
        """
        return (other.timestamp - self.timestamp) * 1000


class FakeCUDA:
    """Provide events and stream identity while making unexpected waits fail loudly.

    Args: clock: ManualClock. Returns: Fake backend. Raises: AssertionError on synchronize.
    Design: Context: No per-stage waits requirement. Concern: Hidden GPU wait regressions.
        Decision: Backend with forbidden synchronize. Alternatives: Mock that silently accepts everything.
        Quality Attributes: Regression detection. Trade-offs: Test-only backend.
    Evidence: Internal: IntervalRecorder. External: None. Findings: finish needs only elapsed_time.
        Metrics: No synchronize calls. Confidence: Exact.
    Constraints: Single fake stream. Reconsider When: Multi-stream support added. References: FakeEvent. ADR: None.
    """
    def __init__(self, clock):
        """Store the clock and one stable stream handle.

        Args: clock: ManualClock. Returns: None. Raises: None.
        Design: Context: Fake backend. Concern: Unstable stream ID. Decision: Handle17.
            Alternatives: Random handle. Quality Attributes: Determinism. Trade-offs: None.
        Evidence: Internal: Stream check test. External: None. Findings: Stable identity. Metrics: None.
            Confidence: Exact. Constraints: Test only. Reconsider When: None. References: FakeCUDA. ADR: None.
        """
        self.clock = clock
        self.stream = SimpleNamespace(cuda_stream=17)

    def current_stream(self, device):
        """Expose the backend's current stream identity.

        Args: device: unused fake device. Returns: Fake stream. Raises: None.
        Design: Context: Same-stream invariant. Concern: Unsupported overlap. Decision: Explicit stream identity.
            Alternatives: Ignore stream changes. Quality Attributes: Accounting safety. Trade-offs: None.
        Evidence: Internal: measure. External: None. Findings: Stable current stream. Metrics: ID17.
            Confidence: Exact. Constraints: Test only. Reconsider When: None. References: FakeCUDA. ADR: None.
        """
        return self.stream

    def Event(self, enable_timing):
        """Construct a timing event on the shared analytic clock.

        Args: enable_timing: must be true. Returns: FakeEvent. Raises: AssertionError if disabled.
        Design: Context: CUDA API substitute. Concern: Inaccurate event options.
            Decision: Assert timing enabled. Alternatives: Ignore flag. Quality Attributes: Contract detection.
            Trade-offs: None. Evidence: Internal: Recorder constructor. External: None. Findings: Timing required.
        Metrics: None. Confidence: Exact. Constraints: CPU only. Reconsider When: None. References: FakeEvent. ADR: None.
        """
        assert enable_timing
        return FakeEvent(self.clock)

    def synchronize(self, *args):
        """Reject any stage-level attempt to wait on the fake backend.

        Args: args: ignored. Returns: Never. Raises: AssertionError.
        Design: Context: Measurement invariant. Concern: Per-stage waits change native execution.
            Decision: Fail immediately. Alternatives: No-op. Quality Attributes: Regression sensitivity.
            Trade-offs: None. Evidence: Internal: Recorder contract. External: None. Findings: Wait forbidden.
        Metrics: Zero waits. Confidence: Exact. Constraints: Test only. Reconsider When: None. References: FakeCUDA. ADR: None.
        """
        raise AssertionError('recorder must not synchronize')


def test_nested_intervals_partition_without_double_counting_or_synchronization():
    """Require analytic parent/child host and stream partitions to recover one root total.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Prefix includes vision. Concern: Summing both double counts work.
        Decision: Independent clock assigns root10, prefix6, vision2, tail1 milliseconds.
        Alternatives: Check only report formatting. Quality Attributes: Accounting correctness.
        Trade-offs: Fake event backend only.
    Evidence: Internal: IntervalRecorder. External: None. Findings: Exclusive sum10, inclusive sum19.
        Metrics: Exact partition. Confidence: Analytic.
    Constraints: No CUDA. Reconsider When: Tree model changes. References: finish. ADR: None.
    """
    clock = ManualClock()
    recorder = IntervalRecorder(FakeCUDA(clock), 'fake', clock)
    with recorder.measure('request'):
        clock.value = .001
        with recorder.measure('prefix'):
            clock.value = .002
            with recorder.measure('vision'):
                clock.value = .004
            clock.value = .007
        clock.value = .008
        with recorder.measure('postprocess'):
            clock.value = .009
        clock.value = .010
    report = recorder.finish()
    assert report['accounting']['valid']
    assert report['accounting']['host_exclusive_partition_ms'] == pytest.approx(10.)
    assert report['accounting']['cuda_exclusive_partition_ms'] == pytest.approx(10.)
    assert report['stages']['prefix']['host_ms'] == pytest.approx(6.)
    assert report['stages']['prefix']['host_exclusive_ms'] == pytest.approx(4.)
    assert report['stages']['request']['host_exclusive_ms'] == pytest.approx(3.)
    assert sum(stage['host_ms'] for stage in report['stages'].values()) == pytest.approx(19.)


def test_method_and_callable_attributes_restore_exactly_even_after_exception():
    """Ensure inherited method shadows and prior own callables are restored on failure.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Frozen runtime reuse. Concern: Failed profiling leaves wrapped methods installed.
        Decision: Exercise exception path and compare complete instance attribute identity.
        Alternatives: Only successful path. Quality Attributes: Reversibility. Trade-offs: Tiny fake object.
    Evidence: Internal: NativeInstrumentation.wrap/__exit__. External: Python descriptor lookup.
        Findings: Inherited shadow is deleted; own callable identity preserved. Metrics: Exact restoration.
        Confidence: Direct mechanics.
    Constraints: No native models. Reconsider When: Attribute ownership changes. References: __exit__. ADR: None.
    """
    target = SimpleNamespace(callback=lambda: (_ for _ in ()).throw(ValueError('native failure')))
    original = target.callback
    instrumentation = NativeInstrumentation(None, 'fake')
    instrumentation.wrap(target, 'callback', 'native')
    instrumentation.recorder = IntervalRecorder()
    with pytest.raises(ValueError, match='native failure'):
        try:
            target.callback()
        finally:
            instrumentation.__exit__(None, None, None)
    assert target.callback is original
    assert not instrumentation.restorations
    clock = ManualClock()
    initial = vars(clock).copy()
    instrumentation.wrap(clock, '__call__', 'inherited')
    instrumentation.__exit__(None, None, None)
    assert vars(clock) == initial


def test_partial_native_hook_setup_rolls_back_all_installed_wrappers():
    """Require missing native entrypoints to fail loudly without retaining partial hooks.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Dependency upgrades can rename methods. Concern: Incomplete instrumentation looks valid.
        Decision: Deliberately omit image preparation and assert full setup rollback.
        Alternatives: Silently skip hooks. Quality Attributes: Integrity. Trade-offs: Pinned APIs fail early.
    Evidence: Internal: NativeInstrumentation.__enter__. External: None. Findings: Existing callable identities survive.
        Metrics: Attribute identity. Confidence: Direct mechanics.
    Constraints: CPU fake runtime. Reconsider When: Hook discovery changes. References: __enter__. ADR: None.
    """
    policy = SimpleNamespace(model=SimpleNamespace(), predict_action_chunk=lambda batch: batch)
    runtime = SimpleNamespace(policy=policy, preprocess=lambda batch: batch, postprocess=lambda batch: batch)
    prior_runtime, prior_policy = vars(runtime).copy(), vars(policy).copy()
    with pytest.raises(AttributeError):
        with NativeInstrumentation(runtime, 'smolvla'):
            pass
    assert vars(runtime) == prior_runtime
    assert vars(policy) == prior_policy


class FakeRuntime:
    """Emit deterministic seeded action chunks while recording reset and input order.

    Args: None. Returns: CPU fake runtime. Raises: None.
    Design: Context: B-I-B protocol test. Concern: Incorrect state/seed pairing.
        Decision: Separate reset and prediction logs with deterministic NumPy generator.
        Alternatives: Use expensive pretrained model. Quality Attributes: Isolation. Trade-offs: No model fidelity claim.
    Evidence: Internal: run_condition. External: None. Findings: RNG identity is independently observable.
        Metrics: Calls/seeds/actions. Confidence: Exact.
    Constraints: No GPU imports. Reconsider When: Sequence state is benchmarked. References: test_protocol. ADR: None.
    """
    def __init__(self):
        """Create empty logs and a CPU device marker.

        Args: None. Returns: None. Raises: None.
        Design: Context: Protocol fake. Concern: Accidental CUDA calls. Decision: CPU marker.
            Alternatives: Real device. Quality Attributes: Portability. Trade-offs: None.
        Evidence: Internal: synchronize. External: None. Findings: No CUDA needed. Metrics: Log lengths.
            Confidence: Exact. Constraints: Test only. Reconsider When: None. References: FakeRuntime. ADR: None.
        """
        self.device = SimpleNamespace(type='cpu')
        self.reset_log = []
        self.predict_log = []

    def reset(self, seed):
        """Record the seed and reset the independent fake RNG outside prediction.

        Args: seed: nonnegative integer. Returns: None. Raises: NumPy invalid seed errors.
        Design: Context: Stochastic flow analogue. Concern: Global history affecting pairs.
            Decision: New independent generator per reset. Alternatives: Global random state.
            Quality Attributes: Determinism. Trade-offs: None.
        Evidence: Internal: Paired condition test. External: NumPy Generator.
            Findings: Same seed emits same actions. Metrics: Seed equality. Confidence: Exact.
        Constraints: Test only. Reconsider When: None. References: predict. ADR: None.
        """
        self.reset_log.append(seed)
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, prompt):
        """Return deterministic native-shaped actions and record the observation identity.

        Args: observation: fake id array; prompt: test string. Returns: Actions2x7 and metadata.
        Raises: Missing schema fields.
        Design: Context: Paired identity oracle. Concern: Incorrect observation reuse.
            Decision: Encode observation ID in actions and log seed order. Alternatives: Constant zeros.
            Quality Attributes: Test sensitivity. Trade-offs: Not a learned model.
        Evidence: Internal: compare_conditions. External: None. Findings: Both seed and sample affect output.
            Metrics: Exact output identity. Confidence: Exact.
        Constraints: CPU only. Reconsider When: None. References: FakeRuntime. ADR: None.
        """
        identifier = float(observation['id'][0])
        self.predict_log.append((identifier, self.reset_log[-1], prompt))
        return (self.rng.standard_normal((2, 7)) + identifier).astype(np.float32), {'fake': True}


def test_bib_protocol_pairs_observations_and_seeds_and_excludes_warmup():
    """Check schedule, all three prediction counts and exact actions with recording enabled.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Native benchmark must compare identical requests. Concern: Warmup or seed offsets differ.
        Decision: Three states, two repeats and two warmups in each condition with a fake runtime.
        Alternatives: Test scheduler helper only. Quality Attributes: Protocol integrity.
        Trade-offs: No real GPU event accuracy claim.
    Evidence: Internal: run_condition/timed_predict/compare_conditions. External: None.
        Findings: Measured counts6 per condition; total24; paired actions exact.
        Metrics: Order/count/equality. Confidence: Direct protocol test.
    Constraints: CPU only. Reconsider When: Protocol ordering changes. References: make_schedule. ADR: None.
    """
    samples = [{'id': str(index), 'observation': {'id': np.array([index])}, 'sensor_hashes': {'id': str(index)}}
               for index in range(3)]
    schedule = make_schedule(samples, 2, 100)
    runtime = FakeRuntime()
    before, before_actions = run_condition(runtime, 'before', samples, schedule, 'move', 2, 100)
    instrumentation = NativeInstrumentation(runtime, 'fake')
    instrumented, instrumented_actions = run_condition(runtime, 'instrumented', samples, schedule, 'move', 2, 100,
                                                       instrumentation)
    after, after_actions = run_condition(runtime, 'after', samples, schedule, 'move', 2, 100)
    comparison = compare_conditions(before, instrumented, after, before_actions, instrumented_actions, after_actions)
    assert comparison['all_exact_equal'] and comparison['all_allclose']
    assert len(runtime.predict_log) == len(runtime.reset_log) == 24
    assert runtime.predict_log[:8] == runtime.predict_log[8:16] == runtime.predict_log[16:]
    assert before['latency_ms']['count'] == instrumented['latency_ms']['count'] == after['latency_ms']['count'] == 6
    assert len(instrumented['warmup']['calls']) == 2
    assert [call['seed'] for call in instrumented['calls']] == list(range(100, 106))
    assert all(call['profile']['accounting']['valid'] for call in instrumented['calls'])


def test_raw_observation_loader_retains_bytes_and_rejects_extra_sensor(tmp_path):
    """Validate exact dtype/hash preservation and prevent privileged input fields.

    Args: tmp_path: isolated directory. Returns: None. Raises: AssertionError.
    Design: Context: Shared captured inputs. Concern: dtype casting and object truth contaminate comparison.
        Decision: Preserve float64 bytes and reject an extra NPZ key. Alternatives: Filter silently.
        Quality Attributes: Input fidelity. Trade-offs: Strict schema.
    Evidence: Internal: load_observations. External: None. Findings: Sensor bytes remain identical.
        Metrics: SHA identity and exception. Confidence: Exact.
    Constraints: CPU arrays only. Reconsider When: Sensor schema changes. References: RAW_KEYS. ADR: None.
    """
    arrays = {'agentview_image': np.zeros((256, 256, 3), np.uint8),
              'robot0_eye_in_hand_image': np.ones((256, 256, 3), np.uint8),
              'robot0_eef_pos': np.array([.1, .2, .3], np.float64),
              'robot0_eef_quat': np.array([0., 0., 0., 1.], np.float64),
              'robot0_gripper_qpos': np.array([.02, -.02], np.float64)}
    path = tmp_path / 'init1.npz'
    np.savez(path, **arrays)
    hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}
    manifest = {'version': 1, 'prompt': 'move', 'samples': [{'id': 'init1', 'file': path.name, 'sensor_hashes': hashes}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    _, samples = load_observations(tmp_path)
    assert samples[0]['sensor_hashes'] == hashes
    assert samples[0]['observation']['robot0_eef_pos'].dtype == np.float64
    np.savez(path, **arrays, object_truth=np.array([1.]))
    with pytest.raises(ValueError, match='sensor keys'):
        load_observations(tmp_path)


def test_cross_stream_nested_measurement_fails_instead_of_claiming_exclusive_time():
    """Reject unsupported nested stream overlap before producing misleading partitions.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Exclusive subtraction assumes one stream. Concern: Overlap creates invalid totals.
        Decision: Change fake stream inside root and require an explicit failure.
        Alternatives: Sum stream intervals. Quality Attributes: Honesty. Trade-offs: No concurrent support.
    Evidence: Internal: IntervalRecorder stream guard. External: CUDA stream independence.
        Findings: Unsupported case is rejected. Metrics: Exception. Confidence: Direct mechanics.
    Constraints: Restore root stream before context exit. Reconsider When: Multi-stream accounting added.
    References: measure. ADR: None.
    """
    clock = ManualClock()
    cuda = FakeCUDA(clock)
    recorder = IntervalRecorder(cuda, 'fake', clock)
    with recorder.measure('root'):
        cuda.stream.cuda_stream = 18
        try:
            with pytest.raises(RuntimeError, match='one CUDA stream'):
                with recorder.measure('child'):
                    pass
        finally:
            cuda.stream.cuda_stream = 17
