"""Check worker sensor/prefix messaging without loading CUDA models or contacting ROS.

Args: pytest. Returns: Protocol assertions. Raises: AssertionError on schema regressions.
Design: Context: Real GPU smoke is separately recorded.
    Concern: Evaluator truth or the wrong action space could cross IPC unnoticed.
    Decision: Exercise the actual request builder and process loop with explicit dependency fakes.
    Alternatives: Claim mocked transport proves DDS delivery rejected.
    Quality Attributes: Interface correctness. Trade-offs: No timing or ROS performance evidence.
Evidence: Internal: ContinuityWorker schema. External: None.
    Findings: Sensor allowlist and checkpoint normalization are independent boundaries.
    Metrics: Message fields and exact arrays. Confidence: High for the exercised protocol.
Constraints: CPU only; synthetic values are never task outcomes.
Reconsider When: Worker protocol changes. References: continuity_worker.py. ADR: None.
"""
import importlib
from pathlib import Path
import queue
import sys
from types import SimpleNamespace

import numpy as np
import torch


def test_sensor_allowlist_excludes_evaluator_truth_and_copies_received_inputs(monkeypatch):
    """Verify the real predict request carries only cameras/proprioception and action prefix.

    Args: monkeypatch: scoped import-path fixture. Returns: None. Raises: AssertionError.
    Design: Context: Env observation includes ground-truth metrics.
        Concern: Forwarding whole observations leaks target positions.
        Decision: Inspect request emitted by real worker.predict.
        Alternatives: Trust docstrings only rejected. Quality Attributes: Causal integrity.
        Trade-offs: Worker process is not spawned.
    Evidence: Internal: Sentinel privileged keys. External: Five-sensor model contract.
        Findings: Sensing arrays are copied. Metrics: Exact allowlist. Confidence: High.
    Constraints: No DDS claim. Reconsider When: New sensor modalities are approved.
    References: ContinuityWorker.predict. ADR: None.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/realtime_vla'))
    module=importlib.import_module('continuity_worker')
    worker=module.ContinuityWorker.__new__(module.ContinuityWorker)
    worker.input=queue.Queue(maxsize=1);worker.counter=0;worker.inflight=False;worker.prefix='cpu-schema'
    obs={k:np.arange(3,dtype=np.float32) for k in module.SENSORS}
    obs.update(prompt='move bowl',metrics={'bowl_position':[99,99,99]},success=True,object_state='hidden')
    prefix=np.zeros((12,7),dtype=np.float32)
    request=worker.predict(obs,7,1.25,prefix_actions=prefix,inference_delay=13,execution_horizon=12,request_tick=10)
    sent=worker.input.get_nowait()
    assert set(sent['observation'])==set(module.SENSORS)
    assert not {'metrics','success','object_state'} & set(sent)
    assert (sent['observation_tick'],sent['request_tick'],sent['inference_delay'],sent['execution_horizon'])==(7,10,13,12)
    assert request==sent['request_id'] and worker.inflight
    obs[module.SENSORS[0]][0]=999
    assert sent['observation'][module.SENSORS[0]][0]==0


def test_process_normalizes_physical_prefix_and_returns_paired_native_chunks(monkeypatch):
    """Run the actual process loop with deterministic fake model and transport boundaries.

    Args: monkeypatch: scoped imports/functions. Returns: None. Raises: AssertionError.
    Design: Context: Prefix is supplied as physical actuator values.
        Concern: Passing those values directly as RTC latents changes conditioning.
        Decision: Verify normalization occurs before model predict and paired chunks survive output.
        Alternatives: Hardware-bound test for pure message contract rejected.
        Quality Attributes: Compatibility. Trade-offs: Transport hashes here are fake sentinels.
    Evidence: Internal: Add42 fake normalization distinguishes spaces.
        External: None. Findings: Runtime gets normalized prefix and unchanged tick metadata.
        Metrics: Exact normalized/physical values. Confidence: High for message plumbing.
    Constraints: This is not real ROS/GPU/task evidence.
    Reconsider When: Worker supports multiple runtime adapters.
    References: continuity_process. ADR: None.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/realtime_vla'))
    module=importlib.import_module('continuity_worker');calls={}

    class FakeRuntime:
        """Provide a CPU-only protocol fixture, not a replacement learned policy.

        Args: Constructor receives original family choices. Returns: Fixture.
        Raises: Assertions on malformed protocol.
        Design: Context: Interface test. Concern: Accidental GPU load.
            Decision: Explicit fake. Alternatives: Load real weights rejected.
            Quality Attributes: Test isolation. Trade-offs: No model validity evidence.
        Evidence: Internal: Sentinels. External: None. Findings: Spaces remain distinct.
            Metrics: Equality. Confidence: Contract only.
        Constraints: Test-local. Reconsider When: API changes.
        References: test_process_normalizes_physical_prefix_and_returns_paired_native_chunks. ADR: None.
        """
        def __init__(self,*args,**kwargs):
            """Record construction without allocating a model.

            Args: args/kwargs: worker configuration. Returns: None. Raises: None.
            Design: Context: Fake. Concern: Side effects. Decision: Record only.
                Alternatives: Real model. Quality Attributes: Isolation. Trade-offs: No timing.
            Evidence: Internal: Call log. External: None. Findings: None.
                Metrics: None. Confidence: Contract.
            Constraints: CPU. Reconsider When: API changes. References: FakeRuntime. ADR: None.
            """
            calls['constructor']=(args,kwargs)
        def reset(self,seed):
            """Record the untimed reset request.

            Args: seed. Returns: None. Raises: None.
            Design: Context: Fixture. Concern: None. Decision: Record.
                Alternatives: None. Quality Attributes: Observability. Trade-offs: No RNG model.
            Evidence: Internal: Call log. External: None. Findings: None.
                Metrics: Seed. Confidence: Contract.
            Constraints: CPU. Reconsider When: API changes. References: FakeRuntime. ADR: None.
            """
            calls['seed']=seed
        def normalize_actions(self,actions):
            """Transform physical prefix with a distinguishable fixture offset.

            Args: actions[T,7]. Returns: Actions+42. Raises: None.
            Design: Context: Space boundary. Concern: Bypass. Decision: Sentinel transform.
                Alternatives: Identity would miss bypass. Quality Attributes: Test sensitivity.
                Trade-offs: Not checkpoint statistics.
            Evidence: Internal: Exact output check. External: None. Findings: Spaces differ.
                Metrics:42offset. Confidence: Contract.
            Constraints: Synthetic only. Reconsider When: API changes. References: FakeRuntime. ADR: None.
            """
            return np.asarray(actions)+42
        def predict(self,sensors,prompt,**kwargs):
            """Capture normalized RTC inputs and emit two known native chunks.

            Args: sensors/prompt/kwargs. Returns: H50 action/normalized pair. Raises: None.
            Design: Context: Worker plumbing. Concern: Missing metadata.
                Decision: Capture exact input. Alternatives: None.
                Quality Attributes: Inspectability. Trade-offs: No learned behavior.
            Evidence: Internal: Paired sentinels. External: None. Findings: None.
                Metrics: Shape. Confidence: Contract.
            Constraints: CPU. Reconsider When: API changes. References: FakeRuntime. ADR: None.
            """
            calls['predict']=(sensors,prompt,kwargs)
            return {'actions':np.full((50,7),.25,dtype=np.float32),'normalized':np.full((50,7),-2,dtype=np.float32),'metadata':{'variant':'rtc-vjp'}}
        def audit(self):
            """Return an explicit fake audit sentinel.

            Args: None. Returns: Fake audit. Raises: None.
            Design: Context: Protocol. Concern: None. Decision: Sentinel.
                Alternatives: None. Quality Attributes: Test isolation. Trade-offs: No hash evidence.
            Evidence: Internal: Assertions. External: None. Findings: None.
                Metrics: None. Confidence: Contract.
            Constraints: Not real audit. Reconsider When: API changes. References: FakeRuntime. ADR: None.
            """
            return {'fixture':True}
        def close(self):
            """Record clean runtime closure on the owned process boundary.

            Args: None. Returns: None. Raises: None.
            Design: Context: Finally cleanup. Concern: Leak. Decision: Record close.
                Alternatives: None. Quality Attributes: Resource hygiene. Trade-offs: No real tensors.
            Evidence: Internal: Call log. External: None. Findings: None.
                Metrics: Closed flag. Confidence: Contract.
            Constraints: Test only. Reconsider When: API changes. References: FakeRuntime. ADR: None.
            """
            calls['closed']=True

    monkeypatch.setitem(sys.modules,'continuity_runtime',SimpleNamespace(FrozenContinuityPolicy=FakeRuntime))
    monkeypatch.setitem(sys.modules,'ros_client',SimpleNamespace(
        sensor_roundtrip=lambda sensors,prompt,*a,**k:(sensors,prompt,{'fixture':'sensors'}),
        chunk_roundtrip=lambda actions,*a,**k:(np.asarray(actions).copy(),{'fixture':'actions'})))
    monkeypatch.setattr(torch,'set_num_threads',lambda _:None)
    incoming=queue.Queue();outgoing=queue.Queue()
    incoming.put({'op':'reset','request_id':'reset','seed':17})
    prefix=np.zeros((5,7),dtype=np.float32)
    incoming.put({'op':'predict','request_id':'query','observation':{'raw':'fixture'},'prompt':'bowl',
        'prefix_actions':prefix,'inference_delay':4,'raw_predicted_delay':4,'execution_horizon':5,'request_tick':12,'observation_tick':9,'observation_time':1.25})
    incoming.put({'op':'audit','request_id':'audit'});incoming.put({'op':'close'})
    module.continuity_process('smolvla','unused','rtc-vjp',incoming,outgoing,8780)
    output=[outgoing.get_nowait() for _ in range(outgoing.qsize())]
    assert [x['kind'] for x in output]==['ready','reset','chunk','audit']
    predicted=calls['predict'];np.testing.assert_array_equal(predicted[2]['prev_normalized'],prefix+42)
    assert predicted[2]['inference_delay']==4 and predicted[2]['execution_horizon']==5
    assert output[2]['actions'].shape==output[2]['normalized'].shape==(50,7)
    assert (output[2]['observation_tick'],output[2]['request_tick'],output[2]['observation_time'])==(9,12,1.25)
    assert calls['seed']==17 and calls['closed']
