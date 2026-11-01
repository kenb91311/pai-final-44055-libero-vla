"""Verify resource-monitor failure evidence without probing or allocating a GPU.

Args: pytest isolated files and dependency fixtures. Returns: Assertions.
Raises: AssertionError on missing/error-misclassified samples.
Design: Context: Resource traces support clean benchmark validity.
    Concern: Bad phase metadata must not stop hardware samples; probe failures must not look idle.
    Decision: Exercise the actual monitor main loop with a CPU fake NVML boundary.
    Alternatives: Mock the whole monitor or invoke another live GPU probe rejected.
    Quality Attributes: Evidence integrity, low interference. Trade-offs: No NVML ABI validation.
Evidence: Internal: Valid JSON non-object phase and second-probe exception cases.
    External: None. Findings: Metadata and measurement failures have different outcomes.
    Metrics:2complete samples versus1sample/error. Confidence: Main-loop contract.
Constraints: No torch imports, model loads or real GPU calls.
Reconsider When: Monitor error/status schema changes.
References: experiments/realtime_vla/monitor_resources.py. ADR: None.
"""
import json
import sys
from types import SimpleNamespace

import pytest
from experiments.realtime_vla import monitor_resources


@pytest.fixture
def cpu_monitor(monkeypatch):
    """Provide a deterministic CPU-only NVML substitute and preserve cleanup observability.

    Args: monkeypatch: scoped dependency replacement. Returns: Mutable fixture counters.
    Raises: None during fixture setup.
    Design: Context: Current GPU lease belongs to another experiment.
        Concern: Tests must not initialize CUDA/NVML or change process signal handlers.
        Decision: Replace only hardware/host/signal boundaries, keep actual file/timer loop.
        Alternatives: Load a GPU to test failure paths rejected.
        Quality Attributes: Isolation. Trade-offs: Synthetic counters are not hardware evidence.
    Evidence: Internal: Sample/close call counts. External: None.
        Findings: The monitor's main control flow remains real.
        Metrics: Calls and closed flag. Confidence: Contract-level.
    Constraints: Fixture scope only. Reconsider When: Monitor dependencies change.
    References: test_resource_monitor.py. ADR: None.
    """
    state={'calls':0,'closed':False,'fail_at':None}

    def sample():
        """Return fixture bytes/percent or fail at a declared call without inventing idle data.

        Args: None. Returns: Counter mapping. Raises: RuntimeError on selected call.
        Design: Context: Failed probe scenario. Concern: Hidden substitution.
            Decision: Deterministic failure. Alternatives: Random failure rejected.
            Quality Attributes: Test sensitivity. Trade-offs: No hardware semantics.
        Evidence: Internal: Count assertions. External: None.
            Findings: First sample survives later failure. Metrics: Calls. Confidence: Contract.
        Constraints: CPU only. Reconsider When: Sample schema changes.
        References: cpu_monitor. ADR: None.
        """
        state['calls']+=1
        if state['calls']==state['fail_at']:raise RuntimeError('fixture NVML second-sample failure')
        return {'whole_gpu_total_bytes':1024,'whole_gpu_used_bytes':256,'whole_gpu_free_bytes':768,
                'whole_gpu_utilization_percent':25,'whole_gpu_memory_utilization_percent':5}

    def close():
        """Record owned-probe cleanup even when measurement fails.

        Args: None. Returns: None. Raises: None.
        Design: Context: Finally behavior. Concern: Missing cleanup.
            Decision: Observable flag. Alternatives: None.
            Quality Attributes: Reliability. Trade-offs: No real NVML client.
        Evidence: Internal: Closed assertion. External: None. Findings: None.
            Metrics: Closed flag. Confidence: Contract.
        Constraints: Fixture only. Reconsider When: Lifecycle changes.
        References: cpu_monitor. ADR: None.
        """
        state['closed']=True

    monkeypatch.setattr(monitor_resources,'NvmlProbe',lambda:SimpleNamespace(sample=sample,close=close))
    monkeypatch.setattr(monitor_resources,'host_sample',lambda:{'host_memory_available_bytes':8192,'host_memory_total_bytes':16384,'host_swap_used_bytes':0,'host_load1':0.})
    monkeypatch.setattr(monitor_resources.signal,'signal',lambda *_:None)
    return state


def test_valid_json_nonobject_phase_does_not_interrupt_hardware_samples(tmp_path,monkeypatch,cpu_monitor):
    """Continue two real loop iterations when the phase file contains an empty list.

    Args: tmp_path/monkeypatch/cpu_monitor: isolated dependencies. Returns: None.
    Raises: AssertionError. Design: Context: Phase is descriptive metadata.
        Concern: JSON can parse while lacking dictionary fields.
        Decision: Mark phase_read_failed and preserve all hardware rows.
        Alternatives: Abort or silently invent phase rejected.
        Quality Attributes: Trace continuity. Trade-offs: Phase is unknown.
    Evidence: Internal: phase=[] fixture. External: None.
        Findings: Both nonzero hardware samples remain and status is complete.
        Metrics:2rows/2flags. Confidence: High for main-loop behavior.
    Constraints: No live GPU calls. Reconsider When: Phase schema changes.
    References: monitor_resources.main. ADR: None.
    """
    phase=tmp_path/'phase.json';phase.write_text('[]');output=tmp_path/'trace'
    monkeypatch.setattr(sys,'argv',['monitor_resources.py','--output',str(output),'--phase-file',str(phase),'--period','.1','--samples','2'])
    monitor_resources.main()
    rows=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    summary=json.loads((output/'summary.json').read_text())
    assert len(rows)==2 and all(row['phase_read_failed'] is True for row in rows)
    assert all(row['whole_gpu_used_bytes']==256 for row in rows)
    assert summary['status']=='complete' and summary['samples']==2
    assert cpu_monitor['calls']==2 and cpu_monitor['closed']


def test_second_probe_failure_preserves_first_sample_and_writes_error_summary(tmp_path,monkeypatch,cpu_monitor):
    """Never certify a stopped or failed probe as complete idle-hardware evidence.

    Args: tmp_path/monkeypatch/cpu_monitor: isolated dependencies. Returns: None.
    Raises: AssertionError. Design: Context: Error after valid measurement.
        Concern: Missing terminal evidence could be mistaken for a successful trace.
        Decision: Keep first row, write status:error/samples1, and rethrow the actual failure.
        Alternatives: Append a zero sample or swallow the error rejected.
        Quality Attributes: Auditability. Trade-offs: Caller must handle failed monitor.
    Evidence: Internal: Declared failure at second call. External: None.
        Findings: Exactly one valid sample is retained, cleanup still runs.
        Metrics:1row/error summary. Confidence: High for main-loop behavior.
    Constraints: No live GPU calls. Reconsider When: Retry policy is introduced.
    References: monitor_resources.main. ADR: None.
    """
    cpu_monitor['fail_at']=2;output=tmp_path/'trace'
    monkeypatch.setattr(sys,'argv',['monitor_resources.py','--output',str(output),'--period','.1','--samples','3'])
    with pytest.raises(RuntimeError,match='second-sample failure'):monitor_resources.main()
    rows=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    summary=json.loads((output/'summary.json').read_text())
    assert len(rows)==1 and rows[0]['whole_gpu_used_bytes']==256
    assert summary['status']=='error' and summary['samples']==1 and 'second-sample failure' in summary['error']
    assert cpu_monitor['calls']==2 and cpu_monitor['closed']
