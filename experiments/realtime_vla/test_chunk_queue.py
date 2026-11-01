"""Check common timing rules without substituting synthetic data for model evaluation.

Args: pytest. Returns: Assertions. Raises: AssertionError on queue contract violations.
Design:
    Context: Real-time model comparisons share one scheduler.
    Concern: Hold and late-chunk bugs can masquerade as model quality.
    Decision: Deterministic timing examples isolate queue mechanics.
    Alternatives: Full GPU rollouts are too expensive for boundary cases.
    Quality Attributes: Correctness. Trade-offs: No learned policy competence measured.
Evidence:
    Internal: ChunkQueue contract. External: Relative control semantics.
    Findings: Empty queues must not repeat translation.
    Metrics: Exact actions and expired count. Confidence: Contract-level.
Constraints: Unit tests only; not task-success evidence.
Reconsider When: Merge semantics change.
References: chunk_queue.py.
ADR: None.
"""
import numpy as np
import pytest
from chunk_queue import ChunkQueue


def test_empty_holds_pose_and_gripper():
    """Hold pose instead of repeating an executed relative move.

    Args: None. Returns: None. Raises: AssertionError.
    Design:
        Context: Inference can be late. Concern: Repeating increments causes drift.
        Decision: Verify zero delta and preserved gripper.
        Alternatives: Last-action repeat rejected. Quality Attributes: Reliability.
        Trade-offs: Object dynamics still evolve during hold.
    Evidence:
        Internal: One relative X command. External: None. Findings: Next six coordinates zero.
        Metrics: Equality. Confidence: High.
    Constraints: Relative action mode.
    Reconsider When: Absolute targets are used.
    References: ChunkQueue.pop.
    ADR: None.
    """
    q=ChunkQueue(1);q.accept([[.5,0,0,0,0,0,1]],0,1.,0,'q0')
    assert q.pop()['action'][0]==.5
    held=q.pop();assert held['held'];np.testing.assert_array_equal(held['action'],[0,0,0,0,0,0,1])


def test_fifo_preserves_origin_and_prefix():
    """Limit native long chunks to a shared executed prefix and retain source age.

    Args: None. Returns: None. Raises: AssertionError.
    Design:
        Context: Models generate different horizons. Concern: Hidden long prefixes bias comparison.
        Decision: Both consume only configured two actions.
        Alternatives: Native lengths tested separately. Quality Attributes: Fairness.
        Trade-offs: Remaining generated actions discarded.
    Evidence:
        Internal: Four-action input. External: None. Findings: FIFO source tick remains zero.
        Metrics: Depth and identity. Confidence: High.
    Constraints: One in-flight request.
    Reconsider When: Native-horizon condition is added.
    References: ChunkQueue.accept.
    ADR: None.
    """
    q=ChunkQueue(2);q.accept(np.zeros((4,7)),0,3.,2,'q0')
    assert len(q)==2
    assert q.pop()['observation_tick']==0
    q.accept(np.ones((4,7)),3,4.,4,'q1');assert len(q)==3
    assert q.pop()['request_id']=='q0';assert q.pop()['request_id']=='q1'


def test_aligned_expiration_is_not_retimed():
    """Discard an entirely expired prefix rather than relabel it as current.

    Args: None. Returns: None. Raises: AssertionError.
    Design:
        Context: Large inference delay may exceed action horizon.
        Concern: False freshness hides missed deadlines.
        Decision: No action survives beyond prefix.
        Alternatives: FIFO is a separate explicit condition.
        Quality Attributes: Integrity. Trade-offs: Expired chunk can stall the task.
    Evidence:
        Internal: Three-tick delay for two-action chunk.
        External: None. Findings: Both actions expire.
        Metrics: Expired count. Confidence: High.
    Constraints: Logical control ticks define alignment.
    Reconsider When: Model-conditioned RTC is used.
    References: ChunkQueue.accept.
    ADR: None.
    """
    q=ChunkQueue(2,'aligned');assert q.accept(np.ones((2,7)),0,0.,3,'late')==0
    assert q.discarded_expired==2 and q.pop()['held']


@pytest.mark.parametrize('bad',[np.zeros((3,8)),np.full((3,7),np.nan),np.zeros(7)])
def test_invalid_chunk_rejected(bad):
    """Reject malformed chunks before actuator access.

    Args: bad: invalid actions. Returns: None. Raises: AssertionError.
    Design:
        Context: Numeric model errors must stop safely. Concern: NaN reaches physics.
        Decision: Explicit shape and finite check. Alternatives: Silent coercion hides failure.
        Quality Attributes: Reliability. Trade-offs: Episode aborts.
    Evidence:
        Internal: Bad shapes and NaN. External: None. Findings: All raise.
        Metrics: Rejection. Confidence: High.
    Constraints: Seven actuator-controller coordinates.
    Reconsider When: Action shape changes.
    References: ChunkQueue.accept.
    ADR: None.
    """
    with pytest.raises(ValueError):ChunkQueue().accept(bad,0,0.,0,'bad')
