"""Check bounded sensor mailboxes and source-state identity without GPU initialization.

Args: Pytest fixtures. Returns: Assertions. Raises: Test failures.
Design: Context: Physical-loop operations must remain nonblocking.
    Concern: Queue feeder races and episode reuse.
    Decision: Test dropping/replacement and timestamp/episode preservation.
    Alternatives: Only testing successful rendering misses timing edge cases.
    Quality Attributes: Integrity. Trade-offs: Actual cameras have a separate integration smoke.
Evidence: Internal: CameraWorker mailbox/snapshot contracts. External: None.
    Findings: Source timestamps and identities are independent of completion.
    Metrics: Unit assertions. Confidence: Deterministic coverage.
Constraints: No graphics or model weights required.
Reconsider When: IPC protocol changes.
References: camera_worker.py, libero_env.py. ADR: None.
"""
from queue import Queue,Full,Empty
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import time
import numpy as np
camera_spec=importlib.util.spec_from_file_location('camera_worker',Path(__file__).with_name('camera_worker.py'))
camera_module=importlib.util.module_from_spec(camera_spec);camera_spec.loader.exec_module(camera_module)
env_spec=importlib.util.spec_from_file_location('libero_env',Path(__file__).with_name('libero_env.py'))
env_module=importlib.util.module_from_spec(env_spec);env_spec.loader.exec_module(env_module)
_put_latest=camera_module._put_latest
FrozenLiberoEnv=env_module.FrozenLiberoEnv


def test_latest_mailbox_replaces_pending_sample():
    """Keep one newest sample while guaranteeing immediate producer return.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Physics outruns sensors. Concern: Stale FIFO backlog.
        Decision: Verify maxsize1 replacement. Alternatives: Unbounded queue.
        Quality Attributes: Bounded age. Trade-offs: Old samples dropped.
    Evidence: Internal: _put_latest. External: None. Findings: Latest retained.
        Metrics: Queue size1. Confidence: Deterministic.
    Constraints: No thread sleeps. Reconsider When: IPC changes.
    References: CameraWorker.submit. ADR: None.
    """
    channel=Queue(maxsize=1)
    assert _put_latest(channel,{'tick':1})
    assert _put_latest(channel,{'tick':2})
    assert channel.qsize()==1 and channel.get_nowait()=={'tick':2}


class FeederRace:
    """Represent a full multiprocessing slot whose feeder has not published bytes.

    Args: None. Returns: Fake queue. Raises: Full/Empty as documented.
    Design: Context: Queue feeder is asynchronous. Concern: Accidental blocking fallback.
        Decision: Force the full/unreadable race. Alternatives: Timing-sensitive tests.
        Quality Attributes: Determinism. Trade-offs: Mechanical fake only.
    Evidence: Internal: Python queue exception contract. External: None.
        Findings: New submission must drop. Metrics: False result. Confidence: Unit coverage.
    Constraints: No actual threads. Reconsider When: Shared memory replaces queues.
    References: _put_latest. ADR: None.
    """
    def put_nowait(self,value):
        """Report a reserved slot before feeder visibility.

        Args: value: ignored. Returns: Never. Raises: Full.
        Design: Context: Race test. Concern: Blocking. Decision: Always full.
            Alternatives: None. Quality Attributes: Determinism. Trade-offs: Fake.
        Evidence: Internal: Test fixture. External: None. Findings: Full.
            Metrics: Exception. Confidence: Contract.
        Constraints: Test only. Reconsider When: IPC changes.
        References: FeederRace. ADR: None.
        """
        raise Full

    def get_nowait(self):
        """Report bytes not yet available despite the reserved slot.

        Args: None. Returns: Never. Raises: Empty.
        Design: Context: Race test. Concern: Waiting. Decision: Always empty.
            Alternatives: None. Quality Attributes: Determinism. Trade-offs: Fake.
        Evidence: Internal: Test fixture. External: None. Findings: Empty.
            Metrics: Exception. Confidence: Contract.
        Constraints: Test only. Reconsider When: IPC changes.
        References: FeederRace. ADR: None.
        """
        raise Empty


def test_feeder_race_drops_instead_of_waiting():
    """Return false on feeder races rather than blocking physical control.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Maxsize1 queue. Concern: Stalled control deadline.
        Decision: Explicit drop. Alternatives: Retrying blocking get.
        Quality Attributes: Availability. Trade-offs: Sample loss.
    Evidence: Internal: FeederRace. External: None. Findings: No wait path.
        Metrics: False. Confidence: Deterministic.
    Constraints: No sleep. Reconsider When: IPC changes.
    References: _put_latest. ADR: None.
    """
    assert _put_latest(FeederRace(),{'tick':2}) is False


def test_snapshot_copies_state_and_preserves_episode():
    """Separate physical capture time, control tick and episode from rendering completion.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Cross-episode in-flight frames can arrive late.
        Concern: Incorrect source identity or mutation.
        Decision: Copy state and retain explicit reset identity.
        Alternatives: Relabel old frames at receipt.
        Quality Attributes: Integrity. Trade-offs: Small copy.
    Evidence: Internal: FrozenLiberoEnv.snapshot. External: None.
        Findings: Source snapshot remains immutable after physical state changes.
        Metrics: Capture timestamp bounds. Confidence: Unit coverage.
    Constraints: No actual simulator required. Reconsider When: Snapshot schema changes.
    References: CameraWorker.poll. ADR: None.
    """
    state=np.array([.1,.2,.3]);owner=object.__new__(FrozenLiberoEnv)
    owner.step_count=3;owner.env=SimpleNamespace(get_sim_state=lambda:state,sim=SimpleNamespace(data=SimpleNamespace(time=.15)))
    before=time.monotonic();snapshot=owner.snapshot(episode_id='episode-two');after=time.monotonic()
    state[0]=42
    assert snapshot['physics_state'][0]==.1
    assert snapshot['episode_id']=='episode-two' and snapshot['tick']==3
    assert before<=snapshot['capture_time']<=after
    assert snapshot['simulation_time']==.15
