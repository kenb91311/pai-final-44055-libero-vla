"""Render unchanged LIBERO cameras in a separate latest-snapshot process.

Args: Same scene settings as FrozenLiberoEnv and a sensor-rate cap.
Returns: Raw synchronized image/proprio observations retaining physical capture time.
Raises: RuntimeError for renderer startup/failure; invalid snapshots propagate errors.
Design:
    Context: WSL two-camera rendering exceeded the50ms physics budget.
    Concern: Inference-only asynchrony cannot keep physical time when cameras block env.step.
    Decision: No-camera physics plus a canonical scene clone consuming bounded latest snapshots.
    Alternatives: Lower camera resolution would change pretrained visual inputs.
    Quality Attributes: Timing isolation, compatibility. Trade-offs: Scene clone and sensor delay.
Evidence:
    Internal: environment-smoke hardware-render-results reports85ms20Hz camera steps.
    External: LIBERO regenerate_obs_from_state restores official sensors without physics steps.
    Findings: Capture time refers to the source physical state, not image receipt.
    Metrics: Rendering latency, sensor age, submission drops and physical step budget.
    Confidence: Confirmed by paired snapshot and concurrent-physics smoke.
Constraints: No privileged object truth passed to policies; one pending snapshot/result.
Reconsider When: Native accelerated rendering fits the control budget.
References: libero_env.py; parent nonblocking realtime runner.
ADR: None.
"""
from __future__ import annotations
import multiprocessing as mp
from queue import Empty,Full
import time
import traceback


def _put_latest(channel,value):
    """Replace one pending mailbox item without blocking the physical caller.

    Args: channel: bounded multiprocessing queue; value: new message.
    Returns: True when enqueued, false during a concurrent feeder race.
    Raises: Broken queue errors propagate.
    Design:
        Context: Producer can run faster than the sensor consumer.
        Concern: Unbounded stale snapshots inflate observation age.
        Decision: Evict available pending content; never wait on a full mailbox.
        Alternatives: Blocking put would reintroduce a camera dependency in physics.
        Quality Attributes: Bounded latency/memory. Trade-offs: Rare newest sample may drop.
    Evidence:
        Internal: maxsize1 queues; submit reports acceptance/drops.
        External: multiprocessing.Queue asynchronous feeder semantics.
        Findings: A full-but-not-yet-readable feeder must not cause a wait.
        Metrics: submit accepted versus dropped. Confidence: Unit/integration tests.
    Constraints: At most one pending item, excluding the sample being processed.
    Reconsider When: Shared-memory snapshot buffers replace queues.
    References: CameraWorker.submit.
    ADR: None.
    """
    try:
        channel.put_nowait(value);return True
    except Full:
        try:channel.get_nowait()
        except Empty:return False
        try:channel.put_nowait(value);return True
        except Full:return False


def _render_process(requests,results,settings,init_index,seed,camera_hz):
    """Own the official graphics scene and emit only sensors from the newest snapshot.

    Args: requests/results: bounded mailboxes; settings: scene kwargs; init_index/seed;
        camera_hz: maximum rendering frequency.
    Returns: None after shutdown. Raises: Errors are serialized to the owner.
    Design:
        Context: OpenGL contexts must remain in their owning process.
        Concern: Render waits and GPU contention must not pause main-loop physics.
        Decision: Spawn one private renderer, preserve source timestamps, drain pending snapshots.
        Alternatives: Forking an existing CUDA/OpenGL process is unsupported.
        Quality Attributes: Isolation. Trade-offs: Explicit startup before task timing.
    Evidence:
        Internal: Physics state is copied through a bounded mailbox.
        External: LIBERO official state restoration and multiprocessing spawn.
        Findings: Force-updating all observables synchronizes RGB and proprioception.
        Metrics: Render start/finish, source capture time/tick.
        Confidence: Concurrent physical-step integration evidence.
    Constraints: Renderer never computes actions or advances simulated time.
    Reconsider When: A renderer supports direct shared-memory model-state views.
    References: FrozenLiberoEnv.observe_snapshot.
    ADR: None.
    """
    env=None
    try:
        if __package__:
            from .libero_env import FrozenLiberoEnv
        else:
            from libero_env import FrozenLiberoEnv
        from threadpoolctl import threadpool_limits
        limiter=threadpool_limits(limits=1)
        settings=dict(settings,render_mode='sync',output_dir=None)
        env=FrozenLiberoEnv(**settings);env.reset(init_state_index=init_index,seed=seed)
        results.put({'kind':'ready','metadata':env.metadata})
        next_allowed=0.
        while True:
            snapshot=requests.get()
            if snapshot is None:break
            # During rate limiting, keep replacing the pending sample with fresher state.
            while time.monotonic()<next_allowed:
                try:new=requests.get(timeout=max(.0001,next_allowed-time.monotonic()))
                except Empty:break
                if new is None:return
                snapshot=new
            while True:
                try:new=requests.get_nowait()
                except Empty:break
                if new is None:return
                snapshot=new
            start=time.monotonic()
            observation=env.observe_snapshot(snapshot)
            _put_latest(results,{'kind':'observation','observation':observation})
            next_allowed=start+1./camera_hz
        limiter.restore_original_limits()
    except BaseException as exc:
        _put_latest(results,{'kind':'error','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
    finally:
        if env is not None:env.close()


class CameraWorker:
    """Expose nonblocking submit/poll around an independent canonical sensor scene.

    Args: env_kwargs: matching physics scene; init_index/seed: startup reset;
        camera_hz: rate cap; startup_timeout: untimed readiness budget.
    Returns: Worker instance. Raises: RuntimeError for failed startup/rendering.
    Design:
        Context: Main control loop must continue while cameras and VLA compute.
        Concern: Camera latency must remain visible as observation age.
        Decision: Return original capture_time/tick and separately recorded render completion.
        Alternatives: Relabeling old frames as current gives misleading latency metrics.
        Quality Attributes: Observability. Trade-offs: Latest sensing can lag physics.
    Evidence:
        Internal: Same assets/task/initstate SHA as physical environment.
        External: Official camera rendering path preserved.
        Findings: Constructor readiness is excluded from timed episodes.
        Metrics: Submit/drop counts, render time, sensor age.
        Confidence: Integration smoke and unit-tested mailbox semantics.
    Constraints: Call wait only before task time; poll is always nonblocking.
    Reconsider When: Multiple sensor streams need independent timestamps.
    References: Module contract.
    ADR: None.
    """
    def __init__(self,env_kwargs=None,init_index=0,seed=0,camera_hz=10,startup_timeout=180):
        """Spawn and await a ready renderer before the experiment starts.

        Args: See class. Returns: None. Raises: ValueError/RuntimeError.
        Design: Context: Graphics startup is slow. Concern: Task-time initialization.
            Decision: Explicit ready handshake. Alternatives: Lazy scene load.
            Quality Attributes: Predictability. Trade-offs: Untimed startup wait.
        Evidence: Internal: Renderer metadata returned at readiness.
            External: Spawn context. Findings: No CUDA state is forked.
            Metrics: Startup duration. Confidence: Integration smoke.
        Constraints: Positive camera rate and finite startup budget.
        Reconsider When: A persistent renderer service is shared.
        References: CameraWorker. ADR: None.
        """
        if camera_hz<=0 or startup_timeout<=0:raise ValueError('positive camera_hz/startup_timeout required')
        context=mp.get_context('spawn');self.requests=context.Queue(maxsize=1);self.results=context.Queue(maxsize=1)
        self.accepted=0;self.dropped=0;self.closed=False
        self.process=context.Process(target=_render_process,args=(self.requests,self.results,dict(env_kwargs or {}),init_index,seed,float(camera_hz)),daemon=True)
        start=time.monotonic();self.process.start()
        try:message=self.results.get(timeout=startup_timeout)
        except Empty:
            self.close();raise RuntimeError('renderer startup timeout') from None
        if message.get('kind')!='ready':
            self.close();raise RuntimeError(f"renderer startup failed: {message}")
        self.metadata=message['metadata'];self.startup_seconds=time.monotonic()-start

    def submit(self,snapshot):
        """Offer the newest physical snapshot without waiting for rendering.

        Args: snapshot: FrozenLiberoEnv.snapshot mapping. Returns: Accepted bool.
        Raises: RuntimeError if worker is closed/dead.
        Design: Context: Physics producer owns the deadline.
            Concern: Queue pressure must not suspend env.step scheduling.
            Decision: Nonblocking replacement in a maxsize1 mailbox.
            Alternatives: FIFO would replay stale states.
            Quality Attributes: Timing isolation. Trade-offs: Samples are intentionally dropped.
        Evidence: Internal: Accepted/dropped counters.
            External: None. Findings: Consumer sees the latest available snapshot.
            Metrics: Drops and capture ages. Confidence: Mailbox tests.
        Constraints: Snapshot arrays must not be mutated after submission.
        Reconsider When: Zero-copy buffers are introduced.
        References: _put_latest. ADR: None.
        """
        if self.closed or not self.process.is_alive():raise RuntimeError('renderer is not running')
        accepted=_put_latest(self.requests,snapshot)
        self.accepted+=int(accepted);self.dropped+=int(not accepted)
        return accepted

    def poll(self):
        """Return the latest completed sensor observation immediately, or None.

        Args: None. Returns: Raw observation or None. Raises: RuntimeError on failure.
        Design: Context: Main loop polls once per physics boundary.
            Concern: Waiting for a frame would break the camera/physics separation.
            Decision: Drain only currently available messages.
            Alternatives: Blocking receive is restricted to startup wait().
            Quality Attributes: Nonblocking operation. Trade-offs: Consumer retains older frame.
        Evidence: Internal: Queue.get_nowait only.
            External: None. Findings: capture_time remains the source timestamp.
            Metrics: Sensor age. Confidence: Integration smoke.
        Constraints: Renderer errors are explicit.
        Reconsider When: Callback-driven integration is needed.
        References: CameraWorker.wait. ADR: None.
        """
        latest=None
        while True:
            try:message=self.results.get_nowait()
            except Empty:break
            if message.get('kind')=='error':raise RuntimeError(f"renderer failed: {message['error']}\\n{message.get('traceback','')}")
            if message.get('kind')=='observation':latest=message['observation']
        if latest is None and not self.closed and not self.process.is_alive():raise RuntimeError('renderer exited without a sensor result')
        return latest

    def wait(self,timeout=30):
        """Wait for the first sensor only during untimed startup.

        Args: timeout: seconds. Returns: Raw observation. Raises: RuntimeError.
        Design: Context: Bootstrap inference needs an initial RGB frame.
            Concern: Main timed loop must never call a blocking sensor method.
            Decision: Separate explicit startup-only API.
            Alternatives: Blocking poll obscures timing behavior.
            Quality Attributes: Clear timing contract. Trade-offs: Caller must respect lifecycle.
        Evidence: Internal: Root records startup separately.
            External: None. Findings: Subsequent calls use poll exclusively.
            Metrics: Bootstrap sensor delay. Confidence: Integration smoke.
        Constraints: Forbidden inside timed physical execution.
        Reconsider When: A preloaded sensor is supplied.
        References: CameraWorker.poll. ADR: None.
        """
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            value=self.poll()
            if value is not None:return value
            time.sleep(.002)
        raise RuntimeError('initial camera observation timeout')

    def close(self):
        """Stop the private renderer and release its bounded mailboxes.

        Args: None. Returns: None. Raises: None during normal cleanup.
        Design: Context: Evaluations reuse one GPU serially.
            Concern: Lingering graphics contexts consume GPU memory.
            Decision: Graceful sentinel then bounded process termination.
            Alternatives: Waiting forever on a stalled renderer.
            Quality Attributes: Resource hygiene. Trade-offs: Forced termination on timeout.
        Evidence: Internal: Owned process and queues.
            External: multiprocessing lifecycle.
            Findings: Cleanup is idempotent. Metrics: Exit state. Confidence: Integration smoke.
        Constraints: No further submit after close.
        Reconsider When: Shared renderer lifetimes are introduced.
        References: CameraWorker. ADR: None.
        """
        if self.closed:return
        self.closed=True
        _put_latest(self.requests,None)
        self.process.join(timeout=3)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=3)
        for channel in (self.requests,self.results):
            channel.cancel_join_thread();channel.close()
