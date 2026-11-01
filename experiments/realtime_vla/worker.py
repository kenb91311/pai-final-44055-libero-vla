"""Run one frozen GPU policy in an isolated asynchronous inference process.

Args: Model family, local verified artifact path and sensor-only messages.
Returns: Full action chunks and timing/audit results through bounded local queues.
Raises: RuntimeError on child initialization, prediction or communication failure.
Design:
    Context: The physics/control clock must not wait for chunk generation.
    Concern: Synchronous inference would reproduce the prior lockstep experiment.
    Decision: Spawn a separate model process and allow one in-flight inference.
    Alternatives: A single control thread blocks physical progress.
    Quality Attributes: Isolation, observability. Trade-offs: IPC copies two RGB frames.
Evidence:
    Internal: Control loop polls output without waiting.
    External: LeRobot predict_action_chunk API.
    Findings: Each family uses the same worker protocol.
    Metrics: Prediction duration and sensor-to-ready age. Confidence: Runtime measured.
Constraints: Only one model is resident; evaluator truth never crosses this boundary.
Reconsider When: ROS transport is selected instead of local process IPC.
References: docs/architecture/realtime-vla.md.
ADR: None.
"""
import multiprocessing as mp
import queue
import time
import traceback
import uuid
import numpy as np

SENSORS=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def model_process(family,modeldir,incoming,outgoing,ros_transport=False,ros_port=8780):
    """Own GPU initialization and execute serialized chunk requests.

    Args: family/modeldir: frozen SmolVLA, pi0-FAST or pi0.5 artifact; incoming/outgoing: bounded queues.
    Returns: None on shutdown. Raises: Errors are sent as explicit results.
    Design:
        Context: Forking after graphics initialization can corrupt CUDA state.
        Concern: Model failures must not silently become zero actions.
        Decision: Spawn, lazy model imports, explicit ready/error messages.
        Alternatives: Threads share interpreter and model lifecycle.
        Quality Attributes: Reliability. Trade-offs: Startup cost is measured separately.
    Evidence:
        Internal: Family adapters validate official preprocessing and decoding.
        External: Python multiprocessing spawn semantics.
        Findings: Reset and audit occur outside timed control.
        Metrics: Ready time, chunk latency, weights. Confidence: Measured.
    Constraints: No optimizer and no task-state input.
    Reconsider When: Multiple model servers are benchmarked.
    References: Family runtime modules.
    ADR: None.
    """
    runtime=None
    try:
        import torch
        torch.set_num_threads(4)
        start=time.perf_counter()
        if family=='smolvla':
            from smolvla_runtime import FrozenSmolVLA
            runtime=FrozenSmolVLA(modeldir,device='cuda')
        elif family=='pi0fast':
            from pi0fast_runtime import FrozenPI0Fast
            runtime=FrozenPI0Fast(modeldir,device='cuda')
        elif family=='pi05':
            from pi05_runtime import FrozenPI05
            runtime=FrozenPI05(modeldir,device='cuda')
        else:raise ValueError('unknown model family')
        outgoing.put({'kind':'ready','startup_seconds':time.perf_counter()-start})
        while True:
            message=incoming.get();op=message['op'];request_id=message.get('request_id')
            if op=='close':break
            if op=='reset':
                runtime.reset(message['seed']);outgoing.put({'kind':'reset','request_id':request_id});continue
            if op=='audit':
                outgoing.put({'kind':'audit','request_id':request_id,'audit':runtime.audit()});continue
            if op!='predict':raise ValueError('unknown worker operation')
            start=time.monotonic()
            sensors=message['observation'];prompt=message['prompt'];sensor_transport=None;action_transport=None
            if ros_transport:
                from ros_client import sensor_roundtrip,chunk_roundtrip
                sensors,prompt,sensor_transport=sensor_roundtrip(sensors,prompt,request_id,port=ros_port)
            actions,meta=runtime.predict(sensors,prompt)
            if ros_transport:
                actions,action_transport=chunk_roundtrip(actions,request_id,port=ros_port)
            meta=dict(meta,ros_transport=bool(ros_transport),sensor_transport=sensor_transport,action_transport=action_transport)
            actions=np.asarray(actions,dtype=np.float64)
            if actions.ndim!=2 or actions.shape[1]!=7 or not np.isfinite(actions).all():raise ValueError('invalid generated action chunk')
            outgoing.put({'kind':'chunk','request_id':request_id,'actions':actions,'metadata':meta,
                          'started_time':start,'finished_time':time.monotonic(),
                          'observation_tick':message['observation_tick'],'observation_time':message['observation_time']})
    except BaseException as error:
        outgoing.put({'kind':'error','error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()})
    finally:
        if runtime is not None and hasattr(runtime,'close'):runtime.close()


class ModelWorker:
    """Expose one nonblocking chunk request at a time to the control loop.

    Args: family, modeldir, startup_timeout: child initialization contract.
    Returns: Worker instance. Raises: RuntimeError/TimeoutError on startup failure.
    Design:
        Context: Model loading is outside physical task timing.
        Concern: Unbounded pending requests make observation age uninterpretable.
        Decision: Queue size one and caller-managed in-flight flag.
        Alternatives: Unlimited inference backlog amplifies stale commands.
        Quality Attributes: Boundedness. Trade-offs: No speculative parallel chunks.
    Evidence:
        Internal: Evaluator records request source tick/time.
        External: None. Findings: Reset cannot overlap predictions.
        Metrics: In-flight count. Confidence: Contract-level.
    Constraints: Trusted local process messages only.
    Reconsider When: Distributed serving is required.
    References: model_process.
    ADR: None.
    """
    def __init__(self,family,modeldir,startup_timeout=600,ros_transport=False,ros_port=8780):
        """Spawn one family adapter and wait for explicit readiness.

        Args: family/modeldir/timeout. Returns: None. Raises: Startup errors.
        Design:
            Context: CUDA initialization can be slow. Concern: Startup is not control latency.
            Decision: Separate readiness measurement. Alternatives: Lazy first-action initialization.
            Quality Attributes: Predictability. Trade-offs: Warm-start benchmark.
        Evidence:
            Internal: Child emits ready after model load. External: None.
            Findings: No physics ticks occur yet. Metrics: Startup seconds. Confidence: Measured.
        Constraints: Child uses spawn.
        Reconsider When: Cold-start behavior is measured.
        References: model_process.
        ADR: None.
        """
        ctx=mp.get_context('spawn');self.input=ctx.Queue(maxsize=1);self.output=ctx.Queue(maxsize=1)
        self.process=ctx.Process(target=model_process,args=(family,str(modeldir),self.input,self.output,bool(ros_transport),int(ros_port)),daemon=True)
        self.process.start();self.counter=0;self.inflight=False;self.prefix=family+"-"+uuid.uuid4().hex[:12];self.warmed=False;self.warmup_records=[]
        self.ready=self.wait(startup_timeout)
        if self.ready['kind']!='ready':raise RuntimeError(self.ready)

    def send(self,message):
        """Submit one request without accumulating an inference backlog.

        Args: message: operation dict. Returns: request id. Raises: RuntimeError if busy.
        Design:
            Context: Only latest eligible sensing should be queried.
            Concern: Queueing multiple old observations biases latency.
            Decision: One in-flight operation. Alternatives: Unlimited requests.
            Quality Attributes: Comparability. Trade-offs: No batching.
        Evidence:
            Internal: Fixed queue size. External: None. Findings: Busy requests are rejected.
            Metrics: Request IDs. Confidence: High.
        Constraints: Main control-thread use.
        Reconsider When: Batch inference is compared.
        References: poll.
        ADR: None.
        """
        if self.inflight:raise RuntimeError('inference worker already busy')
        self.counter+=1;request_id=f'{self.prefix}-query-{self.counter}';message=dict(message,request_id=request_id)
        try:self.input.put_nowait(message)
        except queue.Full:raise RuntimeError('model request queue unexpectedly full')
        self.inflight=True;return request_id

    def predict(self,observation,tick,observation_time):
        """Copy only permitted RGB and robot sensors into one asynchronous query.

        Args: observation: env result; tick/time: acquisition identity.
        Returns: request id. Raises: KeyError if sensing is incomplete.
        Design:
            Context: Environment results also include evaluator metrics.
            Concern: Goal/object truth must not reach the policy.
            Decision: Explicit allowlist of five raw sensing keys and language prompt.
            Alternatives: Forwarding the full observation leaks privileged state.
            Quality Attributes: Integrity. Trade-offs: Schema-specific adapter.
        Evidence:
            Internal: Both family adapters use official env processors.
            External: LIBERO VLA observation schema.
            Findings: Images stay unrotated until official processing.
            Metrics: Sensor hashes recorded by caller. Confidence: High.
        Constraints: No metrics or success flags are sent.
        Reconsider When: Model modalities change.
        References: Family runtimes.
        ADR: None.
        """
        sensors={key:np.asarray(observation[key]).copy() for key in SENSORS}
        return self.send({'op':'predict','observation':sensors,'prompt':observation['prompt'],
                          'observation_tick':tick,'observation_time':observation_time})

    def poll(self):
        """Read a completed request without waiting or advancing simulated time.

        Args: None. Returns: result or None. Raises: RuntimeError on child error.
        Design:
            Context: Control deadlines must be independent of inference.
            Concern: A blocking result call would pause physics.
            Decision: Nonblocking queue read in timed loop.
            Alternatives: wait belongs only outside realtime control.
            Quality Attributes: Timeliness. Trade-offs: Results applied at next control tick.
        Evidence:
            Internal: get_nowait. External: None. Findings: Empty queue returns immediately.
            Metrics: Result ready time. Confidence: High.
        Constraints: Single consumer.
        Reconsider When: Event-driven scheduling is used.
        References: wait.
        ADR: None.
        """
        try:result=self.output.get_nowait()
        except queue.Empty:return None
        self.inflight=False
        if result['kind']=='error':raise RuntimeError(result)
        return result

    def wait(self,timeout=120):
        """Wait outside realtime control for startup, reset, baseline or final audit.

        Args: timeout: seconds. Returns: result. Raises: RuntimeError/TimeoutError.
        Design:
            Context: Lockstep reference intentionally ignores inference delays.
            Concern: Timed loops must never call this method.
            Decision: Explicitly named blocking API for untimed phases.
            Alternatives: Hidden blocking inside poll is prohibited.
            Quality Attributes: Clarity. Trade-offs: Separate control paths.
        Evidence:
            Internal: Evaluator mode controls use. External: None.
            Findings: Queue timeout is finite. Metrics: Wall waiting. Confidence: Measured.
        Constraints: No use inside realtime tick loop.
        Reconsider When: API changes.
        References: poll.
        ADR: None.
        """
        try:result=self.output.get(timeout=timeout)
        except queue.Empty:raise TimeoutError('model worker result deadline exceeded')
        self.inflight=False
        if result['kind']=='error':raise RuntimeError(result)
        return result

    def reset(self,seed):
        """Reset model state and RNG between episodes after pending inference drains.

        Args: seed: deterministic integer. Returns: None. Raises: RuntimeError on failure.
        Design:
            Context: Model RNG must be paired across schedules.
            Concern: A previous chunk must not cross episode boundaries.
            Decision: Drain, reset and acknowledge outside timed control.
            Alternatives: Reusing queues contaminates trials.
            Quality Attributes: Reproducibility. Trade-offs: Untimed reset cost.
        Evidence:
            Internal: Family reset calls official policy.reset. External: None.
            Findings: Model weights are unchanged. Metrics: Seed. Confidence: Audited.
        Constraints: Not called during control.
        Reconsider When: Recurrent state is transferred intentionally.
        References: Family runtimes.
        ADR: None.
        """
        if self.inflight:self.wait()
        self.send({'op':'reset','seed':int(seed)});self.wait()

    def audit(self):
        """Measure frozen-state identity after all outstanding predictions complete.

        Args: None. Returns: model audit. Raises: RuntimeError on mutation/failure.
        Design:
            Context: No local training is allowed.
            Concern: File identity alone misses in-memory changes.
            Decision: Request family parameter/buffer state digest.
            Alternatives: Trusting eval flags is weaker evidence.
            Quality Attributes: Integrity. Trade-offs: Untimed hashing overhead.
        Evidence:
            Internal: Family adapters hash before/after.
            External: None. Findings: No optimizer exists.
            Metrics: Exact digest. Confidence: High.
        Constraints: Untimed phase.
        Reconsider When: Intentional adaptation is added.
        References: Family audit.
        ADR: None.
        """
        if self.inflight:self.wait()
        self.send({'op':'audit'});return self.wait()['audit']

    def close(self):
        """Release the owned GPU process and its IPC resources with a finite deadline.

        Args: None. Returns: None. Raises: None.
        Design:
            Context: The second model must run without GPU contention.
            Concern: An idle process can retain gigabytes of VRAM.
            Decision: Graceful close followed by finite terminate fallback.
            Alternatives: Relying on interpreter exit leaks resources during a sweep.
            Quality Attributes: Resource hygiene. Trade-offs: Incomplete failing work may be interrupted.
        Evidence:
            Internal: Owned process handle. External: Python process lifecycle.
            Findings: Never terminates unrelated processes.
            Metrics: Child alive flag. Confidence: High.
        Constraints: Called in evaluator finally block.
        Reconsider When: A persistent remote service is used.
        References: model_process.
        ADR: None.
        """
        try:
            if self.inflight:self.wait(120)
            self.input.put({'op':'close'},timeout=1)
            self.process.join(15)
        except Exception:pass
        if self.process.is_alive():self.process.terminate();self.process.join(5)
        self.input.close();self.output.close()
