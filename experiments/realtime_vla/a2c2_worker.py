"""Generate author SmolVLA chunks asynchronously while the owner runs a separate A2C2 head.

Args: Pinned pair directory, device, ROS port and bounded sensor requests.
Returns: ROS-received H50 actions paired with their own source feature/context.
Raises: Explicit serialized initialization, inference or transport errors.
Design: Context: Per-tick correction must continue during expensive base generation.
    Concern: Co-locating base/head in one synchronous call would pause correction.
    Decision: Base-only spawned process, one in-flight request, original lifecycle reuse.
    Alternatives: Modify archived worker/evaluate code.
    Quality Attributes: Isolation, observability. Trade-offs: Shared-GPU contention is measured.
Evidence: Internal: Combined/split CPU parity and all stored tensors restored.
    External: A2C2 paper2509.23224. Findings: Context must travel with its own chunk.
    Metrics: ROS hashes, source identity, latency and per-process CUDA peak.
    Confidence: Functional protocol; task results measured separately.
Constraints: One resident VLA plus one small head, no learning or object truth.
Reconsider When: Model ownership or action semantics change.
References: a2c2_runtime.py. ADR: None.
"""
import multiprocessing as mp
import time
import traceback
import uuid
import numpy as np
if __package__:
    from .worker import ModelWorker, SENSORS
else:
    from worker import ModelWorker, SENSORS


def validate_packet(actions, context):
    """Require exact correspondence between ROS-received actions and the context's base chunk.

    Args: actions: received50x7; context: source feature record.
    Returns: Float64 received actions. Raises: ValueError for any mismatch or invalid feature.
    Design: Context: Original source indices survive aligned prefix dropping.
        Concern: Accidentally pairing a newer feature with an older action.
        Decision: Exact numeric chunk equality, feature shape/finite checks and generation id.
        Alternatives: Trust shape only. Quality Attributes: Causal integrity. Trade-offs: Small scan.
    Evidence: Internal: CPU split/pickle parity. External: Author eval chunk entries.
        Findings: Context actions remain unnormalized. Metrics:50x7/1x960. Confidence: High.
    Constraints: No rescaling. Reconsider When: Quantized transport is introduced.
    References: A2C2ChunkContext. ADR: None.
    """
    received = np.asarray(actions, dtype=np.float64)
    if received.shape != (50, 7) or not np.isfinite(received).all():
        raise ValueError('expected finite ROS-received50x7')
    if context.actions.shape != received.shape or not np.array_equal(received, context.actions):
        raise ValueError('context and ROS-received chunk do not match')
    if context.vlm_hidden.shape != (1, 960) or not np.isfinite(context.vlm_hidden).all():
        raise ValueError('invalid source hidden960')
    if type(context.generation) is not int or context.generation < 1:
        raise ValueError('invalid context generation')
    return received


def a2c2_base_process(modeldir, incoming, outgoing, ros_port, device):
    """Own only the frozen base and return source-bound chunks through real ROS exchanges.

    Args: modeldir/queues/port/device: process-owned configuration.
    Returns: None on close. Raises: Errors become explicit packets.
    Design: Context: Main process runs head and physics while base remains busy.
        Concern: Hidden head loading doubles resources or blocks the control thread.
        Decision: load_head=False; sensor/action ROS receipt before packet emission.
        Alternatives: Blocking base inside head. Quality Attributes: Timeliness, auditability.
        Trade-offs: Per-query IPC and DDS overhead.
    Evidence: Internal: Base-only strict restoration. External: Torch spawn.
        Findings: No weights or optimizer cross IPC. Metrics: Wall and CUDA peak.
        Confidence: Runtime measured.
    Constraints: Sensor allowlist, fixed author model, no training.
    Reconsider When: Multiple base queries are allowed. References: A2C2Worker. ADR: None.
    """
    runtime = None
    try:
        import torch
        if __package__:
            from .a2c2_runtime import FrozenA2C2Runtime
            from .ros_client import sensor_roundtrip, chunk_roundtrip
        else:
            from a2c2_runtime import FrozenA2C2Runtime
            from ros_client import sensor_roundtrip, chunk_roundtrip
        torch.set_num_threads(4)
        started = time.monotonic()
        runtime = FrozenA2C2Runtime(modeldir, device=device, load_base=True, load_head=False)
        outgoing.put({'kind':'ready', 'startup_seconds':time.monotonic()-started, 'metadata':runtime.metadata})
        while True:
            message = incoming.get()
            op, request_id = message['op'], message.get('request_id')
            if op == 'close':
                break
            if op == 'reset':
                runtime.reset(message['seed'])
                outgoing.put({'kind':'reset', 'request_id':request_id})
                continue
            if op == 'audit':
                outgoing.put({'kind':'audit', 'request_id':request_id, 'audit':runtime.audit()})
                continue
            if op != 'predict':
                raise ValueError('unsupported base worker operation')
            if runtime.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(runtime.device)
            started = time.monotonic()
            sensors, prompt, sensor_transport = sensor_roundtrip(message['observation'], message['prompt'], request_id, port=ros_port)
            actions, metadata = runtime.predict(sensors, prompt)
            received, action_transport = chunk_roundtrip(actions, request_id, port=ros_port)
            context = runtime.last_context
            received = validate_packet(received, context)
            metadata.update(sensor_transport=sensor_transport, action_transport=action_transport,
                            peak_cuda_bytes=torch.cuda.max_memory_allocated(runtime.device) if runtime.device.type == 'cuda' else None,
                            allocated_cuda_bytes=torch.cuda.memory_allocated(runtime.device) if runtime.device.type == 'cuda' else None,
                            reserved_cuda_bytes=torch.cuda.memory_reserved(runtime.device) if runtime.device.type == 'cuda' else None)
            outgoing.put({'kind':'chunk', 'request_id':request_id, 'actions':received, 'context':context,
                          'metadata':metadata, 'started_time':started, 'finished_time':time.monotonic(),
                          'observation_tick':message['observation_tick'], 'observation_time':message['observation_time'],
                          'request_tick':message['request_tick']})
    except BaseException as error:
        outgoing.put({'kind':'error', 'error':f'{type(error).__name__}: {error}', 'traceback':traceback.format_exc()})


class A2C2Worker(ModelWorker):
    """Use bounded nonblocking baseline lifecycle with a source-context-aware base process.

    Args: modeldir, ROS port, device and startup_timeout.
    Returns: Ready worker. Raises: Child startup/protocol errors.
    Design: Context: Existing evidence must remain unchanged.
        Concern: New fields should not alter ModelWorker's archived family behavior.
        Decision: Specialize only startup and predict schema.
        Alternatives: Global worker edit. Quality Attributes: Compatibility. Trade-offs: New target.
    Evidence: Internal: ModelWorker send/poll/reset/audit/close.
        External: Python multiprocessing spawn. Findings: One query in flight.
        Metrics: Request/source identity. Confidence: CPU protocol validation.
    Constraints: Parent owns scheduling. Reconsider When: Lifecycle changes.
    References: a2c2_base_process. ADR: None.
    """
    def __init__(self, modeldir, ros_port=8780, device='cuda', startup_timeout=600):
        """Spawn only the author base outside task timing and require explicit readiness.

        Args: See class. Returns: None. Raises: Startup failure/timeout.
        Design: Context: Checkpoint restoration precedes task clock.
            Concern: Cold-start latency is a different experiment.
            Decision: Wait for ready before physics. Alternatives: Lazy startup.
            Quality Attributes: Predictability. Trade-offs: Untimed startup.
        Evidence: Internal: Ready includes exact restoration. External: Spawn.
            Findings: Small head remains in parent. Metrics: Startup. Confidence: Measured.
        Constraints: GPU exclusion managed by caller. Reconsider When: Cold-start benchmark.
        References: ModelWorker. ADR: None.
        """
        context = mp.get_context('spawn')
        self.input, self.output = context.Queue(maxsize=1), context.Queue(maxsize=1)
        self.process = context.Process(target=a2c2_base_process,
                                       args=(str(modeldir), self.input, self.output, int(ros_port), device), daemon=True)
        self.process.start()
        self.counter, self.inflight = 0, False
        self.prefix = 'a2c2-base-'+uuid.uuid4().hex[:12]
        self.warmed, self.warmup_records = False, []
        self.ready = self.wait(startup_timeout)
        if self.ready['kind'] != 'ready':
            raise RuntimeError(self.ready)

    def predict(self, observation, tick, observation_time, request_tick=None):
        """Submit only coherent RGB/proprioception and acquisition provenance.

        Args: observation: latest camera sample with task; tick/time: source; request_tick: now.
        Returns: Unique request id. Raises: Busy/schema errors.
        Design: Context: Camera and physical time advance independently.
            Concern: Relabeling old images as fresh hides alignment loss.
            Decision: Keep capture tick separate from request tick and allowlist sensors.
            Alternatives: Forward evaluator truth. Quality Attributes: Integrity. Trade-offs: Copies.
        Evidence: Internal: Original SENSORS tuple. External: A2C2 sensor contract.
            Findings: No object positions enter worker. Metrics: Source age. Confidence: High.
        Constraints: One inflight. Reconsider When: Modalities change.
        References: a2c2_base_process. ADR: None.
        """
        sensors = {key:np.asarray(observation[key]).copy() for key in SENSORS}
        return self.send({'op':'predict', 'observation':sensors, 'prompt':observation['prompt'],
                          'observation_tick':int(tick), 'observation_time':float(observation_time),
                          'request_tick':int(tick if request_tick is None else request_tick)})


def actuator_process(incoming, outgoing, ros_port, timeout):
    """Send one7D proposal at a time through ROS without blocking physical control.

    Args: bounded queues, bridge port and diagnostic network timeout.
    Returns: None; packets carry source/target identity and actual received values.
    Raises: Startup failures serialized; per-request timeouts become explicit nonusable packets.
    Design: Context: DDS can finish while TCP response delivery stalls.
        Concern: Waiting in the physical process halted v1 after57ticks.
        Decision: CPU-only owner, one inflight and no backlog; no retry or actuator application.
        Alternatives: Increase control timeout. Quality Attributes: Temporal isolation.
        Trade-offs: Late proposals fall back to previously ROS-received base actions.
    Evidence: Internal: v1 failure and276call transport diagnostic.
        External: Existing ROS bridge receipts. Findings: Process imports no Torch models.
        Metrics: Target/deadline/ROS time and timeout status. Confidence: Runtime measured.
    Constraints: Source/base indices preserved; actual values applied only by parent executor.
    Reconsider When: Direct ROS binding replaces TCP. References: ActuatorWorker. ADR: None.
    """
    try:
        import sys
        if __package__:
            from .ros_client import RosRoundTrip
        else:
            from ros_client import RosRoundTrip
        client = RosRoundTrip(port=ros_port, timeout=timeout)
        outgoing.put({'kind':'ready', 'cpu_only':True, 'torch_imported':'torch' in sys.modules})
        while True:
            message = incoming.get()
            if message['op'] == 'close':
                break
            if message['op'] != 'actuate':
                raise ValueError('unexpected actuator operation')
            started = time.monotonic()
            packet = {k:v for k,v in message.items() if k != 'op'}
            try:
                received, metadata = client.actions(np.asarray(message['proposed_action']).reshape(1, 7),
                                                     request_id=message['request_id'])
                packet.update(kind='action', received_action=received[0], action_transport=metadata)
            except TimeoutError as error:
                packet.update(kind='timeout', error=f'{type(error).__name__}: {error}')
            except Exception as error:
                packet.update(kind='error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            packet.update(started_time=started, finished_time=time.monotonic())
            outgoing.put(packet)
    except BaseException as error:
        outgoing.put({'kind':'error', 'error':f'{type(error).__name__}: {error}', 'traceback':traceback.format_exc()})


class ActuatorWorker(ModelWorker):
    """Expose one-inflight zero-backlog ROS delivery independently of the GPU/base worker.

    Args: ros_port/timeout/startup_timeout. Returns: Ready CPU worker.
    Raises: Startup/protocol errors.
    Design: Context: A proposal is useful only for its declared next tick and base source.
        Concern: Old relative commands cannot be reused after a deadline.
        Decision: Parent validates identity; worker only transports and records receipt.
        Alternatives: Infinite command queue. Quality Attributes: Boundedness.
        Trade-offs: Base fallback is explicit when a proposal is late.
    Evidence: Internal: Existing ModelWorker bounded send/close lifecycle.
        External: ROS broker. Findings: No model is loaded in this process.
        Metrics: Late/timeout/rejected source counts. Confidence: Integration checked separately.
    Constraints: One caller and one query; no physical step.
    Reconsider When: Multi-actuator arbitration is needed. References: actuator_process. ADR: None.
    """
    def __init__(self, ros_port=8780, timeout=.5, startup_timeout=30):
        """Start the small CPU-only transport owner before task timing.

        Args: See class. Returns: None. Raises: Startup failures.
        Design: Context: GPU model count stays unchanged.
            Concern: Lazy initialization could consume control budget.
            Decision: Explicit ready. Alternatives: First-tick startup.
            Quality Attributes: Predictability. Trade-offs: One process.
        Evidence: Internal: Ready states Torch import status. External: Python spawn.
            Findings: No simulator in child. Metrics: None. Confidence: Protocol.
        Constraints: Existing bridge only. Reconsider When: Transport changes.
        References: actuator_process. ADR: None.
        """
        context = mp.get_context('spawn')
        self.input, self.output = context.Queue(maxsize=1), context.Queue(maxsize=1)
        self.process = context.Process(target=actuator_process,
                                       args=(self.input, self.output, int(ros_port), float(timeout)), daemon=True)
        self.process.start()
        self.counter, self.inflight = 0, False
        self.prefix = 'a2c2-actuator-'+uuid.uuid4().hex[:12]
        self.ready = self.wait(startup_timeout)
        if self.ready['kind'] != 'ready' or not self.ready['cpu_only']:
            raise RuntimeError(self.ready)

    def submit(self, action, *, source_request_id, source_generation, original_action_index,
               target_tick, target_deadline, proposal_kind):
        """Submit a clipped next-tick proposal with immutable source and target provenance.

        Args: action7 and explicit original generation/index, target tick/deadline, kind.
        Returns: Request ID. Raises: ValueError or busy error.
        Design: Context: New base chunks can supersede a pending correction.
            Concern: Shape alone cannot detect a stale source.
            Decision: Carry full temporal identity through asynchronous ROS receipt.
            Alternatives: Apply last available command. Quality Attributes: Causal integrity.
            Trade-offs: Caller may discard a successfully transported packet.
        Evidence: Internal: Matching executor gate. External: A2C2 per-chunk context.
            Findings: No hidden action slicing. Metrics: One-tick pipeline.
            Confidence: Checked contract.
        Constraints: No out-of-range/nonfinite action. Reconsider When: Horizon changes.
        References: choose_received_action. ADR: None.
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all() or np.abs(action).max() > 1:
            raise ValueError('actuator proposal must be clipped finite7D')
        if type(original_action_index) is not int or not 0 <= original_action_index < 50:
            raise ValueError('actuator proposal needs original50-step index')
        return self.send({'op':'actuate', 'proposed_action':action.copy(),
                          'source_request_id':source_request_id, 'source_generation':int(source_generation),
                          'original_action_index':original_action_index, 'target_tick':int(target_tick),
                          'target_deadline':float(target_deadline), 'proposal_kind':proposal_kind})

    def poll(self):
        """Read one result without waiting; timeouts remain data rather than fatal fallback loss.

        Args: None. Returns: Packet or None. Raises: Unexpected child/protocol errors.
        Design: Context: Parent uses received base rows while delivery is pending.
            Concern: Inherited poll treats all error packets as fatal.
            Decision: Keep timeout explicit; preserve fatal integrity/runtime error behavior.
            Alternatives: Silently retry. Quality Attributes: Transparency. Trade-offs: None.
        Evidence: Internal: Packet kind. External: queue.get_nowait.
            Findings: No waiting or model call. Metrics: None. Confidence: Contract.
        Constraints: Read only when inflight. Reconsider When: More response kinds exist.
        References: actuator_process. ADR: None.
        """
        import queue
        try:
            packet = self.output.get_nowait()
        except queue.Empty:
            return None
        self.inflight = False
        if packet['kind'] == 'error':
            raise RuntimeError(packet)
        return packet
