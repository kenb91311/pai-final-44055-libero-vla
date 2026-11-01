"""Isolate RTC chunk generation from the physical control clock.

Args: Frozen family, RTC variant and sensor/action-prefix messages.
Returns: Full physical and normalized chunks through bounded IPC.
Raises: Explicit child/transport/runtime errors.
Design: Context: RTC differentiates action latents while physics keeps running.
    Concern: Reusing inference_mode or hiding wait time invalidates this experiment.
    Decision: Dedicated runtime process, common inherited nonblocking lifecycle.
    Alternatives: Modify the archived 35-trial scheduler.
    Quality Attributes: Isolation, traceability. Trade-offs: IPC copies observations.
Evidence: Internal: Original ModelWorker bounded protocol. External: LeRobot guided RTC.
    Findings: Prefix is executed/queued actuator history, not privileged scene state.
    Metrics: Worker time and transport hashes. Confidence: Rollout measured separately.
Constraints: One resident VLA, no optimizer or object truth in model inputs.
Reconsider When: Distributed serving or simultaneous models are evaluated.
References: continuity_runtime.py; docs/reference/realtime-continuity-methods.md. ADR: None.
"""
import multiprocessing as mp
import time
import traceback
import uuid
import numpy as np
from worker import ModelWorker,SENSORS


def continuity_process(family,modeldir,variant,incoming,outgoing,ros_port):
    """Own one frozen runtime and transform sensor-only requests into chunks.

    Args: family/modeldir/variant: asset and method; queues: IPC; ros_port: bridge.
    Returns: None. Raises: Errors become explicit error messages.
    Design: Context: Guidance has its own latency and memory cost.
        Concern: Control must never block on it.
        Decision: Spawned process and exactly one inflight query.
        Alternatives: Synchronous generation. Quality Attributes: Timeliness.
        Trade-offs: Startup outside episode time.
    Evidence: Internal: Inherited queue protocol. External: torch multiprocessing spawn.
        Findings: Actuator prefix is normalized with the selected model statistics.
        Metrics: Model/ROS time. Confidence: Runtime validation required.
    Constraints: No training. Reconsider When: Runtime schema changes.
    References: FrozenContinuityPolicy. ADR: None.
    """
    runtime=None
    try:
        import torch
        from continuity_runtime import FrozenContinuityPolicy
        from ros_client import sensor_roundtrip,chunk_roundtrip
        torch.set_num_threads(4);start=time.perf_counter()
        runtime=FrozenContinuityPolicy(family,modeldir,variant=variant,device='cuda')
        outgoing.put({'kind':'ready','startup_seconds':time.perf_counter()-start,'variant':variant})
        while True:
            message=incoming.get();op=message['op'];request_id=message.get('request_id')
            if op=='close':break
            if op=='reset':
                runtime.reset(message['seed']);outgoing.put({'kind':'reset','request_id':request_id});continue
            if op=='audit':
                outgoing.put({'kind':'audit','request_id':request_id,'audit':runtime.audit()});continue
            if op!='predict':raise ValueError('unknown worker operation')
            start=time.monotonic()
            sensors,prompt,sensor_transport=sensor_roundtrip(message['observation'],message['prompt'],request_id,port=ros_port)
            prefix=message.get('prefix_actions')
            normalized=None if prefix is None else runtime.normalize_actions(prefix)
            result=runtime.predict(sensors,prompt,prev_normalized=normalized,inference_delay=message['inference_delay'],execution_horizon=message['execution_horizon'])
            actions,action_transport=chunk_roundtrip(result['actions'],request_id,port=ros_port)
            meta=dict(result['metadata'],ros_transport=True,sensor_transport=sensor_transport,action_transport=action_transport,
                      prefix_length=0 if prefix is None else len(prefix),predicted_delay_ticks=message['inference_delay'],
                      raw_predicted_delay_ticks=message['raw_predicted_delay'],delay_prediction_saturated=message['raw_predicted_delay']>49,
                      execution_horizon=message['execution_horizon'],request_tick=message['request_tick'])
            actions=np.asarray(actions,dtype=np.float64)
            if actions.shape!=(50,7) or not np.isfinite(actions).all():raise ValueError('expected finite full H50x7')
            normalized=np.asarray(result['normalized'],dtype=np.float32)
            if normalized.shape!=actions.shape or not np.isfinite(normalized).all():raise ValueError('invalid paired normalized H50x7')
            outgoing.put({'kind':'chunk','request_id':request_id,'actions':actions,'normalized':normalized,'metadata':meta,
                          'started_time':start,'finished_time':time.monotonic(),'observation_tick':message['observation_tick'],
                          'observation_time':message['observation_time'],'request_tick':message['request_tick']})
    except BaseException as error:
        outgoing.put({'kind':'error','error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()})
    finally:
        if runtime is not None:runtime.close()


class ContinuityWorker(ModelWorker):
    """Reuse nonblocking lifecycle with an RTC-specific sensor/prefix protocol.

    Args: family/modeldir/variant/ros_port/startup_timeout. Returns: Worker.
    Raises: Child initialization or protocol errors.
    Design: Context: Existing worker is an archived baseline.
        Concern: New context must not silently change old evaluations.
        Decision: Subclass only startup and request schema.
        Alternatives: Global baseline edits. Quality Attributes: Compatibility.
        Trade-offs: Dedicated process target.
    Evidence: Internal: ModelWorker reset/poll/wait/audit/close.
        External: Python spawn. Findings: Both paths retain bounded one-query IPC.
        Metrics: Source/request ticks. Confidence: Contract checks.
    Constraints: One consumer. Reconsider When: Core protocol changes.
    References: ModelWorker. ADR: None.
    """
    def __init__(self,family,modeldir,variant='off',ros_port=8780,startup_timeout=600):
        """Start the isolated runtime outside the timed task.

        Args: See class. Returns: None. Raises: RuntimeError/TimeoutError.
        Design: Context: Large checkpoints take minutes to restore and audit.
            Concern: Cold start should not be mistaken for action latency.
            Decision: Explicit readiness. Alternatives: Lazy load in first tick.
            Quality Attributes: Comparability. Trade-offs: Warm-start experiment.
        Evidence: Internal: Child ready message. External: Spawn semantics.
            Findings: No physical ticks during startup. Metrics: Startup seconds.
            Confidence: Measured.
        Constraints: GPU exclusion managed by caller. Reconsider When: Cold start is studied.
        References: continuity_process. ADR: None.
        """
        ctx=mp.get_context('spawn');self.input=ctx.Queue(maxsize=1);self.output=ctx.Queue(maxsize=1)
        self.process=ctx.Process(target=continuity_process,args=(family,str(modeldir),variant,self.input,self.output,int(ros_port)),daemon=True)
        self.process.start();self.counter=0;self.inflight=False;self.prefix=family+'-'+variant+'-'+uuid.uuid4().hex[:12];self.warmed=False;self.warmup_records=[]
        self.ready=self.wait(startup_timeout)
        if self.ready['kind']!='ready':raise RuntimeError(self.ready)

    def predict(self,observation,tick,observation_time,*,prefix_actions=None,inference_delay=0,execution_horizon=30,request_tick=None,raw_predicted_delay=None):
        """Send RGB/proprioception and a time-aligned actuator prefix only.

        Args: observation/tick/time: sensor identity; prefix_actions: physical7D;
            inference_delay/horizon: source-origin guidance indices; request_tick: control now;
            raw_predicted_delay: pre-cap delay estimate, retained to expose saturation.
        Returns: Request ID. Raises: Busy/schema errors.
        Design: Context: Old images may precede the request by several control ticks.
            Concern: RTC prefix must start at the image tick, not query submission.
            Decision: Caller explicitly supplies executed history plus queued future.
            Alternatives: Relabel stale images as current. Quality Attributes: Causal integrity.
            Trade-offs: Prefix sent over local IPC after ROS-verified action receipt.
        Evidence: Internal: Explicit five-sensor allowlist. External: RTC prefix conditioning.
            Findings: Scene evaluator metrics never enter this request.
            Metrics: Capture/request tick and predicted delay. Confidence: Logged.
        Constraints: Prefix contains no future truth. Reconsider When: Action space changes.
        References: evaluate_continuity.py. ADR: None.
        """
        sensors={key:np.asarray(observation[key]).copy() for key in SENSORS}
        return self.send({'op':'predict','observation':sensors,'prompt':observation['prompt'],
                          'observation_tick':int(tick),'observation_time':float(observation_time),
                          'prefix_actions':prefix_actions,'inference_delay':int(inference_delay),
                          'execution_horizon':int(execution_horizon),'request_tick':int(tick if request_tick is None else request_tick),
                          'raw_predicted_delay':int(inference_delay if raw_predicted_delay is None else raw_predicted_delay)})
