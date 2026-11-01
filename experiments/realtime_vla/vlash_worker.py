"""Isolate the exact author VLASH policy behind real ROS sensor/chunk transport.

Args: Local checkpoint, ROS port and sensor-only time-labelled requests.
Returns: Native chunks, independent transport hashes and represented-state timing metadata.
Raises: Explicit child errors; no model fallback.
Design: Context: CPU physics/camera must continue while an author-specific GPU backend runs.
    Concern: Old images and predicted robot state must not be described as a fresh measurement.
    Decision: Dedicated process reuses ModelWorker lifecycle and actual ROS roundtrips;
        capture, measured proprioception and state-origin labels travel separately.
    Alternatives: Edit archived baseline workers or infer synchronously inside physics.
    Quality Attributes: Isolation, inspectability. Trade-offs: Own pinned inference overlay.
Evidence: Internal: Shared five-sensor allowlist and ROS hash verification.
    External: Author VLASH current-state/old-image evaluation pattern.
    Findings: Predictions are explicitly marked, never relabelled as measured sensors.
    Metrics: IPC/ROS/model wall time, image/proprio/state ticks. Confidence: Runtime audit required.
Constraints: One VLA resident, one request inflight, no simulator object truth or training.
Reconsider When: Multiquery serving or another sensor schema is introduced.
References: vlash_runtime.py; evaluate_vlash.py. ADR: None.
"""
import multiprocessing as mp
import time
import traceback
import uuid

import numpy as np
try:
    from .worker import ModelWorker,SENSORS
except ImportError:
    from worker import ModelWorker,SENSORS


def vlash_process(modeldir,incoming,outgoing,ros_port):
    """Own one strict frozen model and perform inference only on ROS-received sensors.

    Args: modeldir: weights; incoming/outgoing: bounded IPC; ros_port: real bridge.
    Returns: None. Raises: Exceptions are serialized as explicit worker errors.
    Design: Context: Separate backend and physical clocks. Concern: Bypassing ROS for model inputs.
        Decision: Received sensor packet→native model→generated chunk→received ROS action packet.
        Alternatives: Hash-only simulated transport. Quality Attributes: Traceability. Trade-offs: Actual overhead.
    Evidence: Internal: Existing ros_client strict roundtrip. External: ROS2 sensor_msgs/Image and String topics.
        Findings: Scheduler metadata never enters model observations. Metrics: Model/transport timings.
        Confidence: Independent bridge logs needed for final validation.
    Constraints: No object/evaluator fields. Reconsider When: ROS transport schema changes.
    References: FrozenVLASH; ModelWorker. ADR: None.
    """
    runtime=None
    try:
        import torch
        try:
            from .vlash_runtime import FrozenVLASH
            from .ros_client import sensor_roundtrip,chunk_roundtrip
        except ImportError:
            from vlash_runtime import FrozenVLASH
            from ros_client import sensor_roundtrip,chunk_roundtrip
        torch.set_num_threads(4);start=time.perf_counter()
        runtime=FrozenVLASH(modeldir,device='cuda')
        outgoing.put({'kind':'ready','startup_seconds':time.perf_counter()-start,'backend':runtime.provenance['backend']})
        while True:
            message=incoming.get();op=message['op'];request_id=message.get('request_id')
            if op=='close':break
            if op=='reset':
                runtime.reset(message['seed']);outgoing.put({'kind':'reset','request_id':request_id});continue
            if op=='audit':
                outgoing.put({'kind':'audit','request_id':request_id,'audit':runtime.audit()});continue
            if op!='predict':raise ValueError('unknown VLASH worker operation')
            started=time.monotonic()
            sensors,prompt,sensor_transport=sensor_roundtrip(message['observation'],message['prompt'],request_id,port=ros_port)
            actions,metadata=runtime.predict(sensors,prompt)
            actions,action_transport=chunk_roundtrip(actions,request_id,port=ros_port)
            actions=np.asarray(actions,dtype=np.float64)
            if actions.shape!=(50,7) or not np.isfinite(actions).all():raise ValueError('invalid VLASH physical action chunk')
            times={key:message[key] for key in ('request_time','request_tick','observation_tick','observation_time',
                                                'measured_proprio_tick','measured_proprio_time','state_origin_tick','state_condition')}
            metadata=dict(metadata,ros_transport=True,sensor_transport=sensor_transport,action_transport=action_transport,
                          state_is_prediction=message['state_condition']=='predicted',predictor=message.get('predictor'),
                          query_image_age_s=max(0.,message['request_time']-message['observation_time']),
                          query_measured_proprio_age_s=max(0.,message['request_time']-message['measured_proprio_time']),**times)
            outgoing.put({'kind':'chunk','request_id':request_id,'actions':actions,'metadata':metadata,
                          'started_time':started,'finished_time':time.monotonic(),**times})
    except BaseException as error:
        outgoing.put({'kind':'error','error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()})
    finally:
        if runtime is not None:runtime.close()


class VLASHWorker(ModelWorker):
    """Reuse the bounded nonblocking worker lifecycle for author VLASH inference.

    Args: modeldir, ros_port, startup_timeout. Returns: Worker. Raises: Startup/protocol errors.
    Design: Context: Existing ModelWorker is frozen for old comparisons. Concern: Cross-method changes.
        Decision: Override only startup and sensor/time schema. Alternatives: Modify core worker.
        Quality Attributes: Isolation. Trade-offs: New specialized child target.
    Evidence: Internal: Existing send/reset/poll/wait/audit/close implementations.
        External: Python spawn. Findings: Actual GPU class is loaded only in the child.
        Metrics: Ready/latency. Confidence: Per-run audit.
    Constraints: Launch with .venv-vlash Python. Reconsider When: External model serving is used.
    References: vlash_process. ADR: None.
    """
    def __init__(self,modeldir,ros_port=8780,startup_timeout=600):
        """Spawn the isolated author runtime before the timed task begins.

        Args: modeldir: checkpoint; ros_port: bridge; startup_timeout: finite seconds.
        Returns: None. Raises: RuntimeError/TimeoutError.
        Design: Context: Large strict restoration. Concern: Startup mistaken for action latency.
            Decision: Explicit readiness phase. Alternatives: Lazy initialization during control.
            Quality Attributes: Comparability. Trade-offs: Warm-start experiment.
        Evidence: Internal: Bounded IPC queues. External: Spawn semantics.
            Findings: No physics runs while startup is measured. Metrics: Startup seconds. Confidence: Measured.
        Constraints: Exclusive GPU/CPU resource lease. Reconsider When: Cold-start control is studied.
        References: vlash_process. ADR: None.
        """
        ctx=mp.get_context('spawn');self.input=ctx.Queue(maxsize=1);self.output=ctx.Queue(maxsize=1)
        self.process=ctx.Process(target=vlash_process,args=(str(modeldir),self.input,self.output,int(ros_port)),daemon=True)
        self.process.start();self.counter=0;self.inflight=False;self.prefix='vlash-'+uuid.uuid4().hex[:12]
        self.warmed=False;self.warmup_records=[];self.ready=self.wait(startup_timeout)
        if self.ready['kind']!='ready':raise RuntimeError(self.ready)

    def predict(self,observation,tick,observation_time,*,request_tick=None,measured_proprio_tick=None,
                measured_proprio_time=None,state_origin_tick=None,state_condition='current',predictor=None):
        """Send allowlisted sensors and three distinct timing identities without truth leakage.

        Args: observation: raw cameras/selected proprioception and prompt; tick/time: image capture;
            request_tick: current control tick; measured_proprio_tick/time: last actual robot sensing;
            state_origin_tick: represented state target; state_condition: coherent/current/predicted;
            predictor: causal forecast metadata outside model input.
        Returns: Request ID. Raises: Invalid conditions or worker busy errors.
        Design: Context: Mixed-age conditioning. Concern: Predicted state mislabeled as observed.
            Decision: Explicit separate provenance fields and sensor-only model packet.
            Alternatives: One overloaded observation tick. Quality Attributes: Auditability. Trade-offs: Metadata size.
        Evidence: Internal: Five raw fields copied. External: VLASH state/action offset semantics.
            Findings: Model sees no success/object metrics or timing labels. Metrics: Three ticks and measured age.
            Confidence: Allowlist boundary.
        Constraints: One inflight query. Reconsider When: New sensor fields are needed.
        References: evaluate_vlash.py. ADR: None.
        """
        if state_condition not in ('coherent','current','predicted'):raise ValueError('unsupported state condition')
        query_tick=int(tick if request_tick is None else request_tick)
        source_tick=int(tick if measured_proprio_tick is None else measured_proprio_tick)
        origin=int(source_tick if state_origin_tick is None else state_origin_tick)
        if int(tick)<0 or int(tick)>query_tick or source_tick>query_tick or source_tick<0 or origin<0:
            raise ValueError('noncausal VLASH timing metadata')
        if state_condition!='predicted' and origin>query_tick:raise ValueError('only predicted state can represent a future tick')
        sensors={key:np.asarray(observation[key]).copy() for key in SENSORS}
        return self.send({'op':'predict','observation':sensors,'prompt':observation['prompt'],
                          'observation_tick':int(tick),'observation_time':float(observation_time),
                          'request_tick':query_tick,'request_time':time.monotonic(),
                          'measured_proprio_tick':source_tick,
                          'measured_proprio_time':float(observation_time if measured_proprio_time is None else measured_proprio_time),
                          'state_origin_tick':origin,'state_condition':state_condition,'predictor':predictor})
