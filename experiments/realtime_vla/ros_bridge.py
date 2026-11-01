"""Transport raw LIBERO sensing and frozen-policy chunks through two real ROS2 nodes.

Args: CLI role broker/policy, output path, loopback port and finite request timeout.
Returns: Durable per-node JSONL receive/publish hashes; broker serves newline JSON.
Raises: Malformed data, timeout and inconsistent duplicate identities fail explicitly.
Design:
    Context: Host CUDA/physics and ROSJazzy use isolated dependency environments.
    Concern: Telemetry alone does not establish a ROS input/output path.
    Decision: Actual Image topics are joined by identity; actual received chunks return over RPC.
    Alternatives: Direct host sensor references would bypass ROS.
    Quality Attributes: Traceability, boundedness. Trade-offs: Serialization and DDS overhead.
Evidence:
    Internal: Reuses puzzle image identity, reliable QoS, exact hash and cross-topic join patterns.
    External: ROS2 sensor_msgs/Image and std_msgs/String.
    Findings: Model inference remains in the host GPU worker and is labelled accordingly.
Constraints: Trusted localhost, raw unflipped RGB256, sensor allowlist, no model or training here.
Reconsider When: Direct ROS bindings share the model ABI or camera schema changes.
References: ../pretrained_puzzle/ros_nodes.py; ros_client.py. ADR: None.
"""
import argparse
import base64
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import queue
import socketserver
import threading
import time

CAMERAS=('agentview_image','robot0_eye_in_hand_image')
PROPRIO=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')
MAX_BYTES=4*1024*1024
IMAGE_BYTES=256*256*3


def encoded(value):
    """Serialize finite JSON with a stable representation for hashes and wire framing.

    Args: value: JSON data. Returns: UTF8 bytes. Raises: ValueError on NaN.
    Design: Context: Both node processes compare content. Concern: Formatting drift.
        Decision: Sorted compact JSON. Alternatives: Python object hashes vary.
        Quality Attributes: Reproducibility. Trade-offs: JSON overhead.
    Evidence: Internal: Source and receipt use this function. External: SHA256.
        Findings: Numeric values remain JSON numbers.
    Constraints: Finite data. Reconsider When: Wire schema changes.
    References: digest. ADR: None.
    """
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    """Hash exact bytes or canonical finite JSON.

    Args: value: bytes or JSON. Returns: SHA256. Raises: Serialization errors.
    Design: Context: Logs exclude large image payloads. Concern: Verify receipt.
        Decision: Exact hashes. Alternatives: Full base64 logs are expensive.
        Quality Attributes: Traceability. Trade-offs: Source files needed for reconstruction.
    Evidence: Internal: Smoke matches both node logs. External: SHA256.
        Findings: Changed pixels change the digest.
    Constraints: No approximate equality. Reconsider When: Lossy image transport is introduced.
    References: sensor_digest. ADR: None.
    """
    return hashlib.sha256(value if isinstance(value,bytes) else encoded(value)).hexdigest()


def wire_id(value):
    """Validate the transport UUID carried by each independent ROS topic.

    Args: value: string. Returns: Same string. Raises: ValueError.
    Design: Context: Topics can reorder. Concern: Cross-request joins.
        Decision: Bounded nonempty separator-free ID. Alternatives: Arrival order.
        Quality Attributes: Integrity. Trade-offs: Restricted identifiers.
    Evidence: Internal: Client creates UUIDhex. External: None.
        Findings: Caller request IDs are separately preserved and may repeat between workers.
    Constraints: At most128 characters. Reconsider When: Custom messages replace headers.
    References: JoinedSensors. ADR: None.
    """
    if not isinstance(value,str) or not value or len(value)>128 or '|' in value:
        raise ValueError('invalid transport id')
    return value


def image_bytes(record):
    """Validate raw RGB transport without flipping, resizing or re-encoding pixels.

    Args: record: shape/dtype/base64. Returns: Exact RGB bytes. Raises: ValueError.
    Design: Context: Model adapter owns official image transforms. Concern: Double flips.
        Decision: Require fixed uint8HWC256. Alternatives: Hidden transport transforms.
        Quality Attributes: Fidelity. Trade-offs: Fixed resolution.
    Evidence: Internal: LIBERO raw observation schema. External: ROS rgb8.
        Findings: Payload length is independently checked.
    Constraints: Exact196608 bytes. Reconsider When: Camera resolution changes.
    References: sensor_digest. ADR: None.
    """
    if not isinstance(record,dict) or record.get('shape')!=[256,256,3] or record.get('dtype')!='uint8':
        raise ValueError('expected raw RGB256 uint8')
    raw=base64.b64decode(record['data'],validate=True)
    if len(raw)!=IMAGE_BYTES:raise ValueError('invalid RGB payload length')
    return raw


def sensor_digest(sensors,prompt):
    """Validate approved sensing and hash actual camera bytes plus proprio/prompt.

    Args: sensors: exact five-field mapping; prompt: string. Returns: Combined hash, component hashes.
    Raises: ValueError on privileged/invalid fields or nonfinite proprioception.
    Design: Context: Evaluation has access to object truth. Concern: Policy input leakage.
        Decision: Exact allowlist3/4/2 proprio and two images. Alternatives: Forward whole observation.
        Quality Attributes: Auditability. Trade-offs: New modalities need explicit schema updates.
    Evidence: Internal: Both policies use these raw LIBERO fields. External: None.
        Findings: Success/metrics/object poses cannot enter the approved mapping.
    Constraints: No state conversion in transport. Reconsider When: Policy modalities change.
    References: sensor_roundtrip. ADR: None.
    """
    if not isinstance(sensors,dict) or set(sensors)!=set(CAMERAS+PROPRIO):raise ValueError('sensor allowlist mismatch')
    if not isinstance(prompt,str) or not prompt or len(prompt)>4096:raise ValueError('invalid prompt')
    for key,n in zip(PROPRIO,(3,4,2),strict=True):
        values=sensors[key]
        if not isinstance(values,list) or len(values)!=n or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('invalid proprioception')
    parts={key:digest(image_bytes(sensors[key])) for key in CAMERAS}
    parts['proprio_prompt']=digest({'proprio':{k:sensors[k] for k in PROPRIO},'prompt':prompt})
    return digest(parts),parts


def chunk_digest(actions):
    """Validate a continuous chunk and hash without clipping or numeric rescaling.

    Args: actions: JSON list Nx7. Returns: SHA256. Raises: ValueError.
    Design: Context: Common parent controller applies its own bounds. Concern: Transport alters policy.
        Decision: Preserve all finite numeric values. Alternatives: Clipping would confound evaluation.
        Quality Attributes: Fidelity. Trade-offs: Controller validation remains separate.
    Evidence: Internal: Smol native50 and FAST native10 share action7. External: LIBERO action API.
        Findings: Float tails can exceed1 before controller clipping.
    Constraints: At most1000 rows. Reconsider When: Action dimensionality changes.
    References: chunk_roundtrip. ADR: None.
    """
    if not isinstance(actions,list) or not 0<len(actions)<=1000:raise ValueError('invalid chunk length')
    for row in actions:
        if not isinstance(row,list) or len(row)!=7 or any(type(v) not in (int,float) or not math.isfinite(v) for v in row):raise ValueError('invalid chunk values')
    return digest(actions)


class JoinedSensors:
    """Join independently received ROS images and metadata by exact identity and stamp.

    Args: limit: incomplete count. Returns: Bounded join object. Raises: ValueError on mismatch.
    Design: Context: DDS ordering is per-topic. Concern: Mixed cameras or resets.
        Decision: Identity/stamp join with consistent duplicate checks. Alternatives: Latest-value join.
        Quality Attributes: Correctness. Trade-offs: Missing component causes request timeout.
    Evidence: Internal: Pure tests reorder and duplicate image components. External: ROS2 headers.
        Findings: Complete outputs contain only actual received image bytes.
    Constraints: Two cameras and one metadata record required.
    Reconsider When: Independently timed sensors are intended.
    References: puzzle JoinedObservation pattern. ADR: None.
    """
    def __init__(self,limit=8):
        """Create bounded pending and completed caches.

        Args: limit: positive integer. Returns: None. Raises: ValueError.
        Design: Context: Long experiments. Concern: Retained images. Decision: Bounded caches.
            Alternatives: Unbounded dict. Quality Attributes: Availability. Trade-offs: Eviction times out.
        Evidence: Internal: One outstanding GPU request. External: None. Findings: Eight slots suffice.
        Constraints: Positive limit. Reconsider When: Pipeline depth grows.
        References: JoinedSensors. ADR: None.
        """
        if limit<1:raise ValueError('limit must be positive')
        self.limit=limit;self.pending=OrderedDict();self.completed=OrderedDict()

    def add(self,identity,kind,value,stamp):
        """Store a component and reject changed duplicates or mismatched timestamps.

        Args: identity/kind/value/stamp: received component. Returns: None. Raises: ValueError.
        Design: Context: Messages may repeat. Concern: Same ID with changed bytes.
            Decision: Hash and timestamp equality. Alternatives: Last writer wins.
            Quality Attributes: Integrity. Trade-offs: Inconsistent producers fail.
        Evidence: Internal: Unit duplicate/mismatch checks. External: None. Findings: Reorder is accepted.
        Constraints: Bounded state. Reconsider When: Wire schema changes.
        References: take_ready. ADR: None.
        """
        wire_id(identity)
        if kind not in CAMERAS+('metadata',):raise ValueError('unknown sensor component')
        component_hash=digest(value)
        if identity in self.completed:
            previous=self.completed[identity]
            if previous['stamp']!=stamp or previous[kind]!=component_hash:raise ValueError('changed completed duplicate')
            return
        item=self.pending.setdefault(identity,{'stamp':stamp})
        if item['stamp']!=stamp:raise ValueError('sensor timestamp mismatch')
        if kind in item and digest(item[kind])!=component_hash:raise ValueError('changed sensor duplicate')
        item[kind]=value
        while len(self.pending)>self.limit:self.pending.popitem(last=False)

    def take_ready(self):
        """Consume complete joins while rebuilding image records from subscriber bytes.

        Args: None. Returns: List of identity/sensors/metadata. Raises: Validation errors.
        Design: Context: Broker source payload must not be substituted. Concern: False receipt.
            Decision: Construct base64 from received Image.data only. Alternatives: Echo source payload.
            Quality Attributes: Traceability. Trade-offs: Additional copy.
        Evidence: Internal: Receiver hashes are computed after join. External: None.
            Findings: Completed IDs are deduplicated.
        Constraints: No privileged metadata. Reconsider When: Sensor schema changes.
        References: run_node. ADR: None.
        """
        ready=[]
        for identity,item in list(self.pending.items()):
            if not all(k in item for k in CAMERAS+('metadata',)):continue
            metadata=item['metadata'];sensors={k:metadata['proprio'][k] for k in PROPRIO}
            for camera in CAMERAS:sensors[camera]={'shape':[256,256,3],'dtype':'uint8','data':base64.b64encode(item[camera]).decode()}
            ready.append((identity,sensors,metadata))
            self.completed[identity]={'stamp':item['stamp'],**{k:digest(item[k]) for k in CAMERAS+('metadata',)}}
            del self.pending[identity]
            while len(self.completed)>64:self.completed.popitem(last=False)
        return ready


class EventLog:
    """Append small atomic evidence records without raw image or action arrays.

    Args: output: node directory. Returns: Logger. Raises: Filesystem errors.
    Design: Context: TCP and ROS callbacks use different threads. Concern: Interleaved JSONL.
        Decision: One lock per file. Alternatives: Dumping huge payloads obscures evidence.
        Quality Attributes: Inspectability. Trade-offs: Small synchronous writes.
    Evidence: Internal: Cross-node source/receiver hashes. External: None.
        Findings: Timestamps supplement identity rather than replace it.
    Constraints: No secrets or privileged state. Reconsider When: Logging becomes a bottleneck.
    References: run_node. ADR: None.
    """
    def __init__(self,output):
        """Prepare the owned node output directory.

        Args: output: Path. Returns: None. Raises: Filesystem errors.
        Design: Context: Separate node logs. Concern: Provenance attribution. Decision: Per-role file.
            Alternatives: Shared console scraping. Quality Attributes: Traceability. Trade-offs: Two files.
        Evidence: Internal: Launcher mounts output. External: None. Findings: Append survives stop.
        Constraints: Workspace output. Reconsider When: Central telemetry replaces JSONL.
        References: EventLog. ADR: None.
        """
        self.path=Path(output)/'events.jsonl';self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock()

    def write(self,event,**values):
        """Persist one timestamped event.

        Args: event: label; values: small JSON data. Returns: None. Raises: Filesystem errors.
        Design: Context: Audit lifecycle. Concern: Missing receive evidence. Decision: Append immediately.
            Alternatives: Final summary only. Quality Attributes: Durability. Trade-offs: Disk write.
        Evidence: Internal: Synthetic actual ROS smoke. External: None. Findings: Hash-only logs stay small.
        Constraints: Finite JSON. Reconsider When: Event rate grows.
        References: EventLog. ADR: None.
        """
        with self.lock:
            with self.path.open('ab') as stream:stream.write(encoded({'event':event,'wall_time':time.time(),**values})+b'\n')


def run_node(args):
    """Run a ROS role; only the broker owns the trusted local RPC endpoint.

    Args: args: CLI configuration. Returns: None. Raises: Startup errors.
    Design: Context: Actual separate ROS processes. Concern: Host ABI isolation and no physical clock wait.
        Decision: Threaded TCP waits are contained in GPU worker while ROS executor remains responsive.
        Alternatives: RPC inside env.step would freeze physical control.
        Quality Attributes: Isolation, explicit timeouts. Trade-offs: Local JSON boundary.
    Evidence: Internal: Host worker returns ROS receipt arrays. External: rclpy reliable pub/sub.
        Findings: No simulator or model executes in these nodes.
    Constraints: Loopback8780 and domain45 by launcher; two independent processes.
    Reconsider When: Direct typed ROS services are available in model environment.
    References: run_ros_bridge.sh. ADR: None.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    rclpy.init();log=EventLog(args.output)
    node=Node('realtime_sensor_bridge' if args.role=='broker' else 'realtime_policy_transport')
    qos=QoSProfile(depth=8,reliability=ReliabilityPolicy.RELIABLE,history=HistoryPolicy.KEEP_LAST)
    join=JoinedSensors();pending={};done=OrderedDict();jobs=queue.Queue(maxsize=8);lock=threading.Lock()
    pubs={};server=None;thread=None

    def send(topic,value):
        """Publish canonical finite JSON on an already-created standard String topic.

        Args: topic/value: registered publisher and JSON. Returns: None. Raises: Serialization errors.
        Design: Context: Shared wire contract. Concern: Divergent formatting. Decision: Canonical JSON.
            Alternatives: Custom middleware. Quality Attributes: Simplicity. Trade-offs: JSON strings.
        Evidence: Internal: Hashes use same data. External: std_msgs/String. Findings: Typed images stay separate.
        Constraints: Bounded payload. Reconsider When: Typed custom messages adopted.
        References: run_node. ADR: None.
        """
        pubs[topic].publish(String(data=encoded(value).decode()))

    def ready():
        """Report discovery of every required direction before accepting sensor work.

        Args: None. Returns: Boolean. Raises: None.
        Design: Context: Volatile reliable QoS. Concern: Publication before discovery.
            Decision: Require subscriber counts in both directions. Alternatives: Arbitrary sleep.
            Quality Attributes: Reliability. Trade-offs: Startup can wait.
        Evidence: Internal: Launcher smoke polls ping. External: rclpy discovery API.
            Findings: Discovery is separate from model startup.
        Constraints: One peer role. Reconsider When: Multiple peers are supported.
        References: dispatch. ADR: None.
        """
        return all(pub.get_subscription_count()>0 for pub in pubs.values())

    def finish(message,operation):
        """Verify actual broker subscriber receipt and complete its matching RPC exactly once.

        Args: message: ROS String; operation: observation/actions. Returns: None. Raises: None.
        Design: Context: RPC thread awaits ROS delivery. Concern: Stale or changed content.
            Decision: Match wire identity and recompute received data hashes.
            Alternatives: Accepting a publish acknowledgement is insufficient.
            Quality Attributes: Integrity. Trade-offs: Invalid receipts fail the request.
        Evidence: Internal: Source and peer/actuator hashes are logged. External: None.
            Findings: No original arrays are returned on error.
        Constraints: Matched operation required. Reconsider When: Protocol changes.
        References: Handler.handle. ADR: None.
        """
        try:
            data=json.loads(message.data);identity=wire_id(data['id'])
            with lock:job=pending.get(identity)
            if job is None:return
            if job['request']['op']!=operation:raise ValueError('wrong receipt operation')
            if data.get('error'):raise ValueError(data['error'])
            received,parts=sensor_digest(data['sensors'],data['prompt']) if operation=='observation' else (chunk_digest(data['actions']),{})
            if received!=job['source'] or data['received_sha256']!=received:raise ValueError('ROS source/receiver hash mismatch')
            response={'ok':True,'id':identity,'request_id':job['request'].get('request_id'),'verified':True,'source_sha256':job['source'],'received_sha256':received,'peer_received_sha256':data['received_sha256'],'component_sha256':parts,'ros_roundtrip_seconds':time.monotonic()-job['dispatched'],'nodes':['realtime_sensor_bridge','realtime_policy_transport'],'topics':list(pubs)}
            if operation=='observation':response.update(sensors=data['sensors'],prompt=data['prompt'])
            else:response['actions']=data['actions']
            log.write('sensor_receipt' if operation=='observation' else 'action_receipt',id=identity,request_id=job['request'].get('request_id'),source_sha256=job['source'],received_sha256=received,verified=True,component_sha256=parts,ros_roundtrip_seconds=response['ros_roundtrip_seconds'])
        except Exception as error:
            log.write('receipt_error',error=str(error));response={'ok':False,'id':locals().get('identity'),'error':str(error)}
            if 'job' not in locals() or job is None:return
        with lock:
            pending.pop(identity,None);done[identity]=(job['source'],response)
            while len(done)>64:done.popitem(last=False)
            job['response']=response;job['event'].set()

    def receive_image(message):
        """Record one actual Image message and join by its header identity/stamp.

        Args: message: RGB Image. Returns: None. Raises: None; errors logged.
        Design: Context: Separate image topics. Concern: Pixel corruption or mixed frame pairs.
            Decision: Exact encoding/shape/step plus keyed join. Alternatives: Metadata images.
            Quality Attributes: Fidelity. Trade-offs: Fixed raw format.
        Evidence: Internal: Subscriber computes raw byte hashes. External: sensor_msgs/Image.
            Findings: No cv_bridge transformation occurs.
        Constraints: uint8RGB256. Reconsider When: Image encoding changes.
        References: emit_joined. ADR: None.
        """
        try:
            identity,camera=message.header.frame_id.split('|');wire_id(identity)
            if camera not in CAMERAS or (message.height,message.width,message.encoding,message.step)!=(256,256,'rgb8',768):raise ValueError('invalid ROS image format')
            raw=bytes(message.data)
            if len(raw)!=IMAGE_BYTES:raise ValueError('bad ROS image length')
            stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec
            join.add(identity,camera,raw,stamp);log.write('image_receive',id=identity,camera=camera,sha256=digest(raw),stamp_ns=stamp);emit_joined()
        except Exception as error:log.write('image_error',error=str(error))

    def emit_joined():
        """Publish acknowledged sensing rebuilt from actual subscriber image bytes.

        Args: None. Returns: None. Raises: Validation errors to callback.
        Design: Context: Complete sensor boundary. Concern: Original RPC payload echo.
            Decision: Receiver reconstruction then independent hash before ACK.
            Alternatives: Source hashes alone do not prove receipt.
            Quality Attributes: Traceability. Trade-offs: Base64 ACK serialization.
        Evidence: Internal: Complete join output. External: None. Findings: All three topics required.
        Constraints: Exact source hash match. Reconsider When: Typed service replaces ACK.
        References: receive_metadata. ADR: None.
        """
        for identity,sensors,metadata in join.take_ready():
            received,parts=sensor_digest(sensors,metadata['prompt'])
            if received!=metadata['source_sha256']:raise ValueError('joined sensor source mismatch')
            log.write('sensor_join_receive',id=identity,request_id=metadata.get('request_id'),source_sha256=metadata['source_sha256'],received_sha256=received,component_sha256=parts,verified=True)
            send('/realtime/sensor_ack',{'id':identity,'sensors':sensors,'prompt':metadata['prompt'],'received_sha256':received})

    def receive_metadata(message):
        """Join approved proprioception/prompt metadata with independently received cameras.

        Args: message: JSON String. Returns: None. Raises: None; error logged.
        Design: Context: Metadata can arrive first. Concern: Out-of-order joins.
            Decision: Shared ID/stamp. Alternatives: Time-nearest join.
            Quality Attributes: Determinism. Trade-offs: Missing message times out.
        Evidence: Internal: Join tests reorder components. External: None.
            Findings: Metadata never carries camera bytes or task truth.
        Constraints: Proprio allowlist. Reconsider When: Sensor schema changes.
        References: emit_joined. ADR: None.
        """
        try:
            data=json.loads(message.data);join.add(wire_id(data['id']),'metadata',data,data['stamp_ns']);emit_joined()
        except Exception as error:log.write('metadata_error',error=str(error))

    def receive_generated(message):
        """Forward only a valid chunk actually received by the policy-side ROS process.

        Args: message: generated chunk JSON. Returns: None. Raises: None; errors logged.
        Design: Context: Host GPU owns model computation. Concern: ROS action path bypass.
            Decision: PolicyTransportNode republishes received values to actuator-side topic.
            Alternatives: Broker directly acknowledges source chunk.
            Quality Attributes: Traceability. Trade-offs: Additional DDS exchange.
        Evidence: Internal: Both peer and broker recompute action hashes. External: std_msgs/String.
            Findings: This node is a transport adapter, not the model executor.
        Constraints: No clipping or rescaling. Reconsider When: GPU inference moves into ROS.
        References: finish. ADR: None.
        """
        try:
            data=json.loads(message.data);wire_id(data['id']);received=chunk_digest(data['actions'])
            if received!=data['source_sha256']:raise ValueError('generated chunk hash mismatch')
            log.write('generated_chunk_receive',id=data['id'],request_id=data.get('request_id'),source_sha256=data['source_sha256'],received_sha256=received,verified=True)
            send('/realtime/action_chunk',{'id':data['id'],'actions':data['actions'],'received_sha256':received})
        except Exception as error:log.write('chunk_error',error=str(error))

    def dispatch():
        """Publish queued RPC work in the ROS executor only after peer discovery.

        Args: None. Returns: None. Raises: None; request receives errors.
        Design: Context: TCP handler runs separately. Concern: Executor blocked by waits.
            Decision: Nonblocking bounded queue and timer. Alternatives: Spin inside handler.
            Quality Attributes: Responsiveness. Trade-offs: Timer adds at most5ms scheduling delay.
        Evidence: Internal: Parent physics is independent. External: rclpy timer.
            Findings: Only GPU worker waits for this RPC.
        Constraints: No unbounded backlog. Reconsider When: Multiple inference workers are added.
        References: Handler.handle. ADR: None.
        """
        if not ready():return
        try:job=jobs.get_nowait()
        except queue.Empty:return
        if job['event'].is_set():return
        data=job['request'];job['dispatched']=time.monotonic();identity=data['id']
        try:
            if data['op']=='observation':
                stamp=node.get_clock().now().to_msg();stamp_ns=stamp.sec*10**9+stamp.nanosec
                for camera in CAMERAS:
                    raw=image_bytes(data['sensors'][camera]);msg=Image();msg.header.stamp=stamp;msg.header.frame_id=identity+'|'+camera;msg.height=256;msg.width=256;msg.encoding='rgb8';msg.is_bigendian=0;msg.step=768;msg.data=raw;pubs['/realtime/'+camera].publish(msg)
                send('/realtime/sensor_metadata',{'id':identity,'request_id':data.get('request_id'),'stamp_ns':stamp_ns,'proprio':{k:data['sensors'][k] for k in PROPRIO},'prompt':data['prompt'],'source_sha256':job['source']})
            else:send('/realtime/chunk_generated',{'id':identity,'request_id':data.get('request_id'),'actions':data['actions'],'source_sha256':job['source']})
            log.write('sensor_publish' if data['op']=='observation' else 'generated_chunk_publish',id=identity,request_id=data.get('request_id'),source_sha256=job['source'])
        except Exception as error:
            with lock:pending.pop(identity,None)
            job['response']={'ok':False,'id':identity,'error':str(error)};job['event'].set()

    if args.role=='policy':
        for topic in ('/realtime/sensor_ack','/realtime/action_chunk'):pubs[topic]=node.create_publisher(String,topic,qos)
        for camera in CAMERAS:node.create_subscription(Image,'/realtime/'+camera,receive_image,qos)
        node.create_subscription(String,'/realtime/sensor_metadata',receive_metadata,qos)
        node.create_subscription(String,'/realtime/chunk_generated',receive_generated,qos)
    else:
        for camera in CAMERAS:pubs['/realtime/'+camera]=node.create_publisher(Image,'/realtime/'+camera,qos)
        for topic in ('/realtime/sensor_metadata','/realtime/chunk_generated'):pubs[topic]=node.create_publisher(String,topic,qos)
        node.create_subscription(String,'/realtime/sensor_ack',lambda m:finish(m,'observation'),qos)
        node.create_subscription(String,'/realtime/action_chunk',lambda m:finish(m,'actions'),qos)
        node.create_timer(.005,dispatch)

        class Handler(socketserver.StreamRequestHandler):
            """Serve one bounded RPC while ROS callbacks remain on their own executor.

            Args: Standard socketserver connection. Returns: One JSON line. Raises: Logged failures.
            Design: Context: Trusted loopback ABI boundary. Concern: Oversized or stalled requests.
                Decision:4MiB framing, finite120s wait and identity deduplication.
                Alternatives: Unbounded socket read. Quality Attributes: Boundedness. Trade-offs: One request per connection.
            Evidence: Internal: Actual node smoke. External: Python socketserver.
                Findings: Handler never substitutes original sensors after timeout.
            Constraints: Localhost only. Reconsider When: External deployment is requested.
            References: dispatch. ADR: None.
            """
            def handle(self):
                """Validate, enqueue and reply only after matching ROS subscriber receipt.

                Args: None. Returns: None. Raises: None; protocol errors become response objects.
                Design: Context: Worker blocks independently. Concern: Duplicate/stale IDs.
                    Decision: Exact digest match for completed replay; reject changed identity.
                    Alternatives: Blind republish. Quality Attributes: Integrity. Trade-offs: Bounded64-reply cache.
                Evidence: Internal: Synthetic actual ROS test. External: None.
                    Findings: Wire UUID differs from caller request ID.
                Constraints: Request timeout finite. Reconsider When: Cancellation is added.
                References: Handler. ADR: None.
                """
                identity=None
                try:
                    self.connection.settimeout(args.timeout)
                    line=self.rfile.readline(MAX_BYTES+1)
                    if len(line)>MAX_BYTES or not line.endswith(b'\n'):raise ValueError('incomplete/oversized request')
                    data=json.loads(line);identity=wire_id(data['id'])
                    if data['op']=='ping':response={'ok':True,'id':identity,'ready':ready(),'role':'broker','peer_node':'realtime_policy_transport'}
                    else:
                        if data['op']=='observation':source,_=sensor_digest(data['sensors'],data['prompt'])
                        elif data['op']=='actions':source=chunk_digest(data['actions'])
                        else:raise ValueError('unknown operation')
                        with lock:
                            if identity in done:
                                prior,response=done[identity]
                                if source!=prior:raise ValueError('changed completed request duplicate')
                                job=None
                            elif identity in pending:raise ValueError('duplicate active transport identity')
                            else:
                                job={'request':data,'source':source,'event':threading.Event(),'dispatched':time.monotonic()};pending[identity]=job
                        if job is not None:
                            jobs.put(job,timeout=1)
                            if not job['event'].wait(args.timeout):
                                with lock:pending.pop(identity,None)
                                job['event'].set();raise TimeoutError('actual ROS receipt deadline exceeded')
                            response=job['response']
                    payload=encoded(response)+b'\n'
                    if len(payload)>MAX_BYTES:raise ValueError('oversized response')
                except Exception as error:
                    log.write('rpc_error',id=identity,error=str(error));payload=encoded({'ok':False,'id':identity,'error':str(error)})+b'\n'
                try:self.wfile.write(payload)
                except (BrokenPipeError,ConnectionResetError):log.write('rpc_client_disconnected',id=identity)

        class Server(socketserver.ThreadingTCPServer):
            """Keep independent RPC waits bounded to trusted loopback.

            Args: Standard TCP server args. Returns: Server. Raises: Socket errors.
            Design: Context: ROS executor must keep spinning. Concern: Blocking TCP wait.
                Decision: Daemon request threads. Alternatives: Main-thread server.
                Quality Attributes: Responsiveness. Trade-offs: Local trusted clients only.
            Evidence: Internal: One inference worker. External: socketserver.
                Findings: ROS state updates use an explicit lock.
            Constraints: Backlog8. Reconsider When: Multi-client service needed.
            References: Handler. ADR: None.
            """
            allow_reuse_address=True;daemon_threads=True;request_queue_size=8
        server=Server(('127.0.0.1',args.port),Handler);thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.2},daemon=True);thread.start()
    log.write('node_started',role=args.role,node=node.get_name(),port=args.port if args.role=='broker' else None)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if server is not None:server.shutdown();server.server_close()
        with lock:
            for job in pending.values():job['response']={'ok':False,'id':job['request']['id'],'error':'bridge shutdown'};job['event'].set()
        log.write('node_stopped',role=args.role);node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


def main():
    """Validate CLI configuration and start exactly one ROS process role.

    Args: CLI. Returns: None. Raises: Invalid arguments/startup errors.
    Design: Context: Launcher starts broker and policy separately. Concern: Ambiguous runtime ownership.
        Decision: Explicit role and output. Alternatives: In-process fake peer.
        Quality Attributes: Reproducibility. Trade-offs: Two processes.
    Evidence: Internal: run_ros_bridge.sh. External: ROS2 node graph.
        Findings: Host GPU model remains separate.
    Constraints: Positive finite timeout and valid port. Reconsider When: Deployment changes.
    References: run_node. ADR: None.
    """
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('role',choices=['broker','policy']);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--port',type=int,default=8780);parser.add_argument('--timeout',type=float,default=120);args=parser.parse_args()
    if not 0<args.port<65536 or not math.isfinite(args.timeout) or args.timeout<=0:parser.error('invalid port/timeout')
    run_node(args)


if __name__=='__main__':main()
