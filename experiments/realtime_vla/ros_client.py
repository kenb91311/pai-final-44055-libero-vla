"""Round-trip frozen-policy inputs and chunks through actual ROS2 nodes.

Args: Local bridge port and request timeout.
Returns: Sensor arrays/action chunks reconstructed exclusively from ROS receipts.
Raises: RuntimeError on stale/malformed/unverified topic data or transport failure.
Design:
    Context: Host CUDA/physics and ROSJazzy remain in separate dependency environments.
    Concern: A metadata-only ROS side channel would bypass the demonstrated architecture.
    Decision: Return actual received two-camera/proprio and chunk bytes to the caller.
    Alternatives: Publishing telemetry while consuming original inputs is insufficient.
    Quality Attributes: Traceability, isolation. Trade-offs: Extra serialization and DDS latency.
Evidence:
    Internal: Bridge records exact source/received SHA256 for every exchange.
    External: ROS2 sensor_msgs/Image and std_msgs/String.
    Findings: Worker RPC waits do not block the independently scheduled physics parent.
    Metrics: DDS round-trip wall time and payload identity.
    Confidence: Requires actual Docker smoke, not a mocked transport.
Constraints: Trusted loopback only; no evaluator object poses in sensor messages.
Reconsider When: ROS is installed directly with the model ABI.
References: ros_bridge.py and run_ros_bridge.sh.
ADR: None.
"""
import base64
import json
import socket
import time
import uuid
import numpy as np
try:
    from .ros_bridge import sensor_digest,chunk_digest
except ImportError:
    from ros_bridge import sensor_digest,chunk_digest

CAMERAS=('agentview_image','robot0_eye_in_hand_image')
PROPRIO=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


class RosRoundTrip:
    """Provide a synchronous worker-only boundary around two ROS node roles.

    Args: port: localhost service; timeout: seconds.
    Returns: Client object. Raises: Connection/validation errors on an exchange.
    Design:
        Context: Inference already runs outside the main physics loop.
        Concern: RPC waiting must stay inside that inference worker.
        Decision: Small independent sensor and action exchanges with checked identity.
        Alternatives: Putting the client in env.step would pause physical time.
        Quality Attributes: Correctness. Trade-offs: Inference includes transport overhead.
    Evidence:
        Internal: Every call has a unique request ID.
        External: TCP and ROS2 reliable QoS.
            Findings: Source and received hashes are also independently recomputed here.
        Metrics: receipt hashes and measured durations.
        Confidence: Integration smoke.
    Constraints: One worker; local trusted bridge.
    Reconsider When: Multiple GPU workers are introduced.
    References: Module contract.
    ADR: None.
    """
    def __init__(self,port=8780,timeout=120):
        """Retain local connection settings without starting ROS or changing physics.

        Args: port/timeout: connection settings. Returns: None. Raises: ValueError.
        Design: Context: Worker owns client. Concern: Invalid endpoints. Decision: Validate.
            Alternatives: Implicit remote endpoints. Quality Attributes: Isolation.
            Trade-offs: Loopback only.
        Evidence: Internal: Bridge localhost binding. External: None.
            Findings: No startup side effects. Metrics: None. Confidence: Contract.
        Constraints: Positive timeout and valid port.
        Reconsider When: Remote deployment is required.
        References: RosRoundTrip. ADR: None.
        """
        self.port=int(port);self.timeout=float(timeout)
        if not 0<self.port<65536 or self.timeout<=0:raise ValueError('invalid bridge endpoint')

    def _request(self,payload):
        """Send a bounded request and require verified received content.

        Args: payload: finite JSON mapping. Returns: Bridge response. Raises: RuntimeError.
        Design: Context: Cross-ABI transport. Concern: Missing/partial replies.
            Decision: Bounded line framing with finite deadlines.
            Alternatives: Pickle across Docker. Quality Attributes: Integrity.
            Trade-offs: Base64 overhead.
        Evidence: Internal: Source/received hashes in reply. External: JSON/TCP.
            Findings: Failure never substitutes original data.
            Metrics: request wall seconds. Confidence: Integration smoke.
        Constraints: At most4MiB per message.
        Reconsider When: Image resolution increases.
        References: ros_bridge.py. ADR: None.
        """
        payload=dict(payload,id=uuid.uuid4().hex)
        start=time.monotonic()
        with socket.create_connection(('127.0.0.1',self.port),timeout=self.timeout) as conn:
            conn.settimeout(self.timeout)
            conn.sendall(json.dumps(payload,allow_nan=False,separators=(',',':')).encode()+b'\n')
            response=conn.makefile('rb').readline(4*1024*1024+1)
        if not response.endswith(b'\n') or len(response)>4*1024*1024:raise RuntimeError('incomplete/oversized ROSbridge response')
        result=json.loads(response)
        if not result.get('ok') or result.get('id')!=payload['id']:raise RuntimeError(f"ROSbridge rejected exchange: {result.get('error')}")
        if payload['op']!='ping' and result.get('verified') is not True:raise RuntimeError('ROS source/receipt mismatch')
        if payload['op']=='observation':
            source,_=sensor_digest(payload['sensors'],payload['prompt'])
            received,_=sensor_digest(result['sensors'],result['prompt'])
        elif payload['op']=='actions':source=chunk_digest(payload['actions']);received=chunk_digest(result['actions'])
        else:source=received=None
        if payload['op']!='ping' and not source==received==result.get('source_sha256')==result.get('received_sha256'):
            raise RuntimeError('client independently detected ROS content mismatch')
        result['transport_id']=payload['id']
        result['rpc_wall_seconds']=time.monotonic()-start
        return result

    def observation(self,obs,request_id=None):
        """Send approved raw sensing and return the actual joined ROS receipts.

        Args: obs: raw LIBERO sensors plus prompt; request_id: caller correlation.
        Returns: (received_obs, audit).
        Raises: ValueError on invalid sensors; RuntimeError on transport failure.
        Design: Context: Official model adapters require raw orientation/quaternion.
            Concern: Privileged simulator state must not enter model inputs.
            Decision: Allowlist two RGBs and three proprioceptive fields only.
            Alternatives: Forwarding evaluator metrics. Quality Attributes: Auditability.
            Trade-offs: Extra camera copies.
        Evidence: Internal: Exact image SHA256 across node pair.
            External: Official LIBERO sensor contract.
            Findings: No image flip or state conversion in transport.
            Metrics: DDS and RPC timing. Confidence: Integration smoke.
        Constraints: uint8 RGB256x256, finite proprio3/4/2.
        Reconsider When: Model camera schema changes.
        References: FrozenLiberoEnv. ADR: None.
        """
        sensors={}
        for key in CAMERAS:
            value=np.asarray(obs[key])
            if value.shape!=(256,256,3) or value.dtype!=np.uint8:raise ValueError('ROS bridge requires RGB256 uint8')
            sensors[key]={'shape':[256,256,3],'dtype':'uint8','data':base64.b64encode(value.tobytes()).decode()}
        for key,n in zip(PROPRIO,(3,4,2),strict=True):
            value=np.asarray(obs[key])
            if value.shape!=(n,) or not np.isfinite(value).all():raise ValueError('invalid proprioception')
            sensors[key]=value.tolist()
        result=self._request({'op':'observation','sensors':sensors,'prompt':str(obs['prompt']),'request_id':request_id})
        received=result.pop('sensors');output={key:np.asarray(received[key]) for key in PROPRIO}
        for key in CAMERAS:
            record=received[key];raw=base64.b64decode(record['data'],validate=True)
            output[key]=np.frombuffer(raw,dtype=np.uint8).reshape(record['shape']).copy()
        output['prompt']=result.pop('prompt')
        return output,result

    def actions(self,actions,request_id=None):
        """Return only the chunk received by the ROS actuator-side subscriber.

        Args: actions: finite float chunk[N,7]; request_id: caller correlation.
        Returns: (received_actions,audit).
        Raises: ValueError for malformed chunks; RuntimeError for unverified transport.
        Design: Context: Model returns normalized relative OSC commands.
            Concern: Telemetry publication alone does not put ROS in the action path.
            Decision: Caller consumes chunk reconstructed from the subscriber.
            Alternatives: Returning original action object. Quality Attributes: Traceability.
            Trade-offs: One extra local exchange per chunk.
        Evidence: Internal: Canonical action JSON hashes across both nodes.
            External: ROS2 std_msgs/String.
            Findings: No clipping/rescaling before the common actuator controller.
            Metrics: DDS/RPC duration. Confidence: Integration smoke.
        Constraints: Seven columns, finite values, at most1000 rows.
        Reconsider When: Absolute command representation changes.
        References: ros_bridge.py. ADR: None.
        """
        array=np.asarray(actions,dtype=np.float64)
        if array.ndim!=2 or array.shape[1]!=7 or not 0<len(array)<=1000 or not np.isfinite(array).all():raise ValueError('invalid action chunk')
        result=self._request({'op':'actions','actions':array.tolist(),'request_id':request_id})
        received=np.asarray(result.pop('actions'),dtype=np.float64)
        return received,result

    def ping(self):
        """Check that both ROS roles and their subscriptions are ready.

        Args: None. Returns: Ready response. Raises: Connection/RuntimeError.
        Design: Context: DDS discovery precedes task time. Concern: Startup losses.
            Decision: Readiness flag reports actual publisher/subscriber matching.
            Alternatives: Guessing a sleep duration. Quality Attributes: Reliability.
            Trade-offs: Startup can fail explicitly.
        Evidence: Internal: Publisher subscription counts. External: ROS2 discovery.
            Findings: Ping does not publish policy inputs.
            Metrics: Readiness. Confidence: Integration smoke.
        Constraints: No task-time startup.
        Reconsider When: Discovery topology changes.
        References: run_ros_bridge.sh. ADR: None.
        """
        return self._request({'op':'ping'})


def sensor_roundtrip(observation,prompt,request_id,port=8780):
    """Return only approved sensing reconstructed after two actual ROS image topics.

    Args: observation: raw LIBERO sensors; prompt: task; request_id: caller correlation; port: local RPC.
    Returns: Received raw observation, received prompt and transport evidence.
    Raises: ValueError on malformed sensing; RuntimeError on unverified/failed ROS exchange.
    Design:
        Context: GPU worker waits while the parent physical clock continues.
        Concern: Original sensor references must not bypass ROS transport.
        Decision: Explicit raw allowlist plus unique transport identity on every RPC.
        Alternatives: Publishing telemetry while passing original arrays is insufficient.
        Quality Attributes: Integrity. Trade-offs: Base64/TCP/DDS overhead is measured.
    Evidence:
        Internal: Broker and peer independently hash received images/proprioception.
        External: ROS2 sensor_msgs/Image.
        Findings: Caller IDs can reset across model workers; wire UUIDs remain unique.
    Constraints: Raw orientation; no simulator truth; timeout120s.
    Reconsider When: Direct ROS Python bindings share the CUDA ABI.
    References: ros_bridge.py. ADR: None.
    """
    raw,audit=RosRoundTrip(port=port).observation(dict(observation,prompt=prompt),request_id=request_id)
    received_prompt=raw.pop('prompt')
    return raw,received_prompt,audit


def chunk_roundtrip(actions,request_id,port=8780):
    """Return the action chunk actually received by the actuator-side ROS subscriber.

    Args: actions: finite Nx7 actions; request_id: caller correlation; port: local RPC.
    Returns: Received action array and transport evidence.
    Raises: ValueError or RuntimeError on invalid/unverified exchange.
    Design:
        Context: Generated actions must pass through PolicyTransportNode.
        Concern: A successful publication does not prove actuator-side receipt.
        Decision: Wait for matching subscriber hash and return its values.
        Alternatives: Returning the original chunk would bypass ROS.
        Quality Attributes: Traceability. Trade-offs: One serialized roundtrip per chunk.
    Evidence:
        Internal: generated and received canonical action hashes must agree.
        External: ROS2 std_msgs/String. Findings: No clipping/rescaling occurs in transport.
    Constraints: At most1000 actions; timeout120s.
    Reconsider When: A custom trajectory message replaces JSON.
    References: ros_bridge.py. ADR: None.
    """
    return RosRoundTrip(port=port).actions(actions,request_id=request_id)
