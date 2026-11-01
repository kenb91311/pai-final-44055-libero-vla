"""Verify sensor joins and the actual two-process ROS bridge independently of model scores.

Args: unittest by default; --integration --output PATH runs actual ROS exchanges.
Returns: Unit results or durable synthetic-only transport checks.
Raises: AssertionError on corrupted/mixed sensors, changed duplicates or mismatched actions.
Design:
    Context: Real model performance must be separated from middleware plumbing tests.
    Concern: A mocked RPC cannot prove actual ROS topic transit.
    Decision: Pure edge-case tests plus explicitly labelled actual ROS integration smoke.
    Alternatives: Count synthetic transport as task success, which is invalid.
    Quality Attributes: Integrity. Trade-offs: Integration requires the ROS launcher.
Evidence:
    Internal: Broker/peer JSONL includes source and independently received hashes.
    External: None. Findings: Sensor allowlist and exact action values are checked.
Constraints: No GPU model or physical task score in this test.
Reconsider When: Wire schema or ROS runtime changes.
References: ros_bridge.py and run_ros_bridge.sh. ADR: None.
"""
import argparse
import base64
import json
from pathlib import Path
import socket
import time
import unittest
import numpy as np
try:
    from .ros_bridge import CAMERAS,PROPRIO,JoinedSensors,chunk_digest,sensor_digest
    from .ros_client import RosRoundTrip,sensor_roundtrip,chunk_roundtrip
except ImportError:
    from ros_bridge import CAMERAS,PROPRIO,JoinedSensors,chunk_digest,sensor_digest
    from ros_client import RosRoundTrip,sensor_roundtrip,chunk_roundtrip


def fixture():
    """Build distinct deterministic RGB cameras and valid raw proprioception.

    Args: None. Returns: Synthetic raw observation. Raises: None.
    Design: Context: Transport-only smoke. Concern: Identical cameras hide joins.
        Decision: Different pixel patterns. Alternatives: All-zero frames.
        Quality Attributes: Detectability. Trade-offs: No task realism.
    Evidence: Internal: Exact equality assertions. External: None. Findings: No pretrained model is called.
    Constraints: RGB256 and state3/4/2. Reconsider When: Schema changes.
    References: integration. ADR: None.
    """
    image=np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3)
    return {CAMERAS[0]:image,CAMERAS[1]:255-image,PROPRIO[0]:np.array([.1,.2,.3]),PROPRIO[1]:np.array([0.,0.,0.,1.]),PROPRIO[2]:np.array([.02,.03]),'success':True,'object_truth':{'never':'send'}}


class JoinTests(unittest.TestCase):
    """Exercise ordering, stale identity and privileged-field failures before real ROS.

    Args: unittest framework. Returns: Assertions. Raises: AssertionError.
    Design: Context: Cross-topic concurrency. Concern: False complete observations.
        Decision: Test missing/reordered and changed duplicate components.
        Alternatives: Happy-path shape test only. Quality Attributes: Correctness. Trade-offs: Small fixed schema.
    Evidence: Internal: JoinedSensors pure boundary. External: None. Findings: Completed joins deduplicate.
    Constraints: Middleware tested separately. Reconsider When: Join design changes.
    References: integration. ADR: None.
    """
    def test_reordered_and_duplicate(self):
        """Require both cameras plus metadata and reject changed completed duplicates.

        Args: None. Returns: None. Raises: AssertionError.
        Design: Context: ROS topic reorder. Concern: Mixed frames. Decision: Identity join assertions.
            Alternatives: Arrival order. Quality Attributes: Integrity. Trade-offs: Synthetic fixtures.
        Evidence: Internal: Missing camera yields no output. External: None. Findings: Exact duplicates emit once.
        Constraints: No ROS claim from unit test. Reconsider When: Schema changes.
        References: JoinedSensors. ADR: None.
        """
        join=JoinedSensors();meta={'proprio':{k:[0.]*(n) for k,n in zip(PROPRIO,(3,4,2),strict=True)},'prompt':'synthetic'}
        join.add('a','metadata',meta,9);join.add('a',CAMERAS[1],b'b'*196608,9);self.assertEqual(join.take_ready(),[])
        join.add('a',CAMERAS[0],b'a'*196608,9);self.assertEqual(len(join.take_ready()),1)
        join.add('a',CAMERAS[0],b'a'*196608,9);self.assertEqual(join.take_ready(),[])
        with self.assertRaises(ValueError):join.add('a',CAMERAS[0],b'c'*196608,9)

    def test_stamp_and_allowlist(self):
        """Reject mixed timestamps and extra simulator truth before inference.

        Args: None. Returns: None. Raises: AssertionError.
        Design: Context: Raw simulator has privileged fields. Concern: Leakage.
            Decision: Exact allowlist rejects extras. Alternatives: Entire dict transport.
            Quality Attributes: Auditability. Trade-offs: New modalities require updates.
        Evidence: Internal: sensor_digest validation. External: None. Findings: NaN chunks also rejected.
        Constraints: Finite inputs. Reconsider When: Policy modalities change.
        References: sensor_digest. ADR: None.
        """
        join=JoinedSensors();join.add('a',CAMERAS[0],b'a',1)
        with self.assertRaises(ValueError):join.add('a',CAMERAS[1],b'b',2)
        with self.assertRaises(ValueError):sensor_digest({'success':True},'task')
        with self.assertRaises(ValueError):chunk_digest([[float('nan')]*7])


def raw_rpc(payload,port):
    """Send an explicit wire identity to exercise real broker replay protection.

    Args: payload: JSON; port: local bridge. Returns: Response. Raises: Socket errors.
    Design: Context: Public client normally generates UUIDs. Concern: Duplicate behavior untested.
        Decision: Direct trusted loopback request solely in transport test.
        Alternatives: Mock deduplication. Quality Attributes: Coverage. Trade-offs: Test-only low-level RPC.
    Evidence: Internal: Changed duplicate must return okFalse. External: TCP JSON.
        Findings: This bypasses no ROS for the initial request.
    Constraints: No model metrics. Reconsider When: Protocol changes.
    References: integration. ADR: None.
    """
    with socket.create_connection(('127.0.0.1',port),timeout=10) as connection:
        connection.sendall(json.dumps(payload,allow_nan=False).encode()+b'\n')
        return json.loads(connection.makefile('rb').readline(4*1024*1024+1))


def integration(port,output):
    """Verify actual ROS image/chunk receipts and save synthetic-only provenance.

    Args: port: bridge; output: JSON result path. Returns: None. Raises: AssertionError/errors.
    Design: Context: Run before pretrained policy evaluation. Concern: Middleware failure mistaken for model.
        Decision: Three camera pairs, both model horizons, exact equality and dedup failure.
        Alternatives: Count synthetic results as task success.
        Quality Attributes: Traceability. Trade-offs: Requires existing Docker image.
    Evidence: Internal: Both actual ROS nodes log matching source/received hashes.
        External: None. Findings: Privileged fixture keys are absent from received inputs.
    Constraints: No task accuracy claim. Reconsider When: ROS topology changes.
    References: run_ros_bridge.sh. ADR: None.
    """
    result={'status':'running','input_kind':'synthetic','model_performance_evidence':False,'port':port,'sensors':[],'chunks':[]}
    try:
        deadline=time.monotonic()+45
        while True:
            try:
                if RosRoundTrip(port=port,timeout=1).ping().get('ready'):break
            except (OSError,RuntimeError):pass
            if time.monotonic()>deadline:raise TimeoutError('bridge discovery timeout')
            time.sleep(.2)
        for index in range(3):
            obs=fixture();obs[CAMERAS[0]]=obs[CAMERAS[0]]^index
            received,prompt,audit=sensor_roundtrip(obs,'synthetic transport only',f'synthetic-{index}',port)
            assert set(received)==set(CAMERAS+PROPRIO) and prompt=='synthetic transport only'
            for key in CAMERAS+PROPRIO:np.testing.assert_array_equal(received[key],obs[key])
            result['sensors'].append(audit)
        for length in (10,50):
            chunk=np.random.default_rng(length).normal(size=(length,7)).astype(np.float32)
            received,audit=chunk_roundtrip(chunk,f'synthetic-chunk-{length}',port);np.testing.assert_array_equal(received,chunk)
            result['chunks'].append(audit)
        payload={'op':'actions','id':'dedup-'+str(time.time_ns()),'request_id':'synthetic-duplicate','actions':[[.125]*7]}
        first=raw_rpc(payload,port);replay=raw_rpc(payload,port);assert first['ok'] and replay==first
        payload['actions'][0][0]=.5;changed=raw_rpc(payload,port);assert changed['ok'] is False
        result['dedup_identical_replayed']=True;result['dedup_changed_rejected']=True;result['status']='passed'
    except BaseException as error:
        result['status']='error';result['error']=f'{type(error).__name__}: {error}';raise
    finally:
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'output':str(output),'error':result.get('error')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--integration',action='store_true');parser.add_argument('--port',type=int,default=8780);parser.add_argument('--output',type=Path,default=Path('artifacts/realtime-vla/ros-smoke/transport-checks.json'));args,unknown=parser.parse_known_args()
    if args.integration:integration(args.port,args.output)
    else:unittest.main(argv=[__file__]+unknown)
