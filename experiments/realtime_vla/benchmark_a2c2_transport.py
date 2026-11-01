"""Measure7D action RPC transport independently of GPU inference and physical control.

Args: New output directory and existing shared ROS bridge port.
Returns: Per-call client/DDS timing, errors and persistent-connection capability probe.
Raises: Output/transport startup errors; individual timeouts are retained as diagnostic data.
Design: Context: A task stopped after DDS receipt but before TCP response reached its client.
    Concern: A control timeout is not proof of slow model inference or slow DDS.
    Decision: Fixed7D exchanges at2/20Hz, client TCP_NODELAY comparison and no model loading.
    Alternatives: Increase task deadlines or edit the shared bridge before diagnosis.
    Quality Attributes: Isolation, traceability. Trade-offs: Dispatch timer is included in RPC.
Evidence: Internal: Failed A2C2 tick57; independent DDS receipt0.893ms.
    External: Python socket API. Findings: Current server handles one request per connection.
    Metrics: RPC/DDS/difference p50/p95/max, tails and timeouts.
    Confidence: Small diagnostic series, not task success evidence.
Constraints: No GPU, simulator, model, shared bridge/source changes or retries masking errors.
Reconsider When: Transport is changed. References: ros_client.py/ros_bridge.py. ADR: None.
"""
import argparse
import json
from pathlib import Path
import socket
import time
import uuid
from unittest.mock import patch
import numpy as np
if __package__:
    from . import ros_client
else:
    import ros_client


def stats(values):
    """Describe finite diagnostic samples without silently filling missing responses.

    Args: values: milliseconds. Returns: Count/mean/p50/p95/max or empty count.
    Raises: Numeric errors. Design: Context: Timeouts lack DDS responses.
        Concern: Treating missing as zero biases tails. Decision: Separate count.
        Alternatives: Zero filling. Quality Attributes: Accuracy. Trade-offs: Small samples.
    Evidence: Internal: Raw calls. External: NumPy quantiles.
        Findings: Failures counted elsewhere. Metrics: Descriptive only. Confidence: Empirical.
    Constraints: No extrapolation. Reconsider When: Larger studies. References: phase. ADR: None.
    """
    if not values:
        return {'count':0}
    return {'count':len(values), 'mean':float(np.mean(values)), 'p50':float(np.median(values)),
            'p95':float(np.percentile(values, 95)), 'max':float(np.max(values))}


def phase(client, name, frequency, count, nodelay, output):
    """Measure one serial7D request phase, retaining independent ROS metadata per call.

    Args: client/name/frequency/count/nodelay/output: declared phase settings.
    Returns: Diagnostic summary. Raises: Output errors; call errors logged and continued.
    Design: Context: Per-tick action transport uses one connection per request.
        Concern: Server dispatch wait and DDS are different from socket response delivery.
        Decision: Record client start/end and RPC versus DDS duration separately.
        Alternatives: Label all RPC overhead TCP. Quality Attributes: Attribution.
        Trade-offs: Difference includes serialization, server timer and TCP.
    Evidence: Internal: Receipt hashes returned by client. External: TCP_NODELAY.
        Findings: Client NODELAY does not change server or enable persistent RPC.
        Metrics: All raw times. Confidence: Diagnostic only.
    Constraints: No model/physics execution. Reconsider When: Protocol changes.
    References: RosRoundTrip.actions. ADR: None.
    """
    original_connect = socket.create_connection

    def connect(*args, **kwargs):
        """Set only the client socket's Nagle switch for this isolated benchmark process.

        Args: Standard create_connection args. Returns: Socket. Raises: Socket errors.
        Design: Context: Compare current client against NODELAY.
            Concern: Shared service must remain unchanged. Decision: Local process wrapper.
            Alternatives: Global source edits. Quality Attributes: Isolation. Trade-offs: Client only.
        Evidence: Internal: Exact flag retained. External: TCP_NODELAY.
            Findings: Server unchanged. Metrics: None. Confidence: API contract.
        Constraints: Restored after phase. Reconsider When: Server changes.
        References: phase. ADR: None.
        """
        connection = original_connect(*args, **kwargs)
        if nodelay:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection

    rows, next_call = [], time.monotonic()
    with patch.object(ros_client.socket, 'create_connection', side_effect=connect), (output/(name+'.jsonl')).open('w') as stream:
        for index in range(count):
            delay = next_call-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            request = f'a2c2-transport-{name}-{index}-{uuid.uuid4().hex[:8]}'
            started_wall, started = time.time(), time.perf_counter()
            try:
                received, metadata = client.actions(np.array([[.001*index, 0., 0., 0., 0., 0., -1.]]), request_id=request)
                row = {'index':index, 'request_id':request, 'status':'received', 'metadata':metadata,
                       'received_action':received.tolist()}
            except Exception as error:
                row = {'index':index, 'request_id':request, 'status':'error', 'error':f'{type(error).__name__}: {error}'}
            row.update(started_wall_time=started_wall, finished_wall_time=time.time(),
                       elapsed_ms=(time.perf_counter()-started)*1000)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            next_call = max(next_call+1/frequency, time.monotonic())
    good = [r for r in rows if r['status'] == 'received']
    rpc = [r['elapsed_ms'] for r in good]
    dds = [r['metadata']['ros_roundtrip_seconds']*1000 for r in good]
    gap = [r['elapsed_ms']-r['metadata']['ros_roundtrip_seconds']*1000 for r in good]
    return {'name':name, 'frequency_hz':frequency, 'calls':count, 'client_tcp_nodelay':nodelay,
            'errors':len(rows)-len(good), 'rpc_ms':stats(rpc), 'dds_ms':stats(dds),
            'rpc_minus_dds_ms':stats(gap), 'rpc_ge35ms':sum(v >= 35 for v in rpc),
            'rpc_ge100ms':sum(v >= 100 for v in rpc), 'error_rows':[r for r in rows if r['status'] != 'received']}


def persistent_probe(port):
    """Test whether the existing unmodified broker accepts two requests on one connection.

    Args: port: existing broker. Returns: Capability result and explicit failure/EOF.
    Raises: Initial connection errors. Design: Context: Current Handler reads one line only.
        Concern: A persistent client alone may silently fail on its second request.
        Decision: Two ping messages on one socket, no action execution.
        Alternatives: Assume keepalive. Quality Attributes: Compatibility. Trade-offs: Probe only.
    Evidence: Internal: Shared server implementation. External: socketserver.
        Findings: Lifecycle must be changed on both sides. Metrics: Response count.
        Confidence: Direct capability observation.
    Constraints: No shared service edits. Reconsider When: Broker protocol changes.
    References: Handler.handle. ADR: None.
    """
    received = []
    error = None
    with socket.create_connection(('127.0.0.1', port), timeout=1) as connection:
        connection.settimeout(1)
        with connection.makefile('rb') as stream:
            for _ in range(2):
                try:
                    connection.sendall(json.dumps({'op':'ping', 'id':uuid.uuid4().hex}).encode()+b'\n')
                    raw = stream.readline()
                    if not raw:
                        error = 'EOF: broker closed connection after first response'
                        break
                    received.append(json.loads(raw))
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    break
    return {'responses':len(received), 'two_requests_supported':len(received) == 2, 'error':error}


def main():
    """Run an explicit CPU-only action transport diagnostic against the existing ROS service.

    Args: CLI output/port. Returns: None and diagnostic JSON/JSONL.
    Raises: Startup/IO errors. Design: Context: Separate transport from GPU conclusions.
        Concern: Benchmarks must not overwrite task evidence.
        Decision: New directory and no model imports. Alternatives: Modify task timeout.
        Quality Attributes: Isolation. Trade-offs: About25seconds shared CPU transport load.
    Evidence: Internal: Raw timings/metadata. External: Existing ROS roles.
        Findings: No physical task commands are applied. Metrics: Phase summaries.
        Confidence: Diagnostic. Constraints: No GPU or bridge mutation.
    Reconsider When: Transport changes. References: phase/persistent_probe. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8780)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    client = ros_client.RosRoundTrip(port=args.port, timeout=1)
    ping = client.ping()
    result = {'gpu_used':False, 'physics_stepped':False, 'server_modified':False,
              'diagnostic_timeout_s':1, 'task_timeout_unchanged_s':.5, 'ready':ping,
              'persistent_probe':persistent_probe(args.port), 'phases':[]}
    for name, hz, count, nodelay in [('default-2hz', 2, 20, False), ('default-20hz', 20, 128, False),
                                     ('client-nodelay-20hz', 20, 128, True)]:
        row = phase(client, name, hz, count, nodelay, args.output)
        result['phases'].append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
