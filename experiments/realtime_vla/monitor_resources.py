"""Record low-overhead host/GPU resource evidence during coordinated benchmarks.

Args: New output directory, optional phase/stop files, period and sample limit.
Returns: UTC/monotonic JSONL with whole-GPU usage and host memory/load.
Raises: NVML/IO errors explicitly; never reports a failed probe as idle hardware.
Design: Context: Another authorized experiment can share the desktop GPU.
    Concern: One model in this process tree does not establish global GPU exclusivity.
    Decision: Coordinate model residency separately and continuously sample NVML via ctypes.
    Alternatives: Repeated heavyweight command launches or unverified exclusive claims.
    Quality Attributes: Observability, low interference. Trade-offs: WSL process lists are unavailable.
Evidence: Internal: First RTC series had a concurrent-model observation.
    External: NVIDIA NVML memory/utilization APIs.
    Findings: Whole-device usage includes rendering/desktop and cannot attribute individual kernels.
    Metrics: Memory, utilization, host availability and timestamps. Confidence: Sampled evidence only.
Constraints: Read-only; no model imports, process termination or GPU reservation.
Reconsider When: MIG/process accounting or a dedicated benchmark host is available.
References: gpu-contention-observation.json. ADR: None.
"""
import argparse
import ctypes as C
import ctypes.util
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time


class NvmlProbe:
    """Read device0 NVML memory and utilization without a CUDA or torch context.

    Args: index: physical device index. Returns: Probe. Raises: RuntimeError/OSError.
    Design: Context: Repeated sampling should not compete with model processes.
        Concern: nvidia-smi subprocess startup adds avoidable CPU work.
        Decision: Small documented NVML ABI. Alternatives: Import model libraries.
        Quality Attributes: Low overhead. Trade-offs: NVIDIA-specific.
    Evidence: Internal: Runtime smoke against nvidia-smi. External: NVML C API.
        Findings: No GPU memory is allocated. Metrics: Sample call duration.
        Confidence: Bound to supported API return codes.
    Constraints: No per-process attribution. Reconsider When: NVML ABI changes.
    References: monitor_resources.py. ADR: None.
    """
    class Memory(C.Structure):
        """NVML legacy memory record: total/free/used uint64 bytes.

        Args: None. Returns: ABI record. Raises: None.
        Design: Context: Stable NVML ABI. Concern: Width/order mismatches.
            Decision: Exact three uint64 fields. Alternatives: Guessing textual units.
            Quality Attributes: Correctness. Trade-offs: No reserved-memory field.
        Evidence: Internal: total/free/used sanity check. External: nvmlMemory_t.
            Findings: Byte counts. Metrics: None. Confidence: ABI.
        Constraints: Read-only. Reconsider When: ABI changes.
        References: NvmlProbe. ADR: None.
        """
        _fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]

    class Utilization(C.Structure):
        """NVML utilization record in integer percent.

        Args: None. Returns: ABI record. Raises: None.
        Design: Context: Whole-device samples. Concern: Misstated scope.
            Decision: Preserve gpu/memory percent. Alternatives: Infer per-model utilization.
            Quality Attributes: Honesty. Trade-offs: Driver-defined sampling window.
        Evidence: Internal: Range checks. External: nvmlUtilization_t.
            Findings: Not instantaneous per-kernel timing. Metrics: None. Confidence: ABI.
        Constraints: No process inference. Reconsider When: API changes.
        References: NvmlProbe. ADR: None.
        """
        _fields_=[('gpu',C.c_uint),('memory',C.c_uint)]

    def __init__(self,index=0):
        """Initialize a read-only NVML handle and fail if the device is unavailable.

        Args: index: device. Returns: None. Raises: OSError/RuntimeError.
        Design: Context: WSL library may be outside default loader paths.
            Concern: Silent CPU fallback would fabricate idle GPU evidence.
            Decision: Known WSL path or system lookup; explicit status checks.
            Alternatives: Return zeros. Quality Attributes: Integrity. Trade-offs: Fail closed.
        Evidence: Internal: Driver return codes. External: NVML init_v2.
            Findings: No CUDA context. Metrics: None. Confidence: API response.
        Constraints: One owned handle. Reconsider When: MultiGPU benchmark.
        References: sample. ADR: None.
        """
        path=C.util.find_library('nvidia-ml') or '/usr/lib/wsl/lib/libnvidia-ml.so.1'
        self.lib=C.CDLL(path);self.check(self.lib.nvmlInit_v2())
        self.handle=C.c_void_p();self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(C.c_uint(index),C.byref(self.handle)))

    @staticmethod
    def check(code):
        """Require NVML_SUCCESS without substituting a zero measurement.

        Args: code: C return status. Returns: None. Raises: RuntimeError otherwise.
        Design: Context: Measurements guide scheduling. Concern: Hidden read failures.
            Decision: Explicit rejection. Alternatives: Best-effort zeros.
            Quality Attributes: Integrity. Trade-offs: Probe failure stops monitor.
        Evidence: Internal: Return status. External: NVML_SUCCESS=0.
            Findings: No measurement on failure. Metrics: None. Confidence: Contract.
        Constraints: No retries concealing gaps. Reconsider When: Retry policy is explicit.
        References: NvmlProbe. ADR: None.
        """
        if code!=0:raise RuntimeError(f'NVML status {code}')

    def sample(self):
        """Read whole-device counters, preserving bytes and scope.

        Args: None. Returns: Mapping. Raises: NVML/sanity error.
        Design: Context: Coordinated experiments plus desktop rendering.
            Concern: Global usage is not model-allocated memory.
            Decision: Explicit whole_gpu field names. Alternatives: Claim torch peak equivalence.
            Quality Attributes: Clarity. Trade-offs: Sampled windows.
        Evidence: Internal: total/free/used check. External: NVML memory/utilization.
            Findings: Desktop is included. Metrics: Bytes/percent. Confidence: Driver sample.
        Constraints: Read-only. Reconsider When: Process accounting available.
        References: main. ADR: None.
        """
        memory=self.Memory();util=self.Utilization()
        self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle,C.byref(memory)))
        self.check(self.lib.nvmlDeviceGetUtilizationRates(self.handle,C.byref(util)))
        if memory.used>memory.total or util.gpu>100 or util.memory>100:raise RuntimeError('invalid NVML measurement')
        return {'whole_gpu_total_bytes':memory.total,'whole_gpu_used_bytes':memory.used,'whole_gpu_free_bytes':memory.free,
                'whole_gpu_utilization_percent':util.gpu,'whole_gpu_memory_utilization_percent':util.memory}

    def close(self):
        """Release the NVML client without changing any model or GPU process.

        Args: None. Returns: None. Raises: NVML shutdown error.
        Design: Context: Owned monitor lifecycle. Concern: Resource leaks.
            Decision: Explicit shutdown. Alternatives: Process exit only.
            Quality Attributes: Hygiene. Trade-offs: None.
        Evidence: Internal: Handle ownership. External: nvmlShutdown.
            Findings: No model termination. Metrics: None. Confidence: Contract.
        Constraints: Read-only. Reconsider When: Shared library ownership changes.
        References: main. ADR: None.
        """
        self.check(self.lib.nvmlShutdown())


def host_sample():
    """Read aggregate host pressure without recording users, command lines or identifiers.

    Args: None. Returns: Availability/swap/load mapping. Raises: IO/schema error.
    Design: Context: Large checkpoint restoration can compete for RAM/IO.
        Concern: GPU-only samples miss host pressure.
        Decision: Linux meminfo and load averages. Alternatives: Store unrelated process details.
        Quality Attributes: Privacy, observability. Trade-offs: No process attribution.
    Evidence: Internal: procfs counters. External: Linux meminfo semantics.
        Findings: Existing swap occupancy alone does not prove active swapping.
        Metrics: Available/swap bytes and load1. Confidence: Host sample.
    Constraints: No secret/user/process data. Reconsider When: NonLinux host.
    References: main. ADR: None.
    """
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1);values[key]=int(value.strip().split()[0])*1024
    return {'host_memory_available_bytes':values['MemAvailable'],'host_memory_total_bytes':values['MemTotal'],
            'host_swap_used_bytes':values['SwapTotal']-values['SwapFree'],'host_load1':os.getloadavg()[0]}


def main():
    """Sample until an owned stop file, signal or explicit count; keep every finite sample.

    Args: CLI output/phase/stop/period/samples. Returns: None. Raises: Probe/IO errors.
    Design: Context: One shared monitor spans serial clean method evaluations.
        Concern: Monitoring itself must remain small and failed reads visible.
        Decision: stdlib-only, 0.5s default, monotonic deadlines, append-only JSONL.
        Alternatives: Launch one monitor per worker. Quality Attributes: Low interference.
        Trade-offs: Very short bursts can fall between samples.
    Evidence: Internal: Measured probe_wall_s. External: NVML.
        Findings: Participant coordination plus traces supports, not certifies, exclusivity.
        Metrics: Sampling duration/memory/utilization. Confidence: Sampled.
    Constraints: Fresh output directory; never stop another process.
    Reconsider When: Hard realtime hardware isolation is required.
    References: gpu coordination sidecars. ADR: None.
    """
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase-file',type=Path);p.add_argument('--stop-file',type=Path);p.add_argument('--period',type=float,default=.5);p.add_argument('--samples',type=int)
    args=p.parse_args()
    if not math.isfinite(args.period) or args.period<.1 or (args.samples is not None and args.samples<1):p.error('invalid sample period/count')
    args.output.mkdir(parents=True,exist_ok=False);stop=threading.Event();probe=None;count=0;started=time.monotonic()
    source=Path(__file__).read_bytes();(args.output/'monitor-source.py').write_bytes(source)
    (args.output/'configuration.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),'period_s':args.period,
        'sample_limit':args.samples,'scope':'Whole-device NVML and host pressure; no process attribution'},indent=2)+'\n')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:
        probe=NvmlProbe()
        with (args.output/'samples.jsonl').open('x') as sink:
            deadline=time.monotonic()
            while not stop.is_set() and not (args.stop_file and args.stop_file.exists()):
                t=time.monotonic();row={'utc':datetime.now(timezone.utc).isoformat(),'monotonic':t,**probe.sample(),**host_sample()}
                if args.phase_file and args.phase_file.exists():
                    try:
                        phase=json.loads(args.phase_file.read_text())
                        if not isinstance(phase,dict):raise ValueError('phase must be an object')
                        row['phase']=phase.get('state');row['our_gpu_workers_running']=phase.get('gpu_workers_running')
                    except (OSError,ValueError):row['phase_read_failed']=True
                row['probe_wall_s']=time.monotonic()-t;sink.write(json.dumps(row,allow_nan=False)+'\n');sink.flush();count+=1
                if args.samples is not None and count>=args.samples:break
                deadline+=args.period;stop.wait(max(0,deadline-time.monotonic()))
        (args.output/'summary.json').write_text(json.dumps({'status':'complete','samples':count,'wall_seconds':time.monotonic()-started,'period_s':args.period,
            'scope':'Whole-device driver counters and host pressure; desktop included; no per-process GPU attribution; coordinated model exclusivity is separate evidence'},indent=2)+'\n')
    except Exception as exc:
        (args.output/'summary.json').write_text(json.dumps({'status':'error','samples':count,'wall_seconds':time.monotonic()-started,
            'period_s':args.period,'error':f'{type(exc).__name__}: {exc}','measurement_scope':'Failed probe is not idle hardware evidence'},indent=2)+'\n')
        raise
    finally:
        if probe is not None:probe.close()


if __name__=='__main__':main()
