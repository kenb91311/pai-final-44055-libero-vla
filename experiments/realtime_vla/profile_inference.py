"""Profile frozen native VLA requests against identical captured raw observations.

Args: CLI model, immutable observation directory, fresh output directory and repeat controls.
Returns: report.json, actions.npz and measured baseline/instrumented/baseline evidence.
Raises: Invalid input/output, native inference, frozen audit or instrumentation errors.
Design:
    Context: Parameter counts alone do not explain native SmolVLA, pi05 and FAST latency.
    Concern: Different horizons, image layouts, precision and samplers confound inference comparisons.
    Decision: Preserve native prediction, bracket instrumentation with paired uninstrumented conditions,
        and report nested stage intervals with exact source and input identity.
    Alternatives: Rescale horizons, retrain, compile, quantize, or edit native model implementation.
    Quality Attributes: Reproducibility, isolation. Trade-offs: Sequential runs cannot remove thermal drift.
Evidence:
    Internal: Existing frozen runtimes and capture_profile_inputs.py raw sensor schema.
    External: CUDA events measure stream elapsed time, not kernel busy time.
    Findings: All stage attribution is measured; architectural explanations remain separate from timings.
    Metrics: Full-call latency, stage host/stream elapsed, paired equality and instrumentation overhead.
    Confidence: CPU mechanics tested; benchmark conclusions require actual GPU artifacts.
Constraints: No optimizer, quantization, torch.compile, library source edits or simulator truth inputs.
Reconsider When: Native checkpoint/config/source or CUDA stream behavior changes.
References: profiling_hooks.py. ADR: None.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import numpy as np

try:
    from .profiling_hooks import IntervalRecorder, NativeInstrumentation, describe_tensors
    from .capture_profile_inputs import MODEL_INPUT_KEYS, load_profile_input, sensor_hashes
except ImportError:
    from profiling_hooks import IntervalRecorder, NativeInstrumentation, describe_tensors
    from capture_profile_inputs import MODEL_INPUT_KEYS, load_profile_input, sensor_hashes

MODELS = ('smolvla', 'pi05', 'pi0fast')
RAW_KEYS = MODEL_INPUT_KEYS


def file_sha256(path):
    """Hash an existing artifact without loading all its bytes into memory.

    Args: path: readable file. Returns: Hex SHA256. Raises: File I/O errors.
    Design: Context: Native dependency edits affect interpretation. Concern: Version strings omit local changes.
        Decision: Hash exact files in chunks. Alternatives: Git status or package version alone.
        Quality Attributes: Identity. Trade-offs: Untimed disk I/O.
    Evidence: Internal: Runtime source snapshots. External: SHA256.
        Findings: Content identity is independent of install path. Metrics: SHA256. Confidence: High.
    Constraints: Read only. Reconsider When: Artifacts become remote-only.
    References: collect_metadata. ADR: None.
    """
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_observations(directory):
    """Validate captured NPZ allowlists, file/sensor hashes and prompt before model construction.

    Args: directory: directory containing manifest.json and sample NPZs.
    Returns: Manifest and ordered sample records containing owned arrays.
    Raises: ValueError for schema/hash mismatch; filesystem/NumPy errors for unreadable inputs.
    Design: Context: All models must see byte-identical raw observations.
        Concern: An object-state field or silently cast capture changes the experiment.
        Decision: Exact sensor allowlist, native floating proprio dtype and declared SHA verification.
        Alternatives: Resimulate separately for each model. Quality Attributes: Comparability.
        Trade-offs: Captured states measure latency, not rollout success.
    Evidence: Internal: capture_profile_inputs manifest version1. External: NumPy allow_pickle=False.
        Findings: RGB256 uint8 and finite proprio3/4/2 are sufficient for all runtimes.
        Metrics: Sensor/file hashes. Confidence: Exact identity.
    Constraints: No path traversal or pickle payloads. Reconsider When: Raw sensor schema changes.
    References: RAW_KEYS. ADR: None.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not isinstance(manifest.get('prompt'), str) or not manifest['prompt'].strip():
        raise ValueError('manifest.prompt must be a nonempty string')
    declared = manifest.get('samples')
    if not isinstance(declared, list) or not declared:
        raise ValueError('manifest.samples must be a nonempty ordered list')
    samples = []
    identifiers = set()
    for record in declared:
        identifier = str(record['id'])
        if identifier in identifiers:
            raise ValueError(f'duplicate sample id: {identifier}')
        identifiers.add(identifier)
        path = (directory / record['file']).resolve()
        if not path.is_relative_to(directory) or path.suffix != '.npz':
            raise ValueError('sample must be an NPZ inside observations directory')
        digest = file_sha256(path)
        if record.get('sha256') is not None and record['sha256'] != digest:
            raise ValueError(f'NPZ hash mismatch: {identifier}')
        arrays = load_profile_input(path, resolution=256)
        hashes = sensor_hashes(arrays, resolution=256)
        if record.get('sensor_hashes') is not None and hashes != record['sensor_hashes']:
            raise ValueError(f'sensor hash mismatch: {identifier}')
        samples.append({'id': identifier, 'file': record['file'], 'sha256': digest,
                        'sensor_hashes': hashes, 'observation': arrays})
    return manifest, samples


def make_schedule(samples, repeats, seed):
    """Create identical observation ordering and RNG seeds for all three conditions.

    Args: samples: ordered captures; repeats: positive count; seed: nonnegative base seed.
    Returns: List of sample/repeat/seed entries. Raises: ValueError for invalid counts.
    Design: Context: Flow policies sample noise. Concern: Different RNG histories break output pairing.
        Decision: Reset the runtime outside each timed call using a stable per-pair seed.
        Alternatives: Reset once per condition. Quality Attributes: Reproducibility.
        Trade-offs: Samples are independent requests rather than an online sequence.
    Evidence: Internal: Native reset(seed) methods. External: Torch manual_seed.
        Findings: Both repeated conditions receive the same schedule object.
        Metrics: Exact sample/seed identities. Confidence: CPU fake-runtime test.
    Constraints: Preserve input order. Reconsider When: Sequence context is the target measurement.
    References: run_condition. ADR: None.
    """
    if repeats < 1 or seed < 0 or not samples:
        raise ValueError('positive repeats, samples and nonnegative seed required')
    return [{'sample': sample, 'repeat': repeat, 'seed': seed + repeat * len(samples) + index}
            for repeat in range(repeats) for index, sample in enumerate(samples)]


def synchronize(runtime):
    """Wait only at an outer request boundary when the runtime uses CUDA.

    Args: runtime: native runtime with device and torch module or imported torch.
    Returns: None. Raises: CUDA backend errors.
    Design: Context: Native prediction already contains its original synchronizations.
        Concern: Async launches underreport full-call latency. Decision: Boundary-only additional waits.
        Alternatives: Per-stage waits. Quality Attributes: Timing fidelity. Trade-offs: No cross-request overlap.
    Evidence: Internal: Native predict methods. External: CUDA synchronization semantics.
        Findings: Instrumented stages never synchronize. Metrics: Full-call wall time. Confidence: High.
    Constraints: CPU checks do not import/initialize CUDA. Reconsider When: Multi-request throughput is measured.
    References: timed_predict. ADR: None.
    """
    if getattr(getattr(runtime, 'device', None), 'type', None) == 'cuda':
        torch_module = getattr(runtime, 'torch', None)
        if torch_module is None:
            import torch as torch_module
        torch_module.cuda.synchronize(runtime.device)


def timed_predict(runtime, entry, prompt, instrumentation=None, torch_module=None):
    """Measure one full native request with reset and sensor copies excluded.

    Args: runtime/entry/prompt: native request; instrumentation: optional active wrapper context;
        torch_module: torch backend for CUDA event allocation.
    Returns: Call record and owned action array. Raises: Native errors or invalid returned actions.
    Design: Context: Baseline and instrumented conditions must execute identical arithmetic.
        Concern: Reset, action hashing and device inspection contaminate inference timing.
        Decision: Run setup before timing and all event resolution/value inspection after final boundary.
        Alternatives: Time only policy.forward. Quality Attributes: End-to-end comparability.
        Trade-offs: Native inner synchronizations remain part of the existing runtime contract.
    Evidence: Internal: Native predict output NumPy contract and paired tests. External: perf_counter.
        Findings: FAST generated values are transferred only after the timed call.
        Metrics: Wall, native metadata, SHA and actual token/forward counts. Confidence: Measured.
    Constraints: No new stage waits, no native parameter/config mutation.
    Reconsider When: Native predict stops returning CPU actions.
    References: run_condition. ADR: None.
    """
    runtime.reset(seed=entry['seed'])
    observation = {key: value.copy() for key, value in entry['sample']['observation'].items()}
    synchronize(runtime)
    recorder = None
    if instrumentation is not None:
        is_cuda = getattr(runtime.device, 'type', None) == 'cuda'
        recorder = IntervalRecorder(torch_module.cuda if is_cuda else None, runtime.device)
        instrumentation.recorder = recorder
        instrumentation.sampled_tokens = None
    started = time.perf_counter()
    try:
        with recorder.measure('request_inclusive') if recorder is not None else nullcontext():
            actions, native_metadata = runtime.predict(observation, prompt)
        synchronize(runtime)
        wall_ms = (time.perf_counter() - started) * 1000
    finally:
        if instrumentation is not None:
            instrumentation.recorder = None
    actions = np.asarray(actions).copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f'invalid native action shape/values: {actions.shape}')
    record = {'sample_id': entry['sample']['id'], 'repeat': entry['repeat'], 'seed': entry['seed'],
              'wall_ms': wall_ms, 'action_sha256': hashlib.sha256(actions.tobytes()).hexdigest(),
              'action_dtype': str(actions.dtype), 'action_shape': list(actions.shape),
              'native_metadata': native_metadata, 'sensor_hashes': entry['sample']['sensor_hashes']}
    if recorder is not None:
        record['profile'] = recorder.finish()
        record['outer_boundary_and_recorder_host_overhead_ms'] = wall_ms - record['profile']['accounting']['host_root_ms']
        stages = record['profile']['stages']
        if instrumentation.model == 'pi0fast':
            prefill = stages.get('vlm_prefix_prefill', {}).get('calls', 0)
            cached = stages.get('fast_cached_decode_forward', {}).get('calls', 0)
            emitted = stages.get('fast_logits_projection', {}).get('calls', 0)
            tokens = instrumentation.sampled_tokens
            emitted_ids = tokens.detach().cpu().numpy()[:, :emitted].tolist() if tokens is not None else None
            tokenizer = runtime.policy.model._require_paligemma_tokenizer()
            marker_id = int(tokenizer.convert_tokens_to_ids('|'))
            marker_present = [marker_id in row for row in emitted_ids] if emitted_ids is not None else []
            record['fast_generation'] = {'prefill_forward_count': prefill, 'cached_decode_forward_count': cached,
                                         'lm_head_forward_count': emitted, 'actual_generated_token_count': emitted,
                                         'end_marker_token_id': marker_id,
                                         'end_marker_present_by_batch_row': marker_present,
                                         'actual_count_includes_end_marker': bool(marker_present) and all(marker_present),
                                         'termination_reason': 'end_marker' if marker_present and all(marker_present) else 'decoding_cap_or_native_stop',
                                         'actual_count_method': 'one native LM head invocation for each emitted token',
                                         'emitted_token_ids': emitted_ids,
                                         'returned_token_buffer_shape': list(tokens.shape) if tokens is not None else None,
                                         'kv_cache_forward_count_consistent': emitted == prefill + cached}
    return record, actions


def run_condition(runtime, name, samples, schedule, prompt, warmup, seed, instrumentation=None, torch_module=None):
    """Warm one condition separately, then execute the common paired schedule.

    Args: runtime/name/samples/schedule/prompt: experiment condition; warmup/seed: warmup controls;
        instrumentation/torch_module: optional event recording backend.
    Returns: Condition report and measured action arrays. Raises: Native measurement errors.
    Design: Context: Event creation and kernels have cold-start costs.
        Concern: First instrumented request would include lazy initialization absent from baseline.
        Decision: Warm the actual condition, including recorder hooks/events, and retain warmup separately.
        Alternatives: Warm only baseline. Quality Attributes: Comparability. Trade-offs: Extra inference calls.
    Evidence: Internal: Warmup and measured count tests. External: CUDA lazy initialization behavior.
        Findings: Warmup seeds/order are identical across conditions. Metrics: Separate warmup latency.
        Confidence: CPU protocol test; GPU variation remains measured.
    Constraints: Warmup never enters latency summaries or paired comparisons.
    Reconsider When: Persistent online state is benchmarked.
    References: make_schedule. ADR: None.
    """
    warm_records = []
    for index in range(warmup):
        entry = {'sample': samples[index % len(samples)], 'repeat': -1, 'seed': seed + 100000 + index}
        record, _ = timed_predict(runtime, entry, prompt, instrumentation, torch_module)
        warm_records.append({'sample_id': record['sample_id'], 'seed': record['seed'], 'wall_ms': record['wall_ms']})
    records, outputs = [], []
    for entry in schedule:
        record, output = timed_predict(runtime, entry, prompt, instrumentation, torch_module)
        records.append(record)
        outputs.append(output)
    report = {'name': name, 'warmup': {'excluded_from_summary': True, 'calls': warm_records},
              'calls': records, 'latency_ms': summarize([record['wall_ms'] for record in records])}
    if instrumentation is not None:
        report['stage_summary'] = summarize_stages(records)
    return report, outputs


def summarize_stages(records):
    """Aggregate each named stage across requests while keeping nesting semantics explicit.

    Args: records: instrumented measured call records, excluding warmup.
    Returns: Per-stage call counts and per-request inclusive/exclusive host/stream summaries.
    Raises: Missing profile schema errors.
    Design: Context: Repeated denoise/token calls need readable summaries.
        Concern: Inclusive prefix and vision totals must not be combined.
        Decision: Aggregate only identical labels and preserve explicit inclusive/exclusive metric names.
        Alternatives: One flat summed runtime table. Quality Attributes: Interpretability.
        Trade-offs: Per-request stage totals differ from average individual denoise/token duration.
    Evidence: Internal: Native interval tree. External: None. Findings: Raw instances remain in each call.
        Metrics: Count and descriptive latency. Confidence: Sample descriptions only.
    Constraints: Never sum distinct inclusive stage labels. Reconsider When: Stage hierarchy changes.
    References: IntervalRecorder.finish. ADR: None.
    """
    labels = sorted({label for record in records for label in record['profile']['stages']})
    summary = {}
    for label in labels:
        rows = [record['profile']['stages'].get(label, {}) for record in records]
        metrics = {key for row in rows for key in row if key != 'calls'}
        summary[label] = {'native_invocations_per_request': summarize([row.get('calls', 0) for row in rows]),
                          'per_request_stage_total_ms': {metric: summarize([row.get(metric, 0.) for row in rows])
                                                         for metric in sorted(metrics)}}
    return {'semantics': 'per-request totals of identical stage labels only; inclusive labels nest and must not be summed',
            'stages': summary}


def summarize(values):
    """Describe measured latency without inferring confidence from a tiny sample.

    Args: values: finite numeric observations. Returns: Count/mean/median/min/max/p95; empty count if absent.
    Raises: Numeric errors for invalid inputs.
    Design: Context: Small repeated state set. Concern: Percentiles are descriptive rather than inferential.
        Decision: Preserve every call and give simple sample summaries. Alternatives: Claim population significance.
        Quality Attributes: Transparency. Trade-offs: No confidence interval is inferred.
    Evidence: Internal: Raw calls in report.json. External: NumPy percentile.
        Findings: Raw timings remain available. Metrics: Descriptive milliseconds. Confidence: Sample only.
    Constraints: No warmup samples. Reconsider When: A statistical study is requested.
    References: run_condition. ADR: None.
    """
    if not values:
        return {'count': 0}
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'p95': float(np.percentile(values, 95))}


def compare_conditions(before, instrumented, after, before_actions, instrumented_actions, after_actions):
    """Pair all three conditions by exact sample/repeat/seed and compare returned actions.

    Args: before/instrumented/after: condition reports; corresponding action lists.
    Returns: Per-pair equality, overhead ratios, bracket drift and aggregate diagnostics.
    Raises: ValueError for mismatched schedule/counts.
    Design: Context: Instrumentation overhead and warm drift are distinct.
        Concern: A single baseline can mistake heating or clock drift for hook overhead.
        Decision: Bracket each profiled pair using both baseline conditions and retain exact/allclose checks.
        Alternatives: Compare only unpaired means. Quality Attributes: Auditability.
        Trade-offs: Bracketing cannot reconstruct instantaneous hardware state.
    Evidence: Internal: Byte-identical input hashes and common seeds. External: NumPy comparisons.
        Findings: Action differences invalidate arithmetic-preserving interpretation.
        Metrics: Exact equality, maximum absolute error, ratio and baseline drift. Confidence: Measured.
    Constraints: Identical schedules/counts required. Reconsider When: Stochastic hardware nondeterminism is material.
    References: make_schedule. ADR: None.
    """
    counts = [len(item) for item in (before['calls'], instrumented['calls'], after['calls'],
                                    before_actions, instrumented_actions, after_actions)]
    if len(set(counts)) != 1:
        raise ValueError('paired conditions have different call counts')
    pairs = []
    for baseline, measured, trailing, a, b, c in zip(before['calls'], instrumented['calls'], after['calls'],
                                                   before_actions, instrumented_actions, after_actions):
        keys = [(row['sample_id'], row['repeat'], row['seed'], row['sensor_hashes'])
                for row in (baseline, measured, trailing)]
        if keys[0] != keys[1] or keys[0] != keys[2] or a.shape != b.shape or a.shape != c.shape:
            raise ValueError('paired observations/seeds/action shapes do not match')
        bracket_ms = (baseline['wall_ms'] + trailing['wall_ms']) / 2
        pairs.append({'sample_id': baseline['sample_id'], 'repeat': baseline['repeat'], 'seed': baseline['seed'],
                      'instrumented_exact_equal_before': bool(np.array_equal(a, b)),
                      'instrumented_exact_equal_after': bool(np.array_equal(c, b)),
                      'instrumented_allclose_before': bool(np.allclose(a, b, rtol=1e-5, atol=1e-6)),
                      'instrumented_allclose_after': bool(np.allclose(c, b, rtol=1e-5, atol=1e-6)),
                      'max_abs_error_before': float(np.max(np.abs(a.astype(np.float64) - b))),
                      'max_abs_error_after': float(np.max(np.abs(c.astype(np.float64) - b))),
                      'bracket_baseline_mean_ms': bracket_ms,
                      'instrumentation_overhead_ratio': measured['wall_ms'] / bracket_ms,
                      'instrumentation_overhead_ms': measured['wall_ms'] - bracket_ms,
                      'baseline_after_over_before_ratio': trailing['wall_ms'] / baseline['wall_ms']})
    return {'pairs': pairs, 'all_exact_equal': all(row['instrumented_exact_equal_before'] and
                                                   row['instrumented_exact_equal_after'] for row in pairs),
            'all_allclose': all(row['instrumented_allclose_before'] and row['instrumented_allclose_after'] for row in pairs),
            'allclose_tolerance': {'rtol': 1e-5, 'atol': 1e-6},
            'overhead_ratio': summarize([row['instrumentation_overhead_ratio'] for row in pairs]),
            'baseline_drift_ratio': summarize([row['baseline_after_over_before_ratio'] for row in pairs])}


def construct_runtime(model, modeldir, device):
    """Construct exactly one existing frozen native runtime without optimization overrides.

    Args: model: supported model name; modeldir: local checkpoint; device: CUDA/CPU string.
    Returns: Existing Frozen runtime. Raises: Native restoration/dependency errors.
    Design: Context: Existing runtime behavior is the comparison subject.
        Concern: Reimplementation can alter preprocessing or precision. Decision: Import and call existing classes.
        Alternatives: New model code. Quality Attributes: Native fidelity. Trade-offs: Existing startup audits remain.
    Evidence: Internal: FrozenSmolVLA, FrozenPI05, FrozenPI0Fast. External: None.
        Findings: No compile flag is enabled. Metrics: Restoration proof. Confidence: Native audit.
    Constraints: No training, compile, quantization or config horizon changes.
    Reconsider When: A new runtime is explicitly authorized. References: Existing runtime files. ADR: None.
    """
    prefix = 'experiments.realtime_vla.' if __package__ else ''
    modules = {'smolvla': ('smolvla_runtime', 'FrozenSmolVLA'), 'pi05': ('pi05_runtime', 'FrozenPI05'),
               'pi0fast': ('pi0fast_runtime', 'FrozenPI0Fast')}
    module_name, class_name = modules[model]
    module = __import__(prefix + module_name, fromlist=[class_name])
    return getattr(module, class_name)(modeldir=modeldir, device=device)


def collect_metadata(runtime, model, torch_module):
    """Fingerprint exact native source, saved processors, parameters and execution environment.

    Args: runtime/model/torch_module: loaded native policy and backend.
    Returns: JSON-compatible metadata. Raises: Source/config read errors.
    Design: Context: Local dependency optimizations can explain timing differences.
        Concern: Config dtype and total parameters do not identify active precision or operations.
        Decision: Record stored dtype distribution, architecture/config and exact source hashes separately.
        Alternatives: Report parameter count as a speed predictor. Quality Attributes: Traceability.
        Trade-offs: Distribution includes parameters unused during inference; this is stated explicitly.
    Evidence: Internal: Native parameters/config/snapshot files. External: Distribution RECORD files.
        Findings: Actual activation precision is collected in an independent probe.
        Metrics: Counts, bytes, hashes and threads. Confidence: Runtime metadata.
    Constraints: Hash/audit outside all measured loops. Reconsider When: Dependency layout changes.
    References: capture_precision_probe. ADR: None.
    """
    policy = runtime.policy
    distributions = Counter()
    parameter_tensors = Counter()
    parameter_bytes = Counter()
    for parameter in policy.parameters():
        distributions[str(parameter.dtype)] += parameter.numel()
        parameter_tensors[str(parameter.dtype)] += 1
        parameter_bytes[str(parameter.dtype)] += parameter.numel() * parameter.element_size()
    source_paths = {Path(__file__).resolve(), Path(inspect.getfile(IntervalRecorder)).resolve(),
                    Path(inspect.getfile(type(runtime))).resolve(), Path(inspect.getfile(load_profile_input)).resolve()}
    for module in policy.modules():
        try:
            path = Path(inspect.getfile(type(module))).resolve()
            if any(token in str(path) for token in ('lerobot', 'transformers')):
                source_paths.add(path)
        except (TypeError, OSError):
            pass
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('lerobot.') and path and str(path).endswith('.py'):
            source_paths.add(Path(path).resolve())
    dependencies = {}
    for package in ('torch', 'lerobot', 'transformers', 'numpy', 'safetensors', 'tokenizers'):
        try:
            distribution = importlib.metadata.distribution(package)
            record = next((distribution.locate_file(item) for item in distribution.files or []
                           if str(item).endswith('.dist-info/RECORD')), None)
            dependencies[package] = {'version': distribution.version,
                                     'installed_record_sha256': file_sha256(record) if record else None}
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = {'version': None}
    snapshots = {}
    for path in Path(runtime.modeldir).glob('*'):
        if path.is_file() and (path.suffix == '.json' or ('processor' in path.name and path.suffix == '.safetensors')):
            snapshots[path.name] = file_sha256(path)
    configuration = json.loads((Path(runtime.modeldir) / 'config.json').read_text())
    provenance_path = Path(runtime.modeldir) / 'provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    live_configuration = vars(policy.config)
    module_lists = {name: len(module) for name, module in policy.named_modules()
                    if isinstance(module, torch_module.nn.ModuleList) and any(part in name for part in ('layer', 'block'))}
    return {'model': model, 'modeldir': str(runtime.modeldir), 'model_revision_metadata': getattr(runtime, 'metadata', None),
            'checkpoint_provenance': provenance,
            'restoration': getattr(runtime, 'restoration', None), 'saved_config': configuration,
            'live_config': live_configuration, 'snapshot_sha256': snapshots,
            'runtime_parameter_count': sum(distributions.values()),
            'parameter_elements_by_dtype': dict(distributions), 'parameter_tensors_by_dtype': dict(parameter_tensors),
            'parameter_bytes_by_dtype': dict(parameter_bytes),
            'parameter_count_semantics': 'all registered runtime parameters, including inference-unused heads and tied/derived representation effects',
            'requires_grad_parameter_count': sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad),
            'module_layer_counts': module_lists,
            'source_sha256': {str(path): file_sha256(path) for path in sorted(source_paths)},
            'dependencies': dependencies, 'python': sys.version, 'platform': platform.platform(),
            'torch_cuda_version': torch_module.version.cuda,
            'device': str(runtime.device),
            'gpu_name': torch_module.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else None,
            'threads': {'torch_intraop': torch_module.get_num_threads(), 'torch_interop': torch_module.get_num_interop_threads(),
                        'environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}},
            'precision_backend_flags': {'float32_matmul_precision': torch_module.get_float32_matmul_precision(),
                                        'cuda_matmul_allow_tf32': torch_module.backends.cuda.matmul.allow_tf32,
                                        'cudnn_allow_tf32': torch_module.backends.cudnn.allow_tf32,
                                        'cuda_matmul_allow_bf16_reduced_precision_reduction':
                                            torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                        'cuda_matmul_allow_fp16_reduced_precision_reduction':
                                            torch_module.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                                        'deterministic_algorithms_enabled': torch_module.are_deterministic_algorithms_enabled(),
                                        'cudnn_benchmark': torch_module.backends.cudnn.benchmark,
                                        'flags_changed_by_profiler': False},
            'native_inner_synchronizations_retained': True,
            'host_stage_semantics': 'host elapsed includes launches and implicit waits; exclusive host is an accounting partition, not GPU compute attribution',
            'saved_postprocessing_semantics': 'native saved processor may include CPU device transfer and wait for prior GPU work; inspect processor snapshots',
            'runtime_residual_semantics': 'unwrapped raw conversion, synchronization, final casts/NumPy views/validation, and wrapper overhead; not pure D2H time',
            'optimization_changes': {'training': False, 'quantization': False, 'torch_compile': False,
                                     'horizon_change': False, 'model_source_edit': False}}


def capture_precision_probe(runtime, sample, prompt, seed, torch_module):
    """Observe representative native attention projection activation precision outside timing.

    Args: runtime/sample/prompt/seed/torch_module: one ordinary native request.
    Returns: First vision/language/expert q_proj input/output dtype and ambient autocast records.
    Raises: Native inference errors; hooks are removed in all cases.
    Design: Context: Stored FP32 can compute in BF16 autocast. Concern: Config dtype mislabels vision cost.
        Decision: One independent request with metadata-only forward hooks on representative q_proj modules.
        Alternatives: Infer compute from stored tensors. Quality Attributes: Evidence quality.
        Trade-offs: Representative operations do not prove every kernel's accumulation dtype.
    Evidence: Internal: Native vision precision branches. External: PyTorch forward hooks.
        Findings: Returned activations and ambient autocast directly expose operation output precision.
        Metrics: Input/output dtype. Confidence: Representative activation evidence only.
    Constraints: Probe excluded from timed/warmup sample counts and all hooks removed.
    Reconsider When: Models use fused projections without q_proj. References: collect_metadata. ADR: None.
    """
    selected = {}
    for name, module in runtime.policy.named_modules():
        if not name.endswith('q_proj'):
            continue
        category = ('vision' if 'vision' in name else 'action_expert'
                    if any(part in name for part in ('.gemma_expert.', '.lm_expert.')) else 'language')
        selected.setdefault(category, (name, module))
    observed = {}
    handles = []
    for category, (name, module) in selected.items():
        def hook(owner, inputs, output, category=category, name=name):
            """Capture the first representative output layout without reading tensor values.

            Args: owner/inputs/output: Torch forward hook; category/name: bound module identity.
            Returns: None, preserving output. Raises: Metadata access errors.
            Design: Context: Untimed precision probe. Concern: Hook altering inference output.
                Decision: Return None and capture once. Alternatives: Tensor conversion.
                Quality Attributes: Fidelity. Trade-offs: Representative coverage.
            Evidence: Internal: Native q_proj modules. External: Torch hook contract.
                Findings: Autocast can differ by component. Metrics: Activation dtype. Confidence: Observed.
            Constraints: No tensor value access. Reconsider When: Projection API changes.
            References: capture_precision_probe. ADR: None.
            """
            if category not in observed:
                enabled = bool(torch_module.is_autocast_enabled('cuda'))
                observed[category] = {'module': name, 'inputs': describe_tensors(inputs), 'output': describe_tensors(output),
                                      'cuda_autocast_enabled': enabled,
                                      'cuda_autocast_dtype': str(torch_module.get_autocast_dtype('cuda')) if enabled else None}
        handles.append(module.register_forward_hook(hook))
    try:
        runtime.reset(seed=seed)
        observation = {key: value.copy() for key, value in sample['observation'].items()}
        runtime.predict(observation, prompt)
        synchronize(runtime)
    finally:
        for handle in handles:
            handle.remove()
    return {'excluded_from_latency_and_warmup_counts': True, 'representative_q_projection': observed,
            'coverage_limit': 'output dtype and ambient autocast; not a claim about all kernel accumulation precision'}


def build_parser():
    """Expose the native profiling controls without importing model/CUDA dependencies.

    Args: None. Returns: ArgumentParser. Raises: None.
    Design: Context: Users need a CPU-safe help command. Concern: CLI help initializing a GPU.
        Decision: Lazy runtime/torch imports after parsing and input validation. Alternatives: Top-level model imports.
        Quality Attributes: Usability. Trade-offs: None.
    Evidence: Internal: --help CPU smoke. External: argparse.
        Findings: Defaults preserve native horizons and existing four-thread worker condition.
        Metrics: No GPU calls. Confidence: CPU checked.
    Constraints: Fresh output directory required. Reconsider When: Resume semantics are designed.
    References: main. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modeldir', type=Path, help='Default: artifacts/realtime-vla/models/MODEL')
    parser.add_argument('--warmup', type=int, default=10, help='Excluded calls before EACH of B-I-B conditions')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--interop-threads', type=int, default=1)
    return parser


def main(argv=None):
    """Run baseline/instrumented/baseline with full restoration and frozen-state evidence.

    Args: argv: optional CLI argument list. Returns: Zero on completed report.
    Raises: Invalid arguments, native errors or failed frozen/interval audits.
    Design: Context: A single leased GPU hosts one frozen runtime at a time.
        Concern: Partial output or silent audit failures masquerade as valid measurements.
        Decision: Require a fresh directory, save reports/actions atomically after calls, and fail invalid audits.
        Alternatives: Append into old artifacts. Quality Attributes: Integrity.
        Trade-offs: Runtime construction retains its original tokenizer/restoration side effects.
    Evidence: Internal: Native audit methods and scoped wrapper tests. External: None.
        Findings: A completed artifact contains all raw timings and exact pairing evidence.
        Metrics: Frozen before/after, equality, accounting and condition counts. Confidence: Measured.
    Constraints: No GPU launch by tests/help; caller owns shared GPU lease.
    Reconsider When: Parallel model serving or resumable runs are required.
    References: report.json schema_version1. ADR: None.
    """
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.repeats < 1 or args.seed < 0 or args.threads < 1 or args.interop_threads < 1:
        raise ValueError('invalid warmup/repeats/seed/thread controls')
    manifest, samples = load_observations(args.observations)
    schedule = make_schedule(samples, args.repeats, args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    modeldir = args.modeldir or Path('artifacts/realtime-vla/models') / args.model
    runtime = construct_runtime(args.model, modeldir, args.device)
    try:
        metadata = collect_metadata(runtime, args.model, torch)
        if metadata['requires_grad_parameter_count']:
            raise RuntimeError('Expected a fully frozen runtime')
        before_digest = getattr(runtime, 'weights_before', getattr(runtime, 'weights_before_sha256', None))
        metadata['precision_probe'] = capture_precision_probe(runtime, samples[0], manifest['prompt'], args.seed, torch)
        before, before_actions = run_condition(runtime, 'baseline_before', samples, schedule, manifest['prompt'],
                                               args.warmup, args.seed)
        with NativeInstrumentation(runtime, args.model, torch) as instrumentation:
            instrumented, instrumented_actions = run_condition(runtime, 'instrumented', samples, schedule,
                                                               manifest['prompt'], args.warmup, args.seed,
                                                               instrumentation, torch)
            metadata['installed_hooks'] = instrumentation.hooks
        after, after_actions = run_condition(runtime, 'baseline_after', samples, schedule, manifest['prompt'],
                                             args.warmup, args.seed)
        paired = compare_conditions(before, instrumented, after, before_actions, instrumented_actions, after_actions)
        audit = runtime.audit()
        audit['requires_grad_parameter_count_after'] = sum(parameter.numel() for parameter in runtime.policy.parameters()
                                                          if parameter.requires_grad)
        audit['all_registered_modules_eval_after'] = not any(module.training for module in runtime.policy.modules())
        interval_valid = all(call['profile']['accounting']['valid'] for call in instrumented['calls'])
        peak_cuda = {'allocated_bytes': torch.cuda.max_memory_allocated(runtime.device),
                     'reserved_bytes': torch.cuda.max_memory_reserved(runtime.device)} if runtime.device.type == 'cuda' else None
        report = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'protocol': {'condition_order': ['baseline_before', 'instrumented', 'baseline_after'],
                               'warmup_calls_per_condition': args.warmup, 'samples': len(samples),
                               'repeats': args.repeats, 'measured_calls_per_condition': len(schedule),
                               'seed': args.seed, 'native_prediction_calls_total': 1 + 3 * (args.warmup + len(schedule)),
                               'reset_and_rng_outside_timed_region': True,
                               'observation_copies_and_action_hashing_outside_timed_region': True,
                               'added_cuda_synchronizations': 'outer request boundaries only; original runtime waits retained',
                               'stage_times': 'inclusive and direct-child-exclusive host/stream elapsed; never sum inclusive nested stages',
                               'observation_directory': str(args.observations.resolve()),
                               'observation_manifest_sha256': file_sha256(args.observations / 'manifest.json'),
                               'prompt': manifest['prompt'],
                               'sample_provenance': [{key: value for key, value in sample.items() if key != 'observation'}
                                                     for sample in samples]},
                  'metadata': metadata, 'conditions': {'baseline_before': before, 'instrumented': instrumented, 'baseline_after': after},
                  'paired': paired, 'frozen_audit': {'before_digest': before_digest, **audit},
                  'validation': {'all_nested_interval_accounting_valid': interval_valid,
                                 'all_paired_outputs_exact': paired['all_exact_equal'],
                                 'all_paired_outputs_allclose': paired['all_allclose'],
                                 'weights_unchanged': audit.get('weights_unchanged')},
                  'memory': {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             'process_peak_rss_scope': 'entire process including model construction/restoration/audit',
                             'cuda_peak': peak_cuda}}
        action_arrays = {f'{condition}_{index:04d}': array for condition, arrays in
                         [('baseline_before', before_actions), ('instrumented', instrumented_actions), ('baseline_after', after_actions)]
                         for index, array in enumerate(arrays)}
        np.savez_compressed(output / 'actions.npz', **action_arrays)
        temporary = output / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + '\n')
        temporary.replace(output / 'report.json')
        print(json.dumps({'report': str(output / 'report.json'), 'validation': report['validation'],
                          'baseline_before_ms': before['latency_ms'], 'instrumented_ms': instrumented['latency_ms'],
                          'baseline_after_ms': after['latency_ms'], 'overhead_ratio': paired['overhead_ratio']}), flush=True)
        if (not audit.get('weights_unchanged') or audit['requires_grad_parameter_count_after'] or
                not audit['all_registered_modules_eval_after'] or not interval_valid or not paired['all_allclose']):
            raise RuntimeError('Profiling validity check failed; inspect saved report.json')
    finally:
        if hasattr(runtime, 'close'):
            runtime.close()
        del runtime
        gc.collect()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
