"""Measure native inference entrypoints with temporary, reversible instance wrappers.

Args: Loaded frozen runtime and an optional CUDA timing backend.
Returns: Nested host and CUDA stream elapsed intervals.
Raises: Invalid nesting, unsupported runtime entrypoints, or wrapped native exceptions.
Design:
    Context: Compare frozen policies without changing their source or inference arithmetic.
    Concern: Per-stage synchronization and summing nested times distort measured latency.
    Decision: Record event pairs asynchronously and subtract only direct, serial children.
    Alternatives: Kernel tracing requires CUPTI; CPU launch timing misses queued GPU work.
    Quality Attributes: Fidelity, reversibility. Trade-offs: Event/wrapper overhead is measured separately.
Evidence:
    Internal: smolvla_runtime.py, pi05_runtime.py, pi0fast_runtime.py and installed native methods.
    External: CUDA events measure elapsed stream intervals, including idle gaps.
    Findings: Prefix image embeddings, cached prefill and denoising are nested native calls.
    Metrics: Host elapsed, stream elapsed and accounting residual. Confidence: CPU mechanics tested.
Constraints: No stage synchronization, tensor value reads, training, or dependency source edits.
Reconsider When: Policies use concurrent streams or change their entrypoint contracts.
References: profile_inference.py. ADR: None.
"""
from contextlib import contextmanager
from functools import wraps
import time


def describe_tensors(value, depth=0):
    """Describe activation layout without reading device tensor values.

    Args: value: Native argument/result; depth: recursion bound.
    Returns: JSON-compatible shape/dtype/device descriptors. Raises: None for ordinary native objects.
    Design: Context: Actual precision can differ from config. Concern: GPU value reads synchronize.
        Decision: Read tensor metadata only. Alternatives: Copy activations to CPU.
        Quality Attributes: Low overhead. Trade-offs: Only the first four sequence members are described.
    Evidence: Internal: Native vision entrypoints. External: PyTorch metadata semantics.
        Findings: Shape and dtype require no device transfer. Metrics: None. Confidence: High.
    Constraints: Never inspect tensor values. Reconsider When: Non-tensor device wrappers are used.
    References: NativeInstrumentation. ADR: None.
    """
    if depth > 3:
        return None
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(getattr(value, 'device', 'cpu'))}
    if isinstance(value, dict):
        return {str(key): describe_tensors(item, depth + 1) for key, item in value.items()
                if hasattr(item, 'shape') or isinstance(item, (list, tuple))}
    if isinstance(value, (list, tuple)):
        return [describe_tensors(item, depth + 1) for item in value[:4]]
    for attribute in ('last_hidden_state', 'pooler_output'):
        if hasattr(value, attribute):
            return {attribute: describe_tensors(getattr(value, attribute), depth + 1)}
    return None


class IntervalRecorder:
    """Own one request's nested intervals and resolve CUDA events after its boundary.

    Args: cuda: torch.cuda-compatible backend or None; device: CUDA device; clock: seconds callable.
    Returns: Recorder with measure() and finish(). Raises: RuntimeError for invalid nesting/streams.
    Design: Context: Native stages nest deeply. Concern: Inclusive stages cannot be summed as a total.
        Decision: Preserve the tree and compute exclusive times from immediate children only.
        Alternatives: Independent synchronized timers. Quality Attributes: Causal accounting.
        Trade-offs: Stream elapsed includes launch gaps and does not equal kernel busy time.
    Evidence: Internal: Nested fake-event tests. External: CUDA event ordering.
        Findings: Serial children partition a parent except wrapper and native residual work.
        Metrics: Partition error and interval validity. Confidence: Mechanics validated separately from GPU.
    Constraints: Serial calls on one current stream. Reconsider When: Multi-stream policies are introduced.
    References: finish. ADR: None.
    """
    def __init__(self, cuda=None, device=None, clock=time.perf_counter):
        """Initialize empty request-local recording without allocating CUDA events yet.

        Args: cuda/device/clock: class contract. Returns: None. Raises: None.
        Design: Context: Warmup and measurement are separate. Concern: Cross-call stage leakage.
            Decision: One recorder per predict. Alternatives: Global accumulators.
            Quality Attributes: Isolation. Trade-offs: Small host allocations.
        Evidence: Internal: run_condition. External: None. Findings: No model state changes.
            Metrics: Interval count. Confidence: High.
        Constraints: No CUDA initialization for CPU tests. Reconsider When: Event pooling is needed.
        References: IntervalRecorder. ADR: None.
        """
        self.cuda = cuda
        self.device = device
        self.clock = clock
        self.intervals = []
        self.stack = []
        self.root_stream = None

    @contextmanager
    def measure(self, stage, details=None):
        """Record one nested interval while propagating every native exception.

        Args: stage: stable label; details: metadata mapping mutated by the caller if needed.
        Returns: Context manager yielding interval mapping. Raises: Native exception or stream violation.
        Design: Context: Every wrapped call must retain original semantics. Concern: Failed calls leak hooks.
            Decision: Finally closes both timestamps; restoration belongs to the outer wrapper context.
            Alternatives: Suppress native errors. Quality Attributes: Reliability. Trade-offs: Failed calls remain marked.
        Evidence: Internal: Exception restoration tests. External: CUDA event API.
            Findings: No synchronization is introduced. Metrics: Host and stream bounds. Confidence: High.
        Constraints: One serial stream. Reconsider When: Native async host threads enter a wrapper.
        References: NativeInstrumentation. ADR: None.
        """
        entry = {'stage': stage, 'parent': self.stack[-1] if self.stack else None,
                 'details': details if details is not None else {}, 'failed': False}
        entry['_host_start'] = self.clock()
        if self.cuda is not None:
            stream = self.cuda.current_stream(self.device)
            stream_id = int(stream.cuda_stream)
            if self.root_stream is None:
                self.root_stream = stream_id
            if stream_id != self.root_stream:
                raise RuntimeError('Nested timing requires one CUDA stream; concurrent-stream accounting is unsupported')
            entry['_event_start'] = self.cuda.Event(enable_timing=True)
            entry['_event_end'] = self.cuda.Event(enable_timing=True)
            entry['_event_start'].record(stream)
            entry['cuda_stream'] = stream_id
        index = len(self.intervals)
        self.intervals.append(entry)
        self.stack.append(index)
        try:
            yield entry
        except BaseException:
            entry['failed'] = True
            raise
        finally:
            if self.cuda is not None:
                end_stream = self.cuda.current_stream(self.device)
                if int(end_stream.cuda_stream) != self.root_stream:
                    raise RuntimeError('Native stage changed current CUDA stream; timing is unsupported')
                entry['_event_end'].record(end_stream)
            entry['_host_end'] = self.clock()
            self.stack.pop()

    def finish(self, tolerance_ms=0.1):
        """Resolve completed events and verify the nonoverlapping direct-child partition.

        Args: tolerance_ms: permissible floating event-resolution error.
        Returns: JSON-compatible intervals, stage aggregates and accounting validation.
        Raises: RuntimeError for open intervals; backend errors if caller omitted boundary synchronization.
        Design: Context: CUDA elapsed is available after outer completion. Concern: Nested double counting.
            Decision: Subtract direct children and expose the sum of exclusive intervals as a partition.
            Alternatives: Sum every inclusive stage. Quality Attributes: Auditability.
            Trade-offs: CUDA exclusive times are withheld if intervals cannot be validated.
        Evidence: Internal: Analytic nested-clock/event test. External: CUDA elapsed_time.
            Findings: Host exclusive partition equals root elapsed algebraically.
            Metrics: Partition error, child bounds and stream validity. Confidence: Mechanical identity.
        Constraints: Caller synchronizes once after predict, never here. Reconsider When: Streams overlap.
        References: IntervalRecorder.measure. ADR: None.
        """
        if self.stack:
            raise RuntimeError('Cannot finalize open profiling intervals')
        if not self.intervals:
            return {'intervals': [], 'stages': {}, 'accounting': {'valid': True}}
        root = self.intervals[0]
        rows = []
        for index, entry in enumerate(self.intervals):
            children = [child for child in self.intervals if child['parent'] == index]
            host_ms = (entry['_host_end'] - entry['_host_start']) * 1000
            children_ms = sum((child['_host_end'] - child['_host_start']) * 1000 for child in children)
            host_valid = all(entry['_host_start'] <= child['_host_start'] <= child['_host_end'] <= entry['_host_end']
                             for child in children)
            host_valid = host_valid and all(left['_host_end'] <= right['_host_start']
                                            for left, right in zip(children, children[1:]))
            row = {key: value for key, value in entry.items() if not key.startswith('_')}
            row.update(index=index, host_start_offset_ms=(entry['_host_start'] - root['_host_start']) * 1000,
                       host_ms=host_ms, host_exclusive_ms=host_ms - children_ms,
                       direct_child_count=len(children), valid=host_valid)
            if self.cuda is not None:
                elapsed = entry['_event_start'].elapsed_time(entry['_event_end'])
                start = root['_event_start'].elapsed_time(entry['_event_start'])
                child_elapsed = sum(child['_event_start'].elapsed_time(child['_event_end']) for child in children)
                bounds = [(entry['_event_start'].elapsed_time(child['_event_start']),
                           entry['_event_start'].elapsed_time(child['_event_end'])) for child in children]
                cuda_valid = all(-tolerance_ms <= a <= b <= elapsed + tolerance_ms for a, b in bounds)
                cuda_valid = cuda_valid and all(left[1] <= right[0] + tolerance_ms
                                                for left, right in zip(bounds, bounds[1:]))
                row.update(cuda_start_offset_ms=start, cuda_event_elapsed_ms=elapsed,
                           cuda_event_exclusive_ms=elapsed - child_elapsed if cuda_valid else None)
                row['valid'] = row['valid'] and cuda_valid and elapsed - child_elapsed >= -tolerance_ms
            rows.append(row)
        stages = {}
        for row in rows:
            aggregate = stages.setdefault(row['stage'], {'calls': 0, 'host_ms': 0., 'host_exclusive_ms': 0.})
            aggregate['calls'] += 1
            aggregate['host_ms'] += row['host_ms']
            aggregate['host_exclusive_ms'] += row['host_exclusive_ms']
            if self.cuda is not None:
                aggregate['cuda_event_elapsed_ms'] = aggregate.get('cuda_event_elapsed_ms', 0.) + row['cuda_event_elapsed_ms']
                aggregate['cuda_event_exclusive_ms'] = aggregate.get('cuda_event_exclusive_ms', 0.) + (row['cuda_event_exclusive_ms'] or 0.)
        host_partition = sum(row['host_exclusive_ms'] for row in rows)
        accounting = {'valid': all(row['valid'] for row in rows), 'host_root_ms': rows[0]['host_ms'],
                      'host_exclusive_partition_ms': host_partition,
                      'host_partition_error_ms': host_partition - rows[0]['host_ms'],
                      'cuda_semantics': 'elapsed interval on one CUDA stream, including launch/idle gaps; not kernel busy time',
                      'inclusive_stage_times_must_not_be_summed': True,
                      'exclusive_subtraction': 'immediate nonoverlapping children only; wrapper overhead remains in parent'}
        if self.cuda is not None:
            partition = sum(row['cuda_event_exclusive_ms'] or 0. for row in rows)
            accounting.update(cuda_root_event_elapsed_ms=rows[0]['cuda_event_elapsed_ms'],
                              cuda_exclusive_partition_ms=partition,
                              cuda_partition_error_ms=partition - rows[0]['cuda_event_elapsed_ms'])
        return {'intervals': rows, 'stages': stages, 'accounting': accounting}


class NativeInstrumentation:
    """Install scoped wrappers on one runtime without editing class or library source.

    Args: runtime: loaded native frozen runtime; model: smolvla/pi05/pi0fast; torch_module: torch.
    Returns: Context manager; set recorder before each request. Raises: Missing native hooks.
    Design: Context: The measured condition must preserve native inference. Concern: Persistent monkeypatches.
        Decision: Shadow instance attributes and restore exact previous instance state in reverse order.
        Alternatives: Global library monkeypatches. Quality Attributes: Isolation and exception safety.
        Trade-offs: Native architecture names are version-specific and fingerprinted in the report.
    Evidence: Internal: Installed LeRobot methods and restoration tests. External: Python descriptor semantics.
        Findings: Saving whether an attribute was inherited avoids retaining stale bound methods.
        Metrics: Hook list and actual invocation counts. Confidence: CPU restoration checks.
    Constraints: One runtime, serial inference, unchanged tensor arguments/results.
    Reconsider When: Installed source hash or entrypoint architecture changes.
    References: profile_inference.py. ADR: None.
    """
    def __init__(self, runtime, model, torch_module=None):
        """Store runtime ownership and an initially empty restoration stack.

        Args: runtime/model/torch_module: class contract. Returns: None. Raises: None.
        Design: Context: Hooks are entered after baseline. Concern: Accidental native edits.
            Decision: Defer all wrapping to __enter__. Alternatives: Constructor side effects.
            Quality Attributes: Predictability. Trade-offs: None.
        Evidence: Internal: Context restoration tests. External: None. Findings: Native runtime remains untouched.
            Metrics: None. Confidence: High.
        Constraints: Caller owns runtime lifetime. Reconsider When: Concurrent requests are needed.
        References: __enter__. ADR: None.
        """
        self.runtime = runtime
        self.model = model
        self.torch = torch_module
        self.recorder = None
        self.restorations = []
        self.hooks = []
        self.sampled_tokens = None

    def wrap(self, owner, attribute, stage, capture=False, retain_tokens=False):
        """Wrap one callable attribute, retaining its original descriptor restoration state.

        Args: owner/attribute: native callable; stage: label or argument classifier; capture: layouts;
            retain_tokens: hold sampler result reference until after request timing.
        Returns: None. Raises: AttributeError for missing callable.
        Design: Context: Bound methods and processor callables both need scoped interception.
            Concern: Rebinding __call__ on an instance does not intercept Python special lookup.
            Decision: Replace the runtime callable attribute or the native bound forward method itself.
            Alternatives: Class edits. Quality Attributes: Fidelity. Trade-offs: Modest Python overhead.
        Evidence: Internal: Native attribute call sites. External: Python special method lookup.
            Findings: No device value access. Metrics: Invocation count. Confidence: Direct interception.
        Constraints: Original arguments/results/exceptions pass through. Reconsider When: Native calls bypass attributes.
        References: NativeInstrumentation. ADR: None.
        """
        original = getattr(owner, attribute)
        had_own = attribute in vars(owner)
        old_own = vars(owner).get(attribute)

        @wraps(original)
        def wrapped(*args, **kwargs):
            """Time the native call while preserving its arguments, result and failure.

            Args: args/kwargs: unchanged native arguments. Returns: Original result. Raises: Original exception.
            Design: Context: A scoped profiling condition. Concern: Device synchronization distorts latency.
                Decision: Metadata-only capture and asynchronous events. Alternatives: Per-layer synchronization.
                Quality Attributes: Fidelity. Trade-offs: Wrapper overhead.
            Evidence: Internal: Paired output comparison. External: CUDA events.
                Findings: Value transfers are deferred. Metrics: Overhead ratio. Confidence: Measured per run.
            Constraints: No tensor arithmetic. Reconsider When: Native API changes.
            References: wrap. ADR: None.
            """
            if self.recorder is None:
                return original(*args, **kwargs)
            label = stage(args, kwargs) if callable(stage) else stage
            details = {}
            if capture:
                details['inputs'] = describe_tensors({'args': args, **kwargs})
                if self.torch is not None:
                    enabled = bool(self.torch.is_autocast_enabled('cuda'))
                    details['cuda_autocast_enabled'] = enabled
                    details['cuda_autocast_dtype'] = str(self.torch.get_autocast_dtype('cuda')) if enabled else None
            with self.recorder.measure(label, details):
                result = original(*args, **kwargs)
                if capture:
                    details['output'] = describe_tensors(result)
                if retain_tokens:
                    self.sampled_tokens = result
                return result

        setattr(owner, attribute, wrapped)
        self.restorations.append((owner, attribute, had_own, old_own))
        self.hooks.append({'owner': type(owner).__name__, 'attribute': attribute,
                           'stage': stage if isinstance(stage, str) else 'classified_transformer_forward'})

    def __enter__(self):
        """Resolve native stage boundaries and install wrappers, rolling back partial setup.

        Args: None. Returns: Self. Raises: Unsupported/missing native entrypoint errors.
        Design: Context: Three pinned architectures. Concern: Missing hooks yield misleading empty stages.
            Decision: Required native boundaries fail loudly; optional projector details are explicit.
            Alternatives: Guess stage cost from parameter count. Quality Attributes: Auditability.
            Trade-offs: Pinned source assumptions are exposed in metadata.
        Evidence: Internal: Installed LeRobot source. External: None.
            Findings: Both flow models cache prefix once; FAST uses cached token forwards.
            Metrics: Actual calls. Confidence: Source-confirmed, runtime counted.
        Constraints: No class/source edits. Reconsider When: Native structure changes.
        References: __exit__. ADR: None.
        """
        try:
            runtime = self.runtime
            policy = runtime.policy
            model = policy.model
            self.wrap(runtime, 'preprocess', 'saved_preprocessing', capture=True)
            self.wrap(runtime, 'postprocess', 'saved_postprocessing', capture=True)
            processor = getattr(runtime, 'libero_processor', getattr(runtime, 'env_processor', None))
            if processor is not None:
                self.wrap(processor, 'observation', 'libero_camera_rotation_and_state')
            if hasattr(runtime, 'prepare_observation'):
                self.wrap(runtime, 'prepare_observation', 'observation_preparation_inclusive')
            self.wrap(policy, 'predict_action_chunk', 'policy_predict_inclusive')
            self.wrap(policy, 'prepare_images' if self.model == 'smolvla' else '_preprocess_images',
                      'image_resize_and_policy_preparation', capture=True)
            if self.model == 'smolvla':
                bridge = model.vlm_with_expert
                vision = bridge.get_vlm_model().vision_model
                projector = bridge.get_vlm_model().connector
                self.wrap(model, 'sample_actions', 'action_sampling_inclusive')
                self.wrap(model, 'embed_prefix', 'prefix_embedding_inclusive', capture=True)
                self.wrap(model, 'denoise_step', 'flow_denoise_step')
            else:
                bridge = model.paligemma_with_expert
                vision = bridge.paligemma.model.vision_tower
                projector = bridge.paligemma.model.multi_modal_projector
                if self.model == 'pi05':
                    self.wrap(model, 'sample_actions', 'action_sampling_inclusive')
                    self.wrap(model, 'embed_prefix', 'prefix_embedding_inclusive', capture=True)
                    self.wrap(model, '_embed_images', 'camera_embedding_batch_inclusive', capture=True)
                    self.wrap(model, 'denoise_step', 'flow_denoise_step')
                else:
                    sampler = 'sample_actions_fast_kv_cache' if policy.config.use_kv_cache else 'sample_actions_fast'
                    self.wrap(model, sampler, 'autoregressive_sampling_inclusive', retain_tokens=True)
                    self.wrap(model, 'embed_prefix_fast', 'prefix_embedding_inclusive', capture=True)
                    self.wrap(policy, 'detokenize_actions', 'action_detokenization_inclusive')
                    self.wrap(policy, 'decode_actions_with_fast', 'fast_action_token_decode')
                    self.wrap(bridge.paligemma.lm_head, 'forward', 'fast_logits_projection')
            self.wrap(bridge, 'embed_image', 'vision_embedding_inclusive', capture=True)
            self.wrap(vision, 'forward', 'vision_tower', capture=True)
            self.wrap(projector, 'forward', 'vision_projector', capture=True)
            self.wrap(bridge, 'forward', self.transformer_stage, capture=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def transformer_stage(self, args, kwargs):
        """Classify native prefill, flow expert and FAST cached forwards without tensor reads.

        Args: args/kwargs: native bridge forward arguments. Returns: Stable stage name.
        Raises: None for the pinned keyword-based call sites.
        Design: Context: Bridge forward serves both prefix and action work.
            Concern: Treating all forwards as observation understanding double counts denoising.
            Decision: Read cache and embedding presence, never values. Alternatives: Infer from durations.
            Quality Attributes: Interpretability. Trade-offs: Keyword contract is source-pinned.
        Evidence: Internal: Native sample_actions and denoise_step source. External: None.
            Findings: Prefix embedding is absent for flow expert calls. Metrics: Counts. Confidence: High.
        Constraints: Native kwargs unchanged. Reconsider When: Positional bridge calls appear.
        References: __enter__. ADR: None.
        """
        embeddings = kwargs.get('inputs_embeds')
        if embeddings is not None and len(embeddings) > 1 and embeddings[0] is None:
            return 'flow_action_expert_forward'
        if self.model == 'pi0fast' and kwargs.get('past_key_values') is not None:
            return 'fast_cached_decode_forward'
        return 'vlm_prefix_prefill'

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore all instance attributes even when setup or prediction raises.

        Args: exc_type/exc_value/traceback: context protocol. Returns: False; exceptions propagate.
        Raises: Attribute restoration errors if native objects reject their prior state.
        Design: Context: Runtime may be reused after instrumentation. Concern: Persistent wrappers.
            Decision: Reverse restoration and delete inherited method shadows. Alternatives: Save bound method as own attr.
            Quality Attributes: Reversibility. Trade-offs: Serial ownership is required.
        Evidence: Internal: Successful and failing wrapper tests. External: Python instance dictionaries.
            Findings: Prior own values retain object identity. Metrics: Exact dictionary restoration. Confidence: High.
        Constraints: Do not suppress native exceptions. Reconsider When: Runtime objects use custom descriptors.
        References: wrap. ADR: None.
        """
        self.recorder = None
        while self.restorations:
            owner, attribute, had_own, old_own = self.restorations.pop()
            if had_own:
                setattr(owner, attribute, old_own)
            else:
                delattr(owner, attribute)
        return False
