"""Validate and summarize the separate author-base A2C2 series with independent ROS receipts.

Args: Series root, bridge log root and optional register-only switch.
Returns: Compact results, exact-source provenance and matched receipt events.
Raises: Incomplete grid, protocol, model, context, transport or frozen-state errors.
Design: Context: Method comparison needs the same author base and five paired states.
    Concern: A2C2 and earlier official SmolVLA are different frozen models.
    Decision: Strict registered source/config grid, shared base receipt auditor and final-tick audit.
    Alternatives: Count publications or combine development with formal success rates.
    Quality Attributes: Reproducibility, traceability. Trade-offs: Large shared bridge logs are read.
Evidence: Internal: Raw episode, source context and both process audits.
    External: Independent ROS policy/broker event records.
    Findings: Received proposals and base rows are audited; derived holds have separate provenance.
    Metrics: Native success, paired hashes, head/control costs, RTF and per-process peak.
    Confidence: Five fixed states, not a population benchmark.
Constraints: Read-only raw evidence; no original35 changes or weight redistribution.
Reconsider When: Protocol or bridge schema changes.
References: evaluate_a2c2.py; summarize_continuity.audit_bridge. ADR: None.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

# The reused baseline summary helper uses script-local imports. This transient
# search path supports both python -m and direct script without an installed .pth.
LOCAL_SOURCE = Path(__file__).resolve().parent
if str(LOCAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE))
from summarize_continuity import ROOT, audit_bridge, load_json, source_record, statistics_of, wilson
from download_a2c2 import BASE_REPO, BASE_REV, HEAD_REPO, HEAD_REV, DIRECT_DECODE_UPSTREAM_REV
from ros_bridge import chunk_digest

SOURCE_FILES = ('a2c2_runtime.py', 'a2c2_worker.py', 'evaluate_a2c2.py', 'download_a2c2.py', 'summarize_a2c2.py',
                'chunk_queue.py', 'worker.py', 'libero_env.py', 'camera_worker.py', 'ros_client.py', 'ros_bridge.py')


def write_json(path, value):
    """Persist finite derivative evidence while keeping raw run files unchanged.

    Args: path/value: derivative output. Returns: None. Raises: IO/JSON errors.
    Design: Context: Traceable summaries. Concern: Nonfinite values.
        Decision: Strict JSON. Alternatives: Implicit stringification.
        Quality Attributes: Integrity. Trade-offs: None.
    Evidence: Internal: Read-only source records. External: JSON.
        Findings: Arrays are summarized explicitly. Metrics: None. Confidence: IO contract.
    Constraints: No raw file edits. Reconsider When: Storage changes.
    References: summarize. ADR: None.
    """
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def baseline_equivalence(reference):
    """Verify the reused head-off source's model generation and timed control are unchanged.

    Args: reference: completedv2 series with exact source snapshots.
    Returns: Detailed static-equivalence proof. Raises: ValueError on any unexplained change.
    Design: Context: v3 changes only the head output interpretation and metadata/API support.
        Concern: Reusing head-off results is invalid if base math or scheduling also changed.
        Decision: Exact shared-file hashes and AST equality of base/normalizer/control methods;
            remove only documented decode guard/metadata fields from constructor comparisons.
        Alternatives: Claim equivalence from matching CLI flags.
        Quality Attributes: Causal clarity. Trade-offs: Static proof does not equal simultaneous hardware state.
    Evidence: Internal: v2 registered byte snapshots and current source.
        External: None. Findings: Base checkpoint/normalizers and original runtime methods remain fixed.
        Metrics: Per-method and per-file equality. Confidence: Code equivalence within declared exclusions.
    Constraints: No model loading or prior-evidence modification.
    Reconsider When: Any scheduler/base implementation changes.
    References: register_protocol. ADR: None.
    """
    reference = Path(reference)
    snapshots = reference/'source-snapshot'
    old_protocol = load_json(reference/'registered-protocol.json')
    checks = {}
    for name in ('a2c2_worker.py', 'chunk_queue.py', 'worker.py', 'libero_env.py', 'camera_worker.py', 'ros_client.py', 'ros_bridge.py'):
        checks['file/'+name] = hashlib.sha256((LOCAL_SOURCE/name).read_bytes()).hexdigest() == old_protocol['source_sha256'][name]

    class Normalize(ast.NodeTransformer):
        """Remove only documentation and declared mode/metadata additions from compared ASTs.

        Args: AST nodes. Returns: Normalized nodes. Raises: None.
        Design: Context: Constructor/API metadata grew, math did not.
            Concern: Broad normalization could hide a real control change.
            Decision: Explicit field names and one decode-mode validation branch.
            Alternatives: Text diff only. Quality Attributes: Precision. Trade-offs: Narrow schema.
        Evidence: Internal: Source diff. External: Python AST.
            Findings: No model operations are erased. Metrics: AST equality. Confidence: Static.
        Constraints: Explicit exclusions only. Reconsider When: More changes arise.
        References: baseline_equivalence. ADR: None.
        """
        def visit_Expr(self, node):
            """Remove nonexecuting docstrings from method comparisons.

            Args: AST expression. Returns: Node/None. Raises: None.
            Design: Context: Documentation changes. Concern: False difference.
                Decision: String expressions only. Alternatives: Ignore entire method.
                Quality Attributes: Precision. Trade-offs: None.
            Evidence: Internal: AST. External: Python. Findings: Executable calls remain.
                Metrics: None. Confidence: Static. Constraints: Strings only.
            Reconsider When: Syntax changes. References: Class. ADR: None.
            """
            return None if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) else self.generic_visit(node)

        def visit_Dict(self, node):
            """Drop only added protocol/decode metadata keys.

            Args: Dict AST. Returns: Dict. Raises: None.
            Design: Context: Schema additions. Concern: Numeric result changes hidden.
                Decision: Named metadata keys only. Alternatives: Drop all dictionaries.
                Quality Attributes: Precision. Trade-offs: None.
            Evidence: Internal: Diff. External: AST. Findings: Action fields remain.
                Metrics: None. Confidence: Static. Constraints: Explicit list.
            Reconsider When: Schema changes. References: Class. ADR: None.
            """
            names = {'protocol_version', 'decode_mode', 'head_decode_reference_revision', 'head_output_semantics'}
            pairs = [(key, value) for key, value in zip(node.keys, node.values, strict=True)
                     if not isinstance(key, ast.Constant) or key.value not in names]
            node.keys, node.values = [p[0] for p in pairs], [p[1] for p in pairs]
            return self.generic_visit(node)

        def visit_If(self, node):
            """Exclude the constructor's new mode-name guard, which performs no model operation.

            Args: If AST. Returns: Node/None. Raises: None.
            Design: Context: New explicit parameter. Concern: Unexplained control branch.
                Decision: Test must refer to decode_mode. Alternatives: Drop all validation.
                Quality Attributes: Precision. Trade-offs: Narrow constructor-only usage.
            Evidence: Internal: Source diff. External: AST. Findings: Weight loading untouched.
                Metrics: None. Confidence: Static. Constraints: Applied only selected methods.
            Reconsider When: Guard gains behavior. References: Class. ADR: None.
            """
            if any(isinstance(n, ast.Name) and n.id == 'decode_mode' for n in ast.walk(node.test)):
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            """Exclude only storing the new decoder-name string on the runtime instance.

            Args: Assign AST. Returns: Node/None. Raises: None.
            Design: Context: Mode metadata. Concern: Ignore learned state changes.
                Decision: Exact self.decode_mode target only. Alternatives: Drop assignments.
                Quality Attributes: Precision. Trade-offs: None.
            Evidence: Internal: Source. External: AST. Findings: Model weights retained.
                Metrics: None. Confidence: Static. Constraints: Explicit attribute.
            Reconsider When: State contract changes. References: Class. ADR: None.
            """
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == 'decode_mode':
                return None
            return self.generic_visit(node)

    def methods(path, cls=None):
        """Extract normalized source methods without importing or running a model.

        Args: path/class name. Returns: Name-to-AST strings. Raises: Parse errors.
        Design: Context: CPU-only proof. Concern: GPU side effects.
            Decision: Parse bytes only. Alternatives: Instantiate models.
            Quality Attributes: Isolation. Trade-offs: Static evidence.
        Evidence: Internal: Saved code. External: AST. Findings: No weights read.
            Metrics: AST hashes. Confidence: Static. Constraints: No execution.
        Reconsider When: Source layout changes. References: baseline_equivalence. ADR: None.
        """
        tree = ast.parse(Path(path).read_text())
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body if cls else tree.body
        result = {}
        for node in body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.args.args and node.args.args[-1].arg == 'decode_mode':
                node.args.args.pop()
                node.args.defaults.pop()
            normalized = Normalize().visit(node)
            result[node.name] = ast.dump(normalized, include_attributes=False)
        return result

    old = methods(snapshots/'a2c2_runtime.py', 'FrozenA2C2Runtime')
    new = methods(LOCAL_SOURCE/'a2c2_runtime.py', 'FrozenA2C2Runtime')
    for name in ('__init__', 'predict', 'prepare_observation', '_capture_prefix', 'reset', 'weight_digest', 'audit'):
        checks['base_method/'+name] = old[name] == new[name]
    for cls in ('LegacyMeanStd',):
        checks['normalizer/'+cls] = methods(snapshots/'a2c2_runtime.py', cls) == methods(LOCAL_SOURCE/'a2c2_runtime.py', cls)
    old = methods(snapshots/'evaluate_a2c2.py')
    new = methods(LOCAL_SOURCE/'evaluate_a2c2.py')
    for name in ('run_episode', 'choose_received_action', 'wait_sensor', 'packet_record', 'collect_audits'):
        checks['control_method/'+name] = old[name] == new[name]
    if not all(checks.values()):
        raise ValueError('head-off baseline code changed: '+str([key for key, value in checks.items() if not value]))
    return {'all_checks_passed':True, 'checks':checks,
            'exclusions':['docstrings', 'explicit decode-mode name validation/storage', 'protocol/decode metadata fields'],
            'scope':'same base inference, normalization, queue/ROS/camera/physics and timed scheduler; hardware runs remain separate',
            'reference_protocol':source_record(reference/'registered-protocol.json'),
            'reference_results':source_record(reference/'results-compact.json')}


def register_protocol(series, reference=None):
    """Record the fixed experiment and relevant source hashes before any GPU task execution.

    Args: series: new owned directory. Returns: Protocol mapping.
    Raises: FileExistsError if already registered.
    Design: Context: Separate author-base method series.
        Concern: Retrospective parameter changes can look like method gains.
        Decision: Declare dev0/formal1..5, H50/R30/20Hz/14s and exact source identities.
        Alternatives: Reuse previous official expert baseline.
        Quality Attributes: Causal clarity. Trade-offs: Reruns require new directories.
    Evidence: Internal: Pinned base/head constants. External: Author source.
        Findings: No local training. Metrics: SHA256. Confidence: Exact registration.
    Constraints: No model loading. Reconsider When: A new protocol is intended.
    References: run_a2c2_series.sh. ADR: None.
    """
    series = Path(series)
    series.mkdir(parents=True, exist_ok=False)
    protocol = {'series':'a2c2-author-pair-v2-actuation-pipeline', 'protocol_version':2,
                'variants':['off', 'a2c2'], 'seed':20261102,
                'development_initial_states':[0], 'formal_initial_states':[1, 2, 3, 4, 5],
                'control_hz':20, 'wall_seconds':14., 'camera_hz':10., 'horizon':50,
                'prefetch_steps':30, 'merge':'aligned', 'prefix_conditioning':False,
                'actuation_pipeline_ticks':1, 'actuator_inflight_limit':1, 'actuator_backlog_limit':0,
                'base_repository':BASE_REPO, 'base_revision':BASE_REV,
                'head_repository':HEAD_REPO, 'head_revision':HEAD_REV,
                'source_sha256':{name:hashlib.sha256((LOCAL_SOURCE/name).read_bytes()).hexdigest() for name in SOURCE_FILES},
                'learning':False, 'formal_development_separate':True,
                'input_boundary':'base sensors via ROS; head coherent camera IPC; matching on-time received1x7 or received base row; derived hold separately labeled',
                'timeout_policy':'nonblocking base fallback; no old relative-action reuse and no timeout-driven wall-budget extension',
                'resource_boundary':'one resident large VLA in child and one small head in parent share GPU; actual overruns retained',
                'pair_note':'Author base is explicit in fixed source; exact head/base SHA linkage not published.'}
    write_json(series/'registered-protocol.json', protocol)
    if reference is not None:
        proof = baseline_equivalence(reference)
        protocol.update(series='a2c2-author-pair-v3-direct-decode', protocol_version=3, decode_mode='direct',
                        decode_reference_commit=DIRECT_DECODE_UPSTREAM_REV,
                        decode_formula='unnormalize(head_output); no base addition',
                        checkpoint_training_decode_mode_explicit=False,
                        interpretation_basis='Contemporaneous author MSE/direct code and checkpoint name; exact training-code linkage remains unpublished.',
                        reference_series=str(Path(reference).resolve().relative_to(ROOT)),
                        amendment='Only head output interpretation changes; v2 head-off baseline reused after static equivalence checks.')
        write_json(series/'baseline-equivalence.json', proof)
        write_json(series/'registered-protocol.json', protocol)
    snapshot = series/'source-snapshot'
    snapshot.mkdir()
    records = []
    for name, expected in protocol['source_sha256'].items():
        data = (LOCAL_SOURCE/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('source changed while protocol was being registered')
        (snapshot/name).write_bytes(data)
        records.append({'file':name, 'bytes':len(data), 'sha256':expected})
    write_json(snapshot/'manifest.json', {'all_registered_sources_exact':True, 'files':records})
    return protocol


def verify_frozen(audit, component):
    """Require correct component ownership, full restoration and unchanged frozen weights.

    Args: audit: runtime audit; component: base/head. Returns: None.
    Raises: ValueError for any failed condition.
    Design: Context: Independent CUDA processes have separate models/statistics.
        Concern: A head-only process must not hide another large base.
        Decision: Require zero gradients and exact declared component flags.
        Alternatives: Trust filenames. Quality Attributes: Auditability. Trade-offs: Strict schema.
    Evidence: Internal: Source runtime digest/restoration. External: Torch state_dict.
        Findings: No optimizer exists. Metrics: All restored tensors and digest equality.
        Confidence: Identity only; success is separately measured.
    Constraints: Read-only. Reconsider When: Quantized models are selected.
    References: FrozenA2C2Runtime.audit. ADR: None.
    """
    if not audit or not audit.get('weights_unchanged') or audit.get('local_training') is not False:
        raise ValueError(f'{component} frozen audit failed')
    if audit.get('requires_grad_parameter_count') != 0 or audit.get('parameters_with_grad') != 0:
        raise ValueError(f'{component} gradient audit failed')
    if (audit.get('load_base'), audit.get('load_head')) != ((True, False) if component == 'base' else (False, True)):
        raise ValueError(f'{component} process owns incorrect models')
    if not audit[component+'_restoration']['all_stored_tensors_match']:
        raise ValueError(f'{component} incomplete restoration')
    if (audit['base_model'], audit['base_revision'], audit['head_model'], audit['head_revision']) != (BASE_REPO, BASE_REV, HEAD_REPO, HEAD_REV):
        raise ValueError('checkpoint identity mismatch')


def audit_extra_receipts(episodes, run_summaries, bridge):
    """Match received actuator proposals, readiness probes and warmups to both ROS node receipts.

    Args: episode records/run summaries/bridge root. Returns: Audit and matched events.
    Raises: ValueError for missing, changed, unverified or miscorrelated receipts.
    Design: Context: Base chunk verification alone does not verify head-modified commands.
        Concern: Publishing telemetry could bypass actuator-side reception.
        Decision: Require policy and broker receipts by wire UUID, source hash and request id.
        Alternatives: Reuse base chunk hash for corrected action.
        Quality Attributes: Independent evidence. Trade-offs: Snapshot full append-only logs.
    Evidence: Internal: Per-tick client metadata. External: Actual ROS process event logs.
        Findings: Warmups are audited separately from task rates.
        Metrics: Exchanges/receipts by phase. Confidence: Direct log correlation.
    Constraints: No source rewriting. Reconsider When: Transport schema changes.
    References: summarize_continuity.audit_bridge. ADR: None.
    """
    index, sources = defaultdict(list), []
    for role in ('policy', 'broker'):
        path = Path(bridge)/role/'events.jsonl'
        data = path.read_bytes()
        if data and not data.endswith(b'\n'):
            raise ValueError('partial append-only bridge snapshot')
        sources.append(dict(source_record(path, captured_data=data), role=role, captured_prefix_semantics=True))
        for line in data.splitlines():
            event = json.loads(line)
            if 'id' in event:
                index[(role, event['event'], event['id'])].append(event)
    matched, counts, unacknowledged = [], defaultdict(int), []

    def verify(metadata, expected_request, phase, episode, sensor=False):
        """Check one exchange against two independent receipt roles.

        Args: metadata/request/phase/episode: recorded exchange; sensor: image exchange.
        Returns: None. Raises: ValueError for identity/hash mismatch.
        Design: Context: Final command hashes differ from base chunks.
            Concern: Wrong source correlation. Decision: Exact wire and request join.
            Alternatives: Receipt counts only. Quality Attributes: Integrity. Trade-offs: None.
        Evidence: Internal: Independent logs. External: ROS2.
            Findings: Both roles must agree. Metrics:2receipts/exchange. Confidence: High.
        Constraints: Read-only. Reconsider When: Bridge schema changes.
        References: audit_extra_receipts. ADR: None.
        """
        expected = metadata['source_sha256']
        if metadata.get('verified') is not True or metadata.get('request_id') != expected_request or not expected == metadata.get('received_sha256') == metadata.get('peer_received_sha256'):
            raise ValueError(f'invalid {phase} metadata receipt')
        wire = metadata['transport_id']
        names = ('sensor_join_receive', 'sensor_receipt') if sensor else ('generated_chunk_receive', 'action_receipt')
        for role, event_name in zip(('policy', 'broker'), names, strict=True):
            rows = index[(role, event_name, wire)]
            valid = [r for r in rows if r.get('verified') is True and r.get('source_sha256') == expected == r.get('received_sha256') and r.get('request_id') == expected_request]
            if not valid or len(valid) != len(rows):
                raise ValueError(f'missing/inconsistent independent {phase}/{role} receipt')
            matched.append({'episode':episode, 'phase':phase, 'request_id':expected_request,
                            'role':role, 'wire_transport_id':wire, 'event':valid[0]})
        counts[phase] += 1

    for episode in episodes:
        directory = ROOT/episode['source_directory']
        for packet in load_json(directory/'actuator-packets.json'):
            if packet['kind'] == 'action':
                verify(packet['action_transport'], packet['request_id'], 'actuator_proposal_received', episode['source_directory'])
            elif packet['kind'] == 'timeout':
                expected = chunk_digest([packet['proposed_action']])
                evidence = {}
                for role, event_name in (('policy', 'generated_chunk_receive'), ('broker', 'action_receipt')):
                    candidates = [event for (r, name, _), events in index.items() if r == role and name == event_name
                                  for event in events if event.get('request_id') == packet['request_id']]
                    valid = [event for event in candidates if event.get('verified') is True and event.get('source_sha256') == expected == event.get('received_sha256')]
                    if len(valid) != len(candidates):
                        raise ValueError('inconsistent receipt for timed-out proposal')
                    evidence[role] = valid
                unacknowledged.append({'episode':episode['source_directory'], 'request_id':packet['request_id'],
                                       'client_acknowledged':False, 'physically_used':False,
                                       'independent_receipts':evidence})
            else:
                raise ValueError('fatal actuator response in completed series')
        probe = load_json(directory/'single-action-ros-probe.json')
        verify(probe, probe['request_id'], 'untimed_probe', episode['source_directory'])
    for directory, summary in run_summaries:
        for warm in summary['warmup_records']:
            metadata = warm['metadata']
            request = metadata['sensor_transport']['request_id']
            verify(metadata['sensor_transport'], request, 'untimed_base_warmup_sensor', directory, sensor=True)
            verify(metadata['action_transport'], request, 'untimed_base_warmup_chunk', directory)
    return {'status':'passed', 'verified_exchanges_by_phase':dict(counts), 'verified_node_receipts':len(matched),
            'unacknowledged_proposals':unacknowledged,
            'scope':'all received development/formal actuator proposals, probes, warmups; derived holds are not DDS claims',
            'sources':sources}, matched


def summarize(series, bridge):
    """Recompute the complete formal grid and audit all development/formal ROS exchanges.

    Args: series/bridge: completed raw evidence and shared node logs.
    Returns: Compact derivative summary. Raises: Any protocol/evidence inconsistency.
    Design: Context: Head effect needs an identical author baseline.
        Concern: Different source snapshots or initial sensors confound attribution.
        Decision: Registered source checks, paired hashes and pooled versus episode stats.
        Alternatives: Use static paper numbers or the previous official expert.
        Quality Attributes: Causal clarity. Trade-offs: Small task-specific sample.
    Evidence: Internal: Raw ticks/chunks and exact source registry.
        External: Independent ROS receipts. Findings: Native success primary.
        Metrics: Success/period/freshness/RTF/peak/head and physical stages.
        Confidence: Descriptive only.
    Constraints: No raw evidence edits. Reconsider When: Registered condition changes.
    References: register_protocol. ADR: None.
    """
    series = Path(series)
    protocol = load_json(series/'registered-protocol.json')
    reference = ROOT/protocol['reference_series'] if 'reference_series' in protocol else None
    reference_protocol = load_json(reference/'registered-protocol.json') if reference else None
    if reference and not load_json(series/'baseline-equivalence.json')['all_checks_passed']:
        raise ValueError('reused baseline has no equivalence proof')
    episodes, formal, development, summaries, source_files = [], [], [], [], []
    audits, groups = {}, []
    for variant in protocol['variants']:
        variant_rows = []
        for phase, states in (('development', [0]), ('formal', protocol['formal_initial_states'])):
            reused = reference is not None and variant == 'off'
            directory = (reference if reused else series)/f'{phase}-{variant}'
            summary, config = load_json(directory/'summary.json'), load_json(directory/'protocol.json')
            manifest = load_json(directory/'source-sha256.json')
            if summary['status'] != 'complete' or config['variant'] != variant or config['initial_states'] != states:
                raise ValueError('incomplete or unexpected trial grid')
            if (config['control_hz'], config['wall_seconds'], config['camera_hz'], config['prefetch_steps'], config['seed']) != (20, 14., 10., 30, protocol['seed']):
                raise ValueError('run differs from fixed protocol')
            expected_version = 2 if reused or reference is None else 3
            if config.get('protocol_version') != expected_version or config.get('actuation_pipeline_ticks') != 1:
                raise ValueError('missing common one-tick pipeline')
            expected_decode = 'direct' if reference is not None and not reused else 'residual'
            if config.get('decode_mode', 'residual') != expected_decode:
                raise ValueError('declared head decode mode differs')
            for name, expected in (reference_protocol if reused else protocol)['source_sha256'].items():
                if manifest.get(name) != expected:
                    raise ValueError(f'registered source differs: {name}')
            verify_frozen(summary['base_audit'], 'base')
            if variant == 'a2c2':
                verify_frozen(summary['head_audit'], 'head')
                if reference is not None and summary['head_audit'].get('decode_mode') != 'direct':
                    raise ValueError('direct head audit has wrong interpretation')
            elif summary['head_audit'] is not None:
                raise ValueError('head-off run loaded head')
            key = f'{phase}/{variant}'
            audits[key] = {'base':summary['base_audit'], 'head':summary['head_audit']}
            summaries.append((str(directory.resolve().relative_to(ROOT)), summary))
            for state in states:
                episode_dir = directory/f'realtime-init{state}'
                result = load_json(episode_dir/'result.json')
                initial = load_json(episode_dir/'initial.json')
                rows = [json.loads(line) for line in (episode_dir/'ticks.jsonl').read_text().splitlines()]
                chunks = load_json(episode_dir/'chunks.json')
                packets = load_json(episode_dir/'actuator-packets.json')
                by_source = {chunk['request_id']:chunk for chunk in chunks if chunk.get('kind') == 'chunk'}
                by_packet = {packet['request_id']:packet for packet in packets}
                if result['status'] != 'completed' or result['variant'] != variant or result['init_state_index'] != state:
                    raise ValueError('failed or misidentified physical episode')
                if result.get('protocol_version', 2) != expected_version or result.get('decode_mode', 'residual') != expected_decode:
                    raise ValueError('episode decode/protocol differs from declared condition')
                previous_gripper = -1.
                for row in rows:
                    if not row['held'] and row['chunk_index'] != row['tick']-row['observation_tick']:
                        raise ValueError('original aligned chunk index was not retained')
                    if not row['held'] and row['source_generation'] != by_source[row['request_id']]['context']['generation']:
                        raise ValueError('executed base/source generation mismatch')
                    if max(abs(x) for x in row['action']) > 1:
                        raise ValueError('unclipped executed action')
                    origin = row['action_origin']
                    if origin == 'derived_controller_hold':
                        if not row['held'] or row['action'][:6] != [0.]*6 or row['action'][6] != previous_gripper:
                            raise ValueError('invalid derived executor hold')
                    elif origin == 'base_ros_chunk':
                        chunk = by_source[row['request_id']]
                        expected_action = np.clip(np.asarray(chunk['actions'][row['chunk_index']]), -1, 1)
                        if not np.array_equal(expected_action, row['action']):
                            raise ValueError('base fallback differs from ROS-received chunk row')
                    elif origin in ('corrected_ros_packet', 'base_action_ros_packet'):
                        packet = by_packet[row['actuator_request_id']]
                        if (origin == 'corrected_ros_packet') != (variant == 'a2c2'):
                            raise ValueError('head/base packet origin disagrees with condition')
                        if packet['kind'] != 'action' or not packet.get('used') or packet['target_tick'] != row['tick'] or packet['finished_time'] > packet['target_deadline']:
                            raise ValueError('stale or unreceived actuator packet applied')
                        if (packet['source_request_id'], packet['source_generation'], packet['original_action_index']) != (row['request_id'], row['source_generation'], row['chunk_index']):
                            raise ValueError('executed correction source/index mismatch')
                        if not np.array_equal(packet['received_action'], row['action']):
                            raise ValueError('executed correction differs from actual ROS receipt')
                    else:
                        raise ValueError('unknown executed action provenance')
                    previous_gripper = row['action'][6]
                for chunk in chunks:
                    if chunk.get('kind') != 'chunk':
                        continue
                    context = chunk['context']
                    if hashlib.sha256(np.asarray(chunk['actions'], dtype=np.float32).tobytes()).hexdigest() != context['actions_sha256']:
                        raise ValueError('chunk/context action hash mismatch')
                    if hashlib.sha256(np.asarray(context['hidden'], dtype=np.float32).tobytes()).hexdigest() != context['hidden_sha256']:
                        raise ValueError('source feature hash mismatch')
                source_directory = str(episode_dir.resolve().relative_to(ROOT))
                row = dict(result, phase=phase, source_directory=source_directory, initial_sensor_hashes=initial['sensor_hashes'],
                           reused_baseline=reused)
                row['effective_control_hz'] = result['ticks']/result['wall_seconds']
                row['effective_received_camera_hz'] = max(0, result['sensor_frames_received']-1)/result['wall_seconds']
                initial_z = initial['metrics'].get('bowl_position', [0, 0, 0])[2]
                lifts = [(r['metrics']['bowl_position'][2]-initial_z)*1000 for r in rows if 'bowl_position' in r['metrics']]
                row['maximum_bowl_lift_mm'] = max(lifts) if lifts else None
                tracking = [r['metrics']['eef_tracking_error_m']*1000 for r in rows if 'eef_tracking_error_m' in r['metrics']]
                row['mean_native_tracking_error_mm'] = float(np.mean(tracking)) if tracking else None
                episodes.append(row)
                (formal if phase == 'formal' else development).append(row)
                if phase == 'formal':
                    variant_rows.append(row)
                for name in ('initial.json', 'result.json', 'ticks.jsonl', 'chunks.json', 'actuator-packets.json', 'single-action-ros-probe.json'):
                    source_files.append(source_record(episode_dir/name))
        successes = sum(r['success'] for r in variant_rows)
        group = {'variant':variant, 'formal_n':len(variant_rows), 'native_successes':successes,
                 'decode_mode':'direct' if reference is not None and variant == 'a2c2' else 'residual' if variant == 'a2c2' else 'not_used',
                 'reused_baseline':reference is not None and variant == 'off',
                 'successes_within_wall_budget':sum(r['success_within_wall_budget'] for r in variant_rows),
                 'native_success_wilson95':wilson(successes, len(variant_rows)),
                 'success_by_state':{str(r['init_state_index']):r['success'] for r in variant_rows},
                 'pooled_hold_fraction':sum(r['hold_ticks'] for r in variant_rows)/sum(r['ticks'] for r in variant_rows),
                 'pooled_tick_overrun_fraction':sum(r['tick_budget_overrun_count'] for r in variant_rows)/sum(r['ticks'] for r in variant_rows),
                 'action_origin_counts':{name:sum(r['action_origin_counts'][name] for r in variant_rows)
                                         for name in ('base_ros_chunk', 'base_action_ros_packet', 'corrected_ros_packet', 'derived_controller_hold')},
                 'fallback_base_ticks':sum(r['fallback_base_ticks'] for r in variant_rows),
                 'transport_timeout_packets':sum(r['transport_timeout_packets'] for r in variant_rows),
                 'late_or_source_rejected_packets':sum(r['late_or_source_rejected_packets'] for r in variant_rows),
                 'proposal_skipped_deadline_ticks':sum(r['proposal_skipped_deadline_ticks'] for r in variant_rows),
                 'base_peak_cuda_bytes':max(r['base_peak_cuda_bytes'] for r in variant_rows),
                 'head_peak_cuda_bytes':max((r['head_peak_cuda_bytes'] or 0 for r in variant_rows), default=0)}
        for field in ('effective_control_hz', 'effective_received_camera_hz', 'realtime_factor', 'mean_head_correction_ms',
                      'p95_head_correction_ms', 'mean_action_ros_ms', 'mean_physics_ms', 'mean_control_interval_s',
                      'p95_control_interval_s', 'mean_tick_body_s', 'mean_base_source_age_s', 'mean_head_camera_age_s',
                      'goal_xy_error_m', 'maximum_bowl_lift_mm'):
            group[field+'_across_episode_values'] = statistics_of(r[field] for r in variant_rows if r.get(field) is not None)
        groups.append(group)
    paired = []
    for state in protocol['formal_initial_states']:
        off = next(r for r in formal if r['variant'] == 'off' and r['init_state_index'] == state)
        on = next(r for r in formal if r['variant'] == 'a2c2' and r['init_state_index'] == state)
        paired.append({'state':state, 'initial_sensor_hashes_equal':off['initial_sensor_hashes'] == on['initial_sensor_hashes'],
                       'off_success':off['success'], 'a2c2_success':on['success'],
                       'off_realtime_factor':off['realtime_factor'], 'a2c2_realtime_factor':on['realtime_factor']})
    chunk_audit, chunk_receipts = audit_bridge(episodes, bridge)
    chunk_audit['scope'] = 'all development/formal episode chunks including bootstrap/drained; warmups audited separately'
    extra_audit, extra_receipts = audit_extra_receipts(episodes, summaries, bridge)
    for filename, records in (('base-chunk-independent-receipts.jsonl', chunk_receipts), ('final-action-independent-receipts.jsonl', extra_receipts)):
        with (series/filename).open('w') as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
    compact = {'status':'complete', 'series':protocol['series'], 'formal_episodes':len(formal),
               'development_episodes':len(development), 'groups':groups, 'paired_initial_states':paired,
               'all_paired_initial_sensor_hashes_equal':all(r['initial_sensor_hashes_equal'] for r in paired),
               'development':[{k:r[k] for k in ('variant', 'init_state_index', 'status', 'success', 'realtime_factor')} for r in development],
               'independent_base_chunk_ros':chunk_audit, 'independent_final_action_ros':extra_audit,
               'total_verified_exchanges':chunk_audit['verified_exchanges']+sum(extra_audit['verified_exchanges_by_phase'].values()),
               'total_verified_node_receipts':chunk_audit['verified_node_receipts']+extra_audit['verified_node_receipts'],
               'model_audits':audits, 'source_files':source_files,
               'limitations':['Five paired fixed initial states of one task; descriptive intervals only.',
                              'Author450M base differs from previous official SmolVLA604M expert.',
                              'Head camera inputs use coherent camera IPC; base inputs and all final outputs use ROS.',
                              'Per-process peaks are separate maxima and must not be claimed as simultaneous total GPU usage.',
                              'Head and base share GPU; overruns and achieved simulation progress are retained.']}
    if reference is not None:
        previous = load_json(reference/'results-compact.json')
        previous_head = next(g for g in previous['groups'] if g['variant'] == 'a2c2')
        compact.update(reference_series=protocol['reference_series'], newly_evaluated_formal_episodes=5,
                       reused_baseline_formal_episodes=5, baseline_equivalence=load_json(series/'baseline-equivalence.json'),
                       previous_residual_diagnostic={'formal_n':previous_head['formal_n'],
                                                     'native_successes':previous_head['native_successes'],
                                                     'source':source_record(reference/'results-compact.json'),
                                                     'interpretation':'residual candidate; not a universal A2C2 performance claim'})
    write_json(series/'results-compact.json', compact)
    write_json(series/'episodes-recomputed.json', episodes)
    return compact


def main():
    """Register or validate one new A2C2 series without loading any GPU models.

    Args: CLI --series/--bridge/--register-only. Returns: None.
    Raises: Protocol/evidence errors. Design: Context: GPU lease belongs to executor.
        Concern: Summary tools should have no model side effects.
        Decision: Read-only evidence analysis. Alternatives: Automatic rerun.
        Quality Attributes: Isolation. Trade-offs: Incomplete series rejected.
    Evidence: Internal: Raw files. External: Independent node logs.
        Findings: No Torch model construction. Metrics: Audit counts. Confidence: Traceable.
    Constraints: Preserve all source evidence. Reconsider When: CLI changes.
    References: summarize. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--series', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, default=Path('artifacts/realtime-vla/pi05-ros-system'))
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--reference-series', type=Path, help='register direct-decodev3 while reusing equivalentv2 head-off baseline')
    args = parser.parse_args()
    result = register_protocol(args.series, reference=args.reference_series) if args.register_only else summarize(args.series, args.bridge)
    print(json.dumps({'status':'registered' if args.register_only else result['status'], 'series':str(args.series)}))


if __name__ == '__main__':
    main()
