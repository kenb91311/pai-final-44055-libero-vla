"""Compare a frozen author base with and without per-tick A2C2 correction in actual time.

Args: Variant off/a2c2, pinned assets, declared timing and fixed initial states.
Returns: Separate new-series protocol, ticks/chunks, native outcomes and frozen audits.
Raises: Explicit startup/runtime/transport errors; failed trials remain in evidence.
Design: Context: A small head should react while the slow base generates asynchronously.
    Concern: Fast isolated inference is insufficient when GPU, ROS and physics share a deadline.
    Decision: Spawn base-only and CPU actuator; head proposes for the next tick in both variants.
    Alternatives: Block physics for base or change the14s budget after observing failure.
    Quality Attributes: Causal comparison, measurement. Trade-offs: Soft realtime with shared GPU.
Evidence: Internal: Split/runtime strict-restoration parity. External: A2C2 paper2509.23224.
    Findings: Policy actions use received base/packet values; derived controller holds are separate.
    Metrics: Native success, elapsed/sim time, queue holds, source age, head/ROS/step cost.
    Confidence: Small fixed-state case study, not a general model benchmark.
Constraints: No training, object-truth input or modifications to original35 trials.
Reconsider When: Sensor/action space or model pairing changes.
References: docs/reference/realtime-continuity-methods.md. ADR: None.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
if __package__:
    from .a2c2_worker import A2C2Worker, ActuatorWorker, validate_packet
    from .chunk_queue import ChunkQueue
    from .ros_client import RosRoundTrip
else:
    from a2c2_worker import A2C2Worker, ActuatorWorker, validate_packet
    from chunk_queue import ChunkQueue
    from ros_client import RosRoundTrip


def write_json(path, value):
    """Write strict JSON evidence including NumPy arrays and CLI paths.

    Args: path/value: output and finite data. Returns: None. Raises: IO/ValueError.
    Design: Context: Immutable evidence. Concern: Nonfinite or hidden stringified context.
        Decision: Explicit array/path conversion and forbid NaN. Alternatives: Console only.
        Quality Attributes: Traceability. Trade-offs: Rewrite only outside task timing.
    Evidence: Internal: Original evaluator convention. External: JSON.
        Findings: Source context is serialized explicitly. Metrics: None. Confidence: IO contract.
    Constraints: No image payloads. Reconsider When: Evidence storage changes.
    References: packet_record. ADR: None.
    """
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                   default=lambda x:x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n')


def packet_record(packet, **extra):
    """Convert a source-bound packet into explicit auditable numeric evidence.

    Args: packet: chunk/context result; extra: scheduling outcome.
    Returns: JSON-compatible mapping. Raises: Protocol errors.
    Design: Context: Python dataclasses are not JSON evidence.
        Concern: A string representation would truncate hidden context arrays.
        Decision: Save complete960feature, source generation and exact context hashes.
        Alternatives: Omit source feature. Quality Attributes: Provenance. Trade-offs: Small logs.
    Evidence: Internal: IPC parity. External: A2C2 source context requirement.
        Findings: Actions retain original50 indices. Metrics: Hashes. Confidence: High identity.
    Constraints: No model weights/images. Reconsider When: Context schema changes.
    References: A2C2ChunkContext. ADR: None.
    """
    context = packet['context']
    validate_packet(packet['actions'], context)
    result = {k:v for k,v in packet.items() if k != 'context'}
    result['context'] = {'generation':context.generation, 'hidden':context.vlm_hidden,
                         'actions_sha256':hashlib.sha256(context.actions.tobytes()).hexdigest(),
                         'hidden_sha256':hashlib.sha256(context.vlm_hidden.tobytes()).hexdigest()}
    return dict(result, **extra)


def choose_received_action(item, context, packet, tick):
    """Select only a matching on-time ROS packet, otherwise the received base or controller hold.

    Args: item/context: selected base source; packet: latest transport result; tick: physical now.
    Returns: Action, origin label and fallback/rejection reason.
    Raises: ValueError for malformed received actions.
    Design: Context: TCP response can outlive the intended actuation step.
        Concern: Reusing stale relative movement can add unintended displacement.
        Decision: Exact target/source/generation/index/deadline gate; no repeated old packet.
        Alternatives: Last available correction. Quality Attributes: Causal integrity.
        Trade-offs: Valid but late ROS receipts are deliberately discarded.
    Evidence: Internal: v1 DDS receipt preceded client timeout.
        External: A2C2 source alignment. Findings: Hold is an executor-derived rule, not a DDS claim.
        Metrics: Provenance and fallback reasons. Confidence: Unit-checkable contract.
    Constraints: No waiting, ROS calls or privileged state.
    Reconsider When: Action space changes. References: run_episode. ADR: None.
    """
    if item['held']:
        return item['action'].copy(), 'derived_controller_hold', 'empty_base_queue'
    fallback = item['action'].copy()
    if packet is None:
        return fallback, 'base_ros_chunk', 'packet_not_received'
    if packet['kind'] == 'timeout':
        return fallback, 'base_ros_chunk', 'transport_timeout'
    if packet['kind'] != 'action':
        raise ValueError('unusable actuator packet kind')
    if packet['target_tick'] != tick:
        return fallback, 'base_ros_chunk', 'wrong_target_tick'
    if packet['finished_time'] > packet['target_deadline']:
        return fallback, 'base_ros_chunk', 'receipt_after_deadline'
    if (packet['source_request_id'], packet['source_generation'], packet['original_action_index']) != (item['request_id'], context.generation, item['chunk_index']):
        return fallback, 'base_ros_chunk', 'source_replaced'
    action = np.asarray(packet['received_action'], dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all() or np.abs(action).max() > 1:
        raise ValueError('invalid received actuator value')
    origin = 'corrected_ros_packet' if packet['proposal_kind'] == 'correction' else 'base_action_ros_packet'
    return action.copy(), origin, None


def wait_sensor(env, camera, episode_id):
    """Refresh a coherent camera/state snapshot outside the timed task.

    Args: env/camera/episode_id: scene and identity. Returns: Sensor sample.
    Raises: TimeoutError for missing/mismatched frame.
    Design: Context: Warmup must not age the initial observation unnoticed.
        Concern: Camera mailbox may retain a previous reset's sample.
        Decision: Require same episode and capture after snapshot submission.
        Alternatives: Accept any queued frame. Quality Attributes: Causal integrity. Trade-offs: Startup.
    Evidence: Internal: CameraWorker source timestamps. External: None.
        Findings: RGB and proprioception share capture. Metrics: Time/tick. Confidence: High.
    Constraints: Untimed only. Reconsider When: Camera protocol changes.
    References: run_episode. ADR: None.
    """
    started = time.monotonic()
    camera.submit(env.snapshot(tick=0, episode_id=episode_id))
    deadline = started+60
    while True:
        sensor = camera.wait(timeout=max(.1, deadline-time.monotonic()))
        if sensor['episode_id'] == episode_id and sensor['capture_time'] >= started:
            return sensor
        if time.monotonic() > deadline:
            raise TimeoutError('initial coherent camera deadline')


def run_episode(env, camera, worker, head, actuator, args, index, out):
    """Run a common one-tick actuation pipeline while base generation and ROS progress separately.

    Args: owned physics/camera/base/head/CPU actuator; declared args/state/output.
    Returns: Native outcomes, executed provenance, proposal/receipt and timing evidence.
    Raises: Startup/runtime/integrity errors; transport timeout causes explicit base fallback.
    Design: Context: v1 stalled waiting for a TCP reply although DDS already delivered.
        Concern: The physical clock must not depend on ROS response or old relative commands.
        Decision: Propose for k+1, one inflight, source/deadline gate and received base fallback.
        Alternatives: Increase wall budget or repeat an old correction.
        Quality Attributes: Timeliness, transparency. Trade-offs: One-tick phase and dropped corrections.
    Evidence: Internal:276CPU RPC calls and v1 failed tick57.
        External: A2C2 per-source context. Findings: Derived holds are not claimed as DDS-received actions.
        Metrics: Native success, fallback/late counts, actual periods, head cost and RTF.
        Confidence: Measured task/system only.
    Constraints: No blocking transport/base wait inside the physical loop; head remains synchronous.
    Reconsider When: Strict real-time hardware is used.
    References: choose_received_action; ActuatorWorker. ADR: None.
    """
    import torch
    out.mkdir(parents=True, exist_ok=False)
    period = 1/args.control_hz
    max_ticks = round(args.wall_seconds*args.control_hz)
    episode_id = f'author-smol-{args.variant}-init{index}'
    obs = env.reset(init_state_index=index, seed=args.seed)
    sensor = wait_sensor(env, camera, episode_id)
    if not worker.warmed:
        for _ in range(2):
            worker.reset(args.seed)
            worker.predict(sensor, 0, sensor['capture_time'])
            warm = worker.wait()
            warm_record = {'metadata':warm['metadata'], 'worker_wall_s':warm['finished_time']-warm['started_time']}
            if head is not None:
                for _ in range(3):
                    _, cost = head.correct(sensor, sensor['prompt'], warm['context'], 0)
                    warm_record.setdefault('head_warmup', []).append(cost)
            worker.warmup_records.append(warm_record)
        worker.warmed = True
    obs = env.reset(init_state_index=index, seed=args.seed)
    sensor = wait_sensor(env, camera, episode_id)
    hashes = {key:hashlib.sha256(np.asarray(sensor[key]).tobytes()).hexdigest()
              for key in ('agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')}
    write_json(out/'initial.json', {'init_state_index':index, 'seed':args.seed, 'sensor_hashes':hashes,
                                  'metrics':obs['metrics'], 'environment':env.metadata, 'camera':camera.metadata})
    worker.reset(args.seed)
    if head is not None:
        head.reset(args.seed)
        if head.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(head.device)
    boot_started = time.monotonic()
    worker.predict(sensor, 0, sensor['capture_time'])
    first = worker.wait()
    bootstrap_s = time.monotonic()-boot_started
    buffer = ChunkQueue(50, 'aligned')
    buffer.accept(first['actions'], 0, first['observation_time'], 0, first['request_id'])
    contexts = {first['request_id']:first['context']}
    chunks = [packet_record(first, bootstrap=True, applied=True, ready_tick=0)]
    packets, ticks, switches = [], [], []
    writer = None
    probe = RosRoundTrip(port=args.ros_port, timeout=args.action_timeout)
    _, probe_receipt = probe.actions(np.zeros((1, 7)), request_id=episode_id+'-probe')
    write_json(out/'single-action-ros-probe.json', probe_receipt)
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(out/'rollout.mp4'), fps=args.control_hz, codec='libx264', macro_block_size=2)
    start = time.monotonic()
    next_tick, previous_tick, previous_action = start, None, None
    status, error, frames, stale = 'completed', None, 1, 0
    try:
        with (out/'ticks.jsonl').open('w') as stream:
            for tick in range(max_ticks):
                delay = next_tick-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if time.monotonic()-start >= args.wall_seconds:
                    break
                tick_started = time.monotonic()
                new = camera.poll()
                if new is not None and new['episode_id'] == episode_id and new['tick'] >= sensor['tick']:
                    sensor = new
                    frames += 1
                tick_sensor = sensor
                interval = None if previous_tick is None else tick_started-previous_tick
                previous_tick = tick_started
                ready = worker.poll() if worker.inflight else None
                changed = False
                if ready is not None:
                    validate_packet(ready['actions'], ready['context'])
                    count = buffer.accept(ready['actions'], ready['observation_tick'], ready['observation_time'], tick, ready['request_id'])
                    contexts = {ready['request_id']:ready['context']}
                    chunks.append(packet_record(ready, bootstrap=False, applied=count > 0, ready_tick=tick,
                                                actual_delay_ticks=tick-ready['observation_tick'], queued_actions=count))
                    changed = count > 0
                if len(buffer) <= args.prefetch_steps and not worker.inflight:
                    if tick_started-sensor['capture_time'] <= args.max_sensor_age:
                        if int(sensor['tick']) > tick:
                            raise ValueError('base request has future camera tick')
                        worker.predict(sensor, int(sensor['tick']), sensor['capture_time'], request_tick=tick)
                    else:
                        stale += 1
                depth = len(buffer)
                item = buffer.pop()
                context = None if item['held'] else contexts[item['request_id']]
                packet = actuator.poll() if actuator.inflight else None
                action, origin, reason = choose_received_action(item, context, packet, tick)
                if packet is not None:
                    packet.update(polled_tick=tick, used=origin in ('corrected_ros_packet', 'base_action_ros_packet'),
                                  rejection_reason=reason)
                    packets.append(packet)
                buffer.gripper = float(action[6])
                before = time.monotonic()
                obs = env.step(action)
                physics_done = time.monotonic()
                camera.submit(env.snapshot(tick=tick+1, episode_id=episode_id))
                task_done = bool(obs['success'] or obs['done'])
                proposal, correction, proposal_error = None, None, None
                target_deadline = next_tick+period
                if not task_done and tick+1 < max_ticks:
                    if actuator.inflight:
                        proposal = {'status':'not_submitted_worker_busy', 'target_tick':tick+1}
                    elif not buffer.queue:
                        proposal = {'status':'not_submitted_empty_queue', 'target_tick':tick+1}
                    else:
                        future = buffer.queue[0]
                        source = contexts[future['request_id']]
                        proposed = future['action'].copy()
                        try:
                            # Only coherent camera/state samples enter the head. A frame
                            # produced after this physical step may validly be at target k+1.
                            latest = camera.poll()
                            if latest is not None and latest['episode_id'] == episode_id and latest['tick'] >= sensor['tick']:
                                sensor = latest
                                frames += 1
                            if int(sensor['tick']) > tick+1:
                                raise ValueError('head request has future camera tick')
                            if head is not None:
                                proposed, correction = head.correct(sensor, sensor['prompt'], source, int(future['chunk_index']))
                            clipped = np.clip(proposed, -1, 1)
                            proposal = {'target_tick':tick+1, 'target_deadline':target_deadline,
                                        'source_request_id':future['request_id'], 'source_generation':source.generation,
                                        'original_action_index':int(future['chunk_index']), 'proposed_action':clipped.tolist(),
                                        'proposal_kind':'correction' if head is not None else 'base_action',
                                        'head_observation_tick':int(sensor['tick']), 'head_observation_time':float(sensor['capture_time']),
                                        'correction':correction, 'clip_max_excess':float(max(0., np.abs(proposed).max()-1.))}
                            if time.monotonic() >= target_deadline:
                                proposal['status'] = 'not_submitted_deadline_exhausted'
                            else:
                                request = actuator.submit(clipped, source_request_id=future['request_id'],
                                                          source_generation=source.generation,
                                                          original_action_index=int(future['chunk_index']),
                                                          target_tick=tick+1, target_deadline=target_deadline,
                                                          proposal_kind=proposal['proposal_kind'])
                                proposal.update(status='submitted', actuator_request_id=request)
                        except Exception as exc:
                            proposal_error = exc
                            proposal = dict(proposal or {}, status='proposal_runtime_error', error=f'{type(exc).__name__}: {exc}')
                control_done = time.monotonic()
                delta = None if previous_action is None else action-previous_action
                record = {
                    'tick':tick, 'wall_time_since_start':before-start, 'tick_interval_s':interval,
                    'deadline_lateness_s':max(0., tick_started-next_tick), 'tick_body_s':control_done-tick_started,
                    'tick_budget_overrun':control_done-tick_started > period, 'step_wall_s':physics_done-before,
                    'queue_before':depth, 'queue_after':len(buffer), 'held':item['held'],
                    'request_id':item['request_id'], 'chunk_index':item['chunk_index'],
                    'source_generation':None if item['held'] else context.generation,
                    'observation_tick':item['observation_tick'],
                    'base_observation_age_s':None if item['observation_time'] is None else before-item['observation_time'],
                    'current_camera_capture_tick':int(tick_sensor['tick']), 'current_camera_age_s':before-tick_sensor['capture_time'],
                    'action_origin':origin, 'fallback_reason':reason,
                    'actuator_request_id':packet['request_id'] if origin in ('corrected_ros_packet', 'base_action_ros_packet') else None,
                    'actuator_packet_kind':None if packet is None else packet['kind'],
                    'head_correction_for_next_tick':correction, 'next_proposal':proposal,
                    'action':action.tolist(), 'queued_base_action':item['action'].tolist(),
                    'translation_delta_l2':None if delta is None else float(np.linalg.norm(delta[:3])),
                    'rotation_delta_l2':None if delta is None else float(np.linalg.norm(delta[3:6])),
                    'gripper_delta_abs':None if delta is None else float(abs(delta[6])),
                    'chunk_switched':changed, 'success':bool(obs['success']), 'metrics':obs['metrics'],
                }
                if changed and delta is not None:
                    switches.append({key:record[key] for key in ('tick', 'translation_delta_l2', 'rotation_delta_l2', 'gripper_delta_abs')})
                ticks.append(record)
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
                previous_action = action.copy()
                if writer:
                    writer.append_data(np.concatenate([np.rot90(sensor['agentview_image'], 2), np.rot90(sensor['robot0_eye_in_hand_image'], 2)], axis=1))
                next_tick = max(target_deadline, control_done)
                if proposal_error is not None:
                    raise proposal_error
                if task_done:
                    break
    except Exception as exc:
        status, error = 'error', f'{type(exc).__name__}: {exc}'
        write_json(out/'error.json', {'error':error, 'traceback':traceback.format_exc()})
    finally:
        elapsed = time.monotonic()-start
        if writer:
            writer.close()
        if worker.inflight:
            try:
                chunks.append(packet_record(worker.wait(), bootstrap=False, applied=False, completed_after_episode=True))
            except Exception as exc:
                status, error = 'error', f'base drain failure: {type(exc).__name__}: {exc}'
        if actuator.inflight:
            try:
                packets.append(dict(actuator.wait(), used=False, completed_after_episode=True,
                                    rejection_reason='trial_finished'))
            except Exception as exc:
                status, error = 'error', f'actuator drain failure: {type(exc).__name__}: {exc}'
        write_json(out/'chunks.json', chunks)
        write_json(out/'actuator-packets.json', packets)
    intervals = [row['tick_interval_s'] for row in ticks if row['tick_interval_s'] is not None]
    corrections = [row['head_correction_for_next_tick']['correction_ms'] for row in ticks if row['head_correction_for_next_tick'] is not None]
    durations = [row['tick_body_s'] for row in ticks]
    base_age = [row['base_observation_age_s'] for row in ticks if row['base_observation_age_s'] is not None]
    steady_chunks = [chunk for chunk in chunks if not chunk.get('bootstrap')]
    successful_packets = [packet for packet in packets if packet['kind'] == 'action']
    origins = {name:sum(row['action_origin'] == name for row in ticks)
               for name in ('base_ros_chunk', 'base_action_ros_packet', 'corrected_ros_packet', 'derived_controller_hold')}
    source_proposals = {row['next_proposal']['actuator_request_id']:row['next_proposal']
                        for row in ticks if row['next_proposal'] and row['next_proposal'].get('status') == 'submitted'}
    head_ages = [row['wall_time_since_start']+start-source_proposals[row['actuator_request_id']]['head_observation_time']
                for row in ticks if row['action_origin'] == 'corrected_ros_packet']
    result = {
        'status':status, 'error':error, 'model':'author-smolvla450m', 'variant':args.variant,
        'protocol_version':getattr(args, 'protocol_version', 2), 'decode_mode':getattr(args, 'decode_mode', 'residual'),
        'actuation_pipeline_ticks':1,
        'mode':'realtime', 'init_state_index':index, 'seed':args.seed,
        'success':bool(obs['success']) and status == 'completed',
        'success_within_wall_budget':bool(obs['success']) and status == 'completed' and elapsed <= args.wall_seconds+period,
        'wall_budget_s':args.wall_seconds, 'wall_seconds':elapsed, 'sim_seconds':len(ticks)*period,
        'realtime_factor':len(ticks)*period/elapsed, 'control_hz':args.control_hz, 'ticks':len(ticks),
        'native_generation_horizon':50, 'execute_steps':50, 'merge':'aligned', 'prefetch_steps':args.prefetch_steps,
        'hold_ticks':buffer.holds, 'hold_fraction':buffer.holds/max(1, len(ticks)),
        'expired_prefix_actions':buffer.discarded_expired, 'replaced_actions':buffer.discarded_overlap,
        'action_origin_counts':origins, 'fallback_base_ticks':origins['base_ros_chunk'],
        'transport_timeout_packets':sum(packet['kind'] == 'timeout' for packet in packets),
        'late_or_source_rejected_packets':sum(packet.get('rejection_reason') in ('wrong_target_tick', 'receipt_after_deadline', 'source_replaced') for packet in successful_packets),
        'successful_packets_unused_after_trial':sum(packet.get('completed_after_episode', False) for packet in successful_packets),
        'successful_actuator_packets':len(successful_packets),
        'proposal_skipped_busy_ticks':sum(row['next_proposal'] is not None and row['next_proposal']['status'] == 'not_submitted_worker_busy' for row in ticks),
        'proposal_skipped_deadline_ticks':sum(row['next_proposal'] is not None and row['next_proposal']['status'] == 'not_submitted_deadline_exhausted' for row in ticks),
        'mean_control_interval_s':float(np.mean(intervals)) if intervals else None,
        'p95_control_interval_s':float(np.percentile(intervals, 95)) if intervals else None,
        'tick_budget_overrun_count':sum(row['tick_budget_overrun'] for row in ticks),
        'mean_tick_body_s':float(np.mean(durations)) if durations else None,
        'p95_tick_body_s':float(np.percentile(durations, 95)) if durations else None,
        'mean_head_correction_ms':float(np.mean(corrections)) if corrections else None,
        'p95_head_correction_ms':float(np.percentile(corrections, 95)) if corrections else None,
        'mean_action_ros_ms':float(np.mean([p['action_transport']['rpc_wall_seconds']*1000 for p in successful_packets])) if successful_packets else None,
        'mean_physics_ms':float(np.mean([row['step_wall_s'] for row in ticks])*1000) if ticks else None,
        'mean_base_source_age_s':float(np.mean(base_age)) if base_age else None,
        'mean_head_camera_age_s':float(np.mean(head_ages)) if head_ages else None,
        'mean_current_camera_age_s':float(np.mean([row['current_camera_age_s'] for row in ticks])) if ticks else None,
        'mean_base_worker_s':float(np.mean([c['finished_time']-c['started_time'] for c in steady_chunks])) if steady_chunks else None,
        'base_peak_cuda_bytes':max((c['metadata'].get('peak_cuda_bytes', 0) or 0 for c in chunks), default=0),
        'head_peak_cuda_bytes':torch.cuda.max_memory_allocated(head.device) if head is not None and head.device.type == 'cuda' else None,
        'bootstrap_seconds':bootstrap_s, 'sensor_frames_received':frames, 'stale_sensor_ticks':stale,
        'goal_xy_error_m':obs['metrics'].get('goal_xy_error_m'), 'final_metrics':obs['metrics'], 'switches':switches,
        'ros_transport':True, 'camera_mode':'asynchronous_clone', 'camera_hz':args.camera_hz,
        'input_boundary':'base RGB/state via ROS; head coherent camera IPC; received base or matching received1x7 correction; derived controller hold separately labeled',
        'hold_boundary':'zero6 plus last applied gripper is executor-derived and is not claimed as a full DDS-received7D command',
        'video_time_note':'raw video uses simulated tick rate; ticks.jsonl records actual time and realtime_factor',
    }
    write_json(out/'result.json', result)
    return result


def collect_audits(worker, head, output):
    """Save both process weight audits before closing resources, including on failed runs.

    Args: optional base/head runtimes and output directory.
    Returns: Base audit, head audit and explicit audit errors.
    Raises: IO failures. Design: Context: v1 runtime error occurred before final audits.
        Concern: Failure must not erase frozen-state evidence or fabricate a successful audit.
        Decision: Audit each available owner independently and retain unavailable results as errors.
        Alternatives: Success-only audit. Quality Attributes: Traceability. Trade-offs: Untimed hashing.
    Evidence: Internal: Exact runtime digests. External: None.
        Findings: An absent after digest is not proof of invariance.
        Metrics: Digest/gradient flags. Confidence: Exact identity when obtained.
    Constraints: Never infer an unavailable result. Reconsider When: Resource ownership changes.
    References: FrozenA2C2Runtime.audit. ADR: None.
    """
    base_audit = head_audit = None
    errors = {}
    if worker is not None:
        try:
            base_audit = worker.audit()
            write_json(output/'base-model-audit.json', base_audit)
        except Exception as exc:
            errors['base'] = f'{type(exc).__name__}: {exc}'
    if head is not None:
        try:
            head_audit = head.audit()
        except Exception as exc:
            errors['head'] = f'{type(exc).__name__}: {exc}'
    write_json(output/'head-model-audit.json', head_audit)
    if errors:
        write_json(output/'audit-errors.json', errors)
    return base_audit, head_audit, errors


def main():
    """Publish protocol before startup, then evaluate one independent author-base condition.

    Args: CLI described by --help. Returns: None and new evidence directory.
    Raises: Validation/startup/task errors. Design: Context: Serial exclusive benchmark runs.
        Concern: Overwriting original trials or using a different expert confounds conclusions.
        Decision: Exclusive output, pinned identities, separate strict audits and explicit boundaries.
        Alternatives: Reuse previous official Smol results as head-off control.
        Quality Attributes: Reproducibility. Trade-offs: Separate new-series runs.
    Evidence: Internal: Source hashes, per-condition seed and restored assets.
        External: Native LIBERO success. Findings: Same author base under both conditions.
        Metrics: All stages. Confidence: Restricted measured states.
    Constraints: GPU scheduling owned by caller; --protocol-only never initializes models.
    Reconsider When: Experimental protocol changes. References: run_episode. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--variant', choices=['off', 'a2c2'], required=True)
    parser.add_argument('--decode-mode', choices=['residual', 'direct'], default='residual')
    parser.add_argument('--protocol-version', choices=[2, 3], type=int, default=3)
    parser.add_argument('--modeldir', type=Path, default=Path('artifacts/realtime-vla/models/a2c2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-states', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--control-hz', type=int, choices=[20], default=20)
    parser.add_argument('--wall-seconds', type=float, default=14.)
    parser.add_argument('--prefetch-steps', type=int, default=30)
    parser.add_argument('--camera-hz', type=float, default=10.)
    parser.add_argument('--max-sensor-age', type=float, default=.5)
    parser.add_argument('--ros-port', type=int, default=8780)
    parser.add_argument('--action-timeout', type=float, default=.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--protocol-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0 or not 1 <= args.prefetch_steps < 50:
        parser.error('invalid wall budget or remaining-action trigger')
    if not math.isfinite(args.action_timeout) or args.action_timeout <= 0:
        parser.error('invalid actuator ROS timeout')
    if args.protocol_version == 2 and args.decode_mode != 'residual':
        parser.error('protocolv2 is fixed to residual decode; direct needs registeredv3')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(vars(args), model='author-smolvla450m', method='A2C2 candidate reproduction',
                    base_generation_horizon=50, execution_horizon=50, merge='aligned', prefix_conditioning=False,
                    actuation_pipeline_ticks=1, actuator_inflight_limit=1, actuator_backlog_limit=0,
                    task='LIBERO-Spatial task0 black bowl between plate and ramekin onto plate',
                    learning=False, success='native LIBERO check_success',
                    queue_empty='executor-derived zero6D delta and last actually applied gripper; not a DDS-received7D claim',
                    startup='two untimed base warmups; head warmups if enabled; final reset and bootstrap before wall clock',
                    input_boundary='base sensors via ROS; head coherent camera IPC; matching on-time received1x7 proposal or received base row; derived holds separately labeled',
                    contention='one VLA in child and small head in parent share GPU; latency/deadline/RTF measured without compensation',
                    limitation='author base differs from previous official SmolVLA; exact head/base SHA linkage not explicit upstream')
    write_json(args.output/'protocol.json', protocol)
    source_dir = Path(__file__).parent
    write_json(args.output/'source-sha256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in source_dir.glob('*.py')})
    if args.protocol_only:
        print(json.dumps({'status':'protocol_saved', 'variant':args.variant}))
        return
    for key, value in {'MUJOCO_GL':'egl', 'PYOPENGL_PLATFORM':'egl', 'OMP_NUM_THREADS':'4',
                       'OPENBLAS_NUM_THREADS':'1', 'GALLIUM_DRIVER':'d3d12', 'MESA_D3D12_DEFAULT_ADAPTER_NAME':'NVIDIA'}.items():
        os.environ.setdefault(key, value)
    if '/usr/lib/wsl/lib' not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    import torch
    if __package__:
        from .a2c2_runtime import FrozenA2C2Runtime
        from .libero_env import FrozenLiberoEnv
        from .camera_worker import CameraWorker
    else:
        from a2c2_runtime import FrozenA2C2Runtime
        from libero_env import FrozenLiberoEnv
        from camera_worker import CameraWorker
    torch.set_num_threads(4)
    worker = head = env = camera = actuator = None
    audit = head_audit = None
    results = []
    try:
        worker = A2C2Worker(args.modeldir, ros_port=args.ros_port, device=args.device)
        actuator = ActuatorWorker(ros_port=args.ros_port, timeout=args.action_timeout)
        if args.variant == 'a2c2':
            head = FrozenA2C2Runtime(args.modeldir, device=args.device, load_base=False, load_head=True,
                                    decode_mode=args.decode_mode)
        for index in args.initial_states:
            settings = {'horizon':round(args.wall_seconds*args.control_hz), 'control_freq':args.control_hz}
            env = FrozenLiberoEnv(**settings, render_mode='physics', output_dir=args.output/f'environment-init{index}')
            camera = CameraWorker(env_kwargs=settings, init_index=index, seed=args.seed, camera_hz=args.camera_hz)
            result = run_episode(env, camera, worker, head, actuator, args, index, args.output/f'realtime-init{index}')
            results.append(result)
            write_json(args.output/'summary.json', {'status':'running', 'results':results})
            print(json.dumps({k:result[k] for k in ('variant', 'init_state_index', 'status', 'success', 'realtime_factor', 'mean_head_correction_ms')}), flush=True)
            camera.close()
            camera = None
            env.close()
            env = None
            if result['status'] != 'completed':
                raise RuntimeError(result['error'])
        audit, head_audit, audit_errors = collect_audits(worker, head, args.output)
        if audit_errors or audit is None or not audit['weights_unchanged'] or (head_audit and not head_audit['weights_unchanged']):
            raise RuntimeError('frozen weights changed')
        write_json(args.output/'summary.json', {'status':'complete', 'results':results, 'base_audit':audit,
                                              'head_audit':head_audit, 'worker_ready':worker.ready,
                                              'actuator_ready':actuator.ready, 'warmup_records':worker.warmup_records})
    except Exception as exc:
        saved_traceback = traceback.format_exc()
        audit, head_audit, audit_errors = collect_audits(worker, head, args.output)
        write_json(args.output/'summary.json', {'status':'error', 'error':f'{type(exc).__name__}: {exc}',
                                              'traceback':saved_traceback, 'results':results,
                                              'base_audit':audit, 'head_audit':head_audit, 'audit_errors':audit_errors,
                                              'worker_ready':worker.ready if worker is not None else None,
                                              'actuator_ready':actuator.ready if actuator is not None else None,
                                              'warmup_records':worker.warmup_records if worker is not None else []})
        raise
    finally:
        if camera is not None:
            camera.close()
        if env is not None:
            env.close()
        if worker is not None:
            worker.close()
        if actuator is not None:
            actuator.close()


if __name__ == '__main__':
    main()
