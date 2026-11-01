"""Test independent ROS receipt audit rejection and append-only snapshot provenance.

Args: pytest temporary files. Returns: Assertions. Raises: AssertionError on audit regressions.
Design: Context: A report must not certify metadata when node evidence disagrees.
    Concern: A weak join could create a false verified result.
    Decision: Positive four-receipt case plus a corrupted independent receiver.
    Alternatives: Only check that JSON parses rejected.
    Quality Attributes: Evidence integrity. Trade-offs: Synthetic logs prove audit logic only.
Evidence: Internal: summarize_continuity.audit_bridge. External: Wire ID vs model request distinction.
    Findings: Both roles and both transport directions are required.
    Metrics: One chunk, two exchanges, four node receipts. Confidence: Contract-level.
Constraints: No real DDS/model/task evidence. Reconsider When: Bridge schema changes.
References: summarize_continuity.py. ADR: None.
"""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def test_independent_receiver_hash_mismatch_rejects_otherwise_verified_metadata(tmp_path,monkeypatch):
    """Require actual policy/broker receipt hashes rather than trusting a worker boolean.

    Args: tmp_path/monkeypatch: isolated source/log fixtures. Returns: None.
    Raises: AssertionError. Design: Context: Independent node correlation.
        Concern: Request IDs and wire IDs differ. Decision: Use explicit wire join.
        Alternatives: Count publications rejected. Quality Attributes: Audit reliability.
        Trade-offs: Fixture hash values are synthetic.
    Evidence: Internal: Four correct events then one corrupt receive hash.
        External: ROS bridge schema. Findings: Corruption changes audit pass to failure.
        Metrics:2exchanges/4receipts. Confidence: High for join contract.
    Constraints: Test-local files. Reconsider When: Schema changes.
    References: audit_bridge. ADR: None.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/realtime_vla'))
    module=importlib.import_module('summarize_continuity');monkeypatch.setattr(module,'ROOT',tmp_path)
    source=tmp_path/'run/realtime-init1';source.mkdir(parents=True)
    metadata={side:{'transport_id':wire,'source_sha256':sha,'received_sha256':sha,'peer_received_sha256':sha,'verified':True}
              for side,wire,sha in [('sensor_transport','sensor-wire','sensors-sha'),('action_transport','action-wire','actions-sha')]}
    (source/'chunks.json').write_text(json.dumps([{'kind':'chunk','request_id':'model-query','metadata':metadata}]))
    bridge=tmp_path/'bridge';logs={}
    for role in ('policy','broker'):
        folder=bridge/role;folder.mkdir(parents=True)
        events=[]
        names=('sensor_join_receive','generated_chunk_receive') if role=='policy' else ('sensor_receipt','action_receipt')
        for name,wire,sha in zip(names,('sensor-wire','action-wire'),('sensors-sha','actions-sha'),strict=True):
            events.append({'event':name,'id':wire,'request_id':'model-query','source_sha256':sha,'received_sha256':sha,'verified':True})
        logs[role]=events;(folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    audit,matched=module.audit_bridge([{'source_directory':'run/realtime-init1'}],bridge)
    assert (audit['task_chunks'],audit['verified_exchanges'],audit['verified_node_receipts'])==(1,2,4)
    assert len(matched)==4 and {r['role'] for r in matched}=={'policy','broker'}
    logs['policy'][0]['received_sha256']='corrupt-independent-receive'
    (bridge/'policy/events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in logs['policy']))
    with pytest.raises(ValueError,match='independent policy'):
        module.audit_bridge([{'source_directory':'run/realtime-init1'}],bridge)


def test_append_only_log_provenance_hashes_the_captured_prefix(tmp_path,monkeypatch):
    """Preserve verifiable snapshot length/hash when the shared bridge later appends events.

    Args: tmp_path/monkeypatch. Returns: None. Raises: AssertionError.
    Design: Context: Several experiment batches reuse the same running ROS bridge.
        Concern: A later append makes whole-file SHA different without altering old evidence.
        Decision: Store captured byte length and hash of that exact prefix.
        Alternatives: Treat append as corruption rejected. Quality Attributes: Traceability.
        Trade-offs: Verifier reads a byte prefix.
    Evidence: Internal: Prefix snapshot then additional event bytes.
        External: SHA256. Findings: Captured identity stays reproducible.
        Metrics: Byte length/hash. Confidence: High.
    Constraints: Append-only semantics. Reconsider When: Logs are rewritten.
    References: source_record. ADR: None.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/realtime_vla'))
    module=importlib.import_module('summarize_continuity');monkeypatch.setattr(module,'ROOT',tmp_path)
    path=tmp_path/'events.jsonl';captured=b'{"event":"first"}\n';path.write_bytes(captured)
    path.write_bytes(captured+b'{"event":"later"}\n')
    record=module.source_record(path,captured_data=captured)
    assert record['bytes']==len(captured) and record['sha256']==hashlib.sha256(path.read_bytes()[:record['bytes']]).hexdigest()


def test_overlap_disagreement_is_same_time_and_excludes_unknown_future(monkeypatch):
    """Separate a planned0.2 move from a same-time old/new disagreement of0.05.

    Args: monkeypatch: scoped module import. Returns: None. Raises: AssertionError.
    Design: Context: Action[t]-action[t-1] mixes ordinary motion and chunk replacement.
        Concern: A hold has no remaining old planned motion to compare.
        Decision: Same-time old-next/new-current comparison, separate after-hold count.
        Alternatives: Invent future actions after exhaustion rejected.
        Quality Attributes: Interpretability. Trade-offs: Overlap subset only.
    Evidence: Internal: Two-action old plan and one known switch.
        External: None. Findings:0.25actual step is only0.05different from old planned step.
        Metrics:0.05translation/1afterhold. Confidence: High for reconstruction.
    Constraints: Command units, not physical trajectory derivatives.
    Reconsider When: Action modification rules change.
    References: overlap_switch_records. ADR: None.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/realtime_vla'))
    module=importlib.import_module('summarize_continuity')
    old=[[0,0,0,0,0,0,-1],[.2,0,0,0,0,0,-1]]
    chunks=[{'kind':'chunk','request_id':'old','actions':old}]
    ticks=[{'tick':0,'request_id':'old','chunk_index':0,'held':False,'chunk_switched':False,'action':old[0]},
           {'tick':1,'request_id':'new','chunk_index':1,'held':False,'chunk_switched':True,'action':[.25,0,0,0,0,0,-1]},
           {'tick':2,'request_id':None,'chunk_index':None,'held':True,'chunk_switched':False,'action':[0,0,0,0,0,0,-1]},
           {'tick':3,'request_id':'later','chunk_index':2,'held':False,'chunk_switched':True,'action':[.4,0,0,0,0,0,1]}]
    records,afterhold=module.overlap_switch_records(ticks,chunks)
    assert len(records)==1 and records[0]['translation_l2']==pytest.approx(.05)
    assert records[0]['rotation_l2']==0 and records[0]['gripper_abs']==0 and afterhold==1


def test_development0_requires_explicit_mode_and_cannot_hide_unknown_states(monkeypatch):
    """Keep the formal denominator5 and reject unknown, missing, duplicate or reordered samples.

    Args: monkeypatch: import fixture. Returns: None. Raises: AssertionError.
    Design: Context: Clean rerun keeps0..5 in one model process.
        Concern: A filtering shortcut could hide an undeclared trial or count development success.
        Decision: Exact grid/order check and explicit development list.
        Alternatives: Silently intersect recorded/formal sets rejected.
        Quality Attributes: Scientific sample integrity. Trade-offs: Rigid registered order.
    Evidence: Internal: Default reject0, explicit accept0, and four malformed grids.
        External: None. Findings: Formal states remain exactly1..5.
        Metrics:5formal/1excluded. Confidence: Contract-level.
    Constraints: No runtime/model/simulator input. Reconsider When: Formal protocol changes.
    References: validate_summary_state_order. ADR: None.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/realtime_vla'))
    module=importlib.import_module('summarize_continuity');formal=[1,2,3,4,5];combined=[0]+formal
    with pytest.raises(ValueError):module.validate_summary_state_order(combined,combined,formal)
    separate=module.validate_summary_state_order(combined,combined,formal,allow_development_state0=True)
    assert separate=={'formal':formal,'development':[0]} and len(separate['formal'])==5
    for invalid in [[0,1,2,3,4,5,6],[0,1,2,3,4],[0,1,2,3,4,5,5],[1,0,2,3,4,5]]:
        with pytest.raises(ValueError):module.validate_summary_state_order(invalid,invalid,formal,allow_development_state0=True)
