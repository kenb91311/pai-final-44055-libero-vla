"""Check strict native checkpoint coverage without loading torch or model weights.

Args: Pytest runner. Returns: Assertions. Raises: Test failures.
Design: Context: VLASH legacy loading permits missing state. Concern: Silent random fallback.
    Decision: Exercise schema/alias rejection using tiny stand-ins. Alternatives: MultiGB GPU tests in unit suite.
    Quality Attributes: Fast integrity verification. Trade-offs: GPU inference remains a separate smoke.
Evidence: Internal: vlash_runtime.expand_aliases. External: Safetensors alias convention.
    Findings: Every state key must be accounted for and originally tied. Metrics: Seven cases.
    Confidence: Structural checks only.
Constraints: No model construction, torch import or network. Reconsider When: Restoration algorithm changes.
References: vlash_runtime.py. ADR: None.
"""
import json
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.realtime_vla.vlash_runtime import bind_aliases,checkpoint_header,expand_aliases


def test_explicit_original_alias_is_completed():
    """Accept only an alias already shared in the original architecture.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: Deduplicated checkpoint. Concern: Lost tied keys. Decision: Expand explicit original tie.
        Alternatives: Ignore keys. Quality Attributes: Coverage. Trade-offs: None.
    Evidence: Internal: Tiny shared stand-in. External: None. Findings: Same stored object is reused.
        Metrics: Two covered keys. Confidence: Deterministic.
    Constraints: No model imports. Reconsider When: Alias semantics change. References: expand_aliases. ADR: None.
    """
    parameter=SimpleNamespace(shape=(2,));saved=SimpleNamespace(shape=(2,))
    complete,aliases=expand_aliases({'__metadata__':{'tied':'main'}},{'main':parameter,'tied':parameter},{'main':saved})
    assert complete=={'main':saved,'tied':saved} and aliases=={'tied':'main'}


@pytest.mark.parametrize('case',['missing','unexpected','wrong_shape','not_shared','unknown_metadata'])
def test_incomplete_or_invented_state_is_rejected(case):
    """Reject schemas that could hide missing, extra or incorrectly shared weights.

    Args: case: invalid schema kind. Returns: None. Raises: AssertionError if accepted.
    Design: Context: Strict restoration. Concern: Random fallback or accidental sharing.
        Decision: Negative cases at the loader boundary. Alternatives: Successful construction alone.
        Quality Attributes: Integrity. Trade-offs: Values tested in separate smoke.
    Evidence: Internal: Five controlled corruptions. External: None. Findings: All fail closed.
        Metrics: Five rejection paths. Confidence: Deterministic.
    Constraints: Tiny stand-ins. Reconsider When: Key coverage changes. References: expand_aliases. ADR: None.
    """
    parameter=SimpleNamespace(shape=(2,));saved=SimpleNamespace(shape=(2,))
    expected={'main':parameter};stored={'main':saved};header={}
    if case=='missing':expected['unrestored']=parameter
    elif case=='unexpected':stored['alien']=saved
    elif case=='wrong_shape':stored['main']=SimpleNamespace(shape=(3,))
    elif case=='not_shared':
        expected['tied']=SimpleNamespace(shape=(2,));header={'__metadata__':{'tied':'main'}}
    else:header={'__metadata__':{'alien':'main'}}
    with pytest.raises((ValueError,RuntimeError)):expand_aliases(header,expected,stored)


def test_binding_restores_the_same_parameter_object():
    """Rebind validated native aliases to the canonical Parameter identity.

    Args: None. Returns: None. Raises: AssertionError.
    Design: Context: assign=True may split objects. Concern: Duplicate or divergent learned ties.
        Decision: Check explicit object identity after binding. Alternatives: Equal values only.
        Quality Attributes: Fidelity. Trade-offs: Original tie validation is tested separately.
    Evidence: Internal: Tiny module stand-ins. External: Torch parameter registration semantics.
        Findings: Destination points to canonical object. Metrics: One alias. Confidence: Deterministic.
    Constraints: No CUDA. Reconsider When: Binding implementation changes. References: bind_aliases. ADR: None.
    """
    canonical=object();modules={'base':SimpleNamespace(_parameters={'weight':canonical},_buffers={}),
                              'shared':SimpleNamespace(_parameters={'weight':object()},_buffers={})}
    policy=Mock();policy.get_submodule.side_effect=modules.__getitem__
    bind_aliases(policy,{'shared.weight':'base.weight'})
    assert modules['shared']._parameters['weight'] is canonical


def test_header_reads_only_structure(tmp_path):
    """Read a tiny structural header while ignoring arbitrary tensor payload bytes.

    Args: tmp_path: pytest temporary directory. Returns: None. Raises: AssertionError.
    Design: Context: Cheap preflight. Concern: Accidental large payload loading.
        Decision: Length-prefixed header fixture. Alternatives: Full safetensors loader.
        Quality Attributes: Resource isolation. Trade-offs: Content SHA tested at real restoration.
    Evidence: Internal: Minimal file fixture. External: Safetensors layout.
        Findings: Payload does not enter returned data. Metrics: Header equality. Confidence: Deterministic.
    Constraints: No external assets. Reconsider When: Format changes. References: checkpoint_header. ADR: None.
    """
    header={'w':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}
    content=json.dumps(header).encode();path=tmp_path/'model.safetensors'
    path.write_bytes(struct.pack('<Q',len(content))+content+b'payload')
    assert checkpoint_header(path)==header
