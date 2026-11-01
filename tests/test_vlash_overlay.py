"""Check standalone and historical torch-path discovery without installing or importing models.

Args: Pytest. Returns: Path assertions. Raises: Test failures.
Design: Context: Public task runtime is independent; historical workspace used an optional torch fallback.
    Concern: Reproduction must not require that historical environment.
    Decision: Tiny filesystem fixtures for direct and optional-provider branches.
    Alternatives: Heavy fresh CUDA installation during a formal GPU run.
    Quality Attributes: Portability, resource isolation. Trade-offs: Full install proof is separate.
Evidence: Internal: inherited_sites. External: importlib wheel-layout semantics.
    Findings: Standalone needs one site root. Metrics: Three deterministic cases. Confidence: Path contract.
Constraints: No torch/network/GPU. Reconsider When: Package layout changes.
References: prepare_vlash_overlay.py. ADR: None.
"""
import pytest
from experiments.realtime_vla.prepare_vlash_overlay import inherited_sites


def test_standalone_task_torch_needs_no_other_environment(tmp_path):
    """Use only task purelib when CUDA torch is installed directly there.

    Args: tmp_path: temporary fixture. Returns: None. Raises: AssertionError.
    Design: Context: Public140pin setup. Concern: Hidden extra project prerequisite.
        Decision: Direct package fixture. Alternatives: Hardcoded legacy root.
        Quality Attributes: Portability. Trade-offs: No actual wheel install.
    Evidence: Internal: Wheel __init__ layout. External: None. Findings: One root.
        Metrics: Ordered paths. Confidence: Deterministic.
    Constraints: Stdlib only. Reconsider When: Torch layout changes. References: inherited_sites. ADR: None.
    """
    base=tmp_path/'task-site';package=base/'torch';package.mkdir(parents=True)
    origin=package/'__init__.py';origin.write_text('')
    assert inherited_sites(base,origin)==[base]


def test_legacy_provider_is_optional_and_discovered(tmp_path):
    """Keep an existing historical torch provider without assuming its directory name.

    Args: tmp_path: fixture. Returns: None. Raises: AssertionError.
    Design: Context: Measured private runtime. Concern: Packaging repair must not alter it.
        Decision: Discover arbitrary provider root. Alternatives: Require a named environment.
        Quality Attributes: Compatibility. Trade-offs: Only discovered provider is added.
    Evidence: Internal: Two-path fixture. External: None. Findings: Task packages remain first.
        Metrics: Exact order. Confidence: Deterministic.
    Constraints: No packages imported. Reconsider When: Overlay semantics change.
    References: inherited_sites. ADR: None.
    """
    base=tmp_path/'task-site';base.mkdir();provider=tmp_path/'existing-provider'
    package=provider/'torch';package.mkdir(parents=True);origin=package/'__init__.py';origin.write_text('')
    assert inherited_sites(base,origin)==[base,provider]


def test_missing_torch_provider_is_rejected(tmp_path):
    """Fail explicitly rather than invent another environment when torch is absent.

    Args: tmp_path: fixture. Returns: None. Raises: AssertionError if accepted.
    Design: Context: Missing prerequisite. Concern: Implicit unrelated dependencies.
        Decision: Reject missing provider. Alternatives: Guess paths.
        Quality Attributes: Predictability. Trade-offs: Task setup must run first.
    Evidence: Internal: Missing fixture. External: None. Findings: No writes.
        Metrics: Exception. Confidence: Deterministic.
    Constraints: No install. Reconsider When: Bootstrap changes. References: inherited_sites. ADR: None.
    """
    with pytest.raises(ValueError):inherited_sites(tmp_path,tmp_path/'torch/__init__.py')
