"""Check shared profiling inputs and capture lifecycle with no simulator or GPU startup.

Args: pytest fixtures. Returns: Contract assertions. Raises: AssertionError.
Design: Context: The shared GPU is reserved for a separate active experiment.
    Concern: Serialization tests must neither load models nor initialize real rendering.
    Decision: Fake only the environment boundary while exercising real NPZ/JSON IO.
    Alternatives: Simulator integration belongs to root's later capture stage.
    Quality Attributes: Isolation, integrity. Trade-offs: No claim of renderer parity.
Evidence: Internal: capture_profile_inputs.py. External: None.
    Findings: Tests verify preserved raw bytes, allowlists and failure cleanup.
    Metrics: Five sensor arrays per sample. Confidence: CPU contract only.
Constraints: No simulator imports, model weights or GPU probes.
Reconsider When: Saved-input schema changes.
References: experiments/realtime_vla/capture_profile_inputs.py. ADR: None.
"""
from functools import partial
import hashlib
import json

import numpy as np
import pytest

from experiments.realtime_vla import capture_profile_inputs as capture


def raw_sensors(resolution=4, index=1):
    """Create asymmetric RGB and float64 robot sensing plus intentionally forbidden truth.

    Args: resolution: square image size; index: deterministic value offset.
    Returns: Raw fake observation including evaluator fields.
    Raises: None for fixture inputs.
    Design: Context: Rotations and casts must be detectable. Concern: Symmetric fixture images.
        Decision: Sequential pixel bytes and non-float32-exact proprio values.
        Alternatives: All-zero fixtures hide unintended conversion.
        Quality Attributes: Test sensitivity. Trade-offs: Synthetic observations.
    Evidence: Internal: Exact byte assertions. External: None. Findings: None.
        Metrics: dtype preservation. Confidence: Deterministic fixture.
    Constraints: Fixtures only. Reconsider When: Input schema changes.
    References: test_capture_roundtrip. ADR: None.
    """
    image = np.arange(resolution * resolution * 3, dtype=np.uint8).reshape(resolution, resolution, 3)
    return {"agentview_image": image, "robot0_eye_in_hand_image": image[::-1].copy(),
            "robot0_eef_pos": np.array([index / 7, .2, .3], dtype=np.float64),
            "robot0_eef_quat": np.array([0., 0., 0., 1.], dtype=np.float64),
            "robot0_gripper_qpos": np.array([.02, -.02], dtype=np.float64),
            "metrics": {"bowl_position": [1, 2, 3]}, "success": True,
            "object-state": np.ones(4), "prompt": "fixture prompt"}


class FakeEnv:
    """Record physics/camera lifecycle while returning deterministic raw sensor fixtures.

    Args: instances: shared instance log; invalid: corrupt sensor switch; kwargs: scene settings.
    Returns: Fake environment. Raises: None.
    Design: Context: Actual LIBERO would initialize GPU rendering.
        Concern: Test interference. Decision: Replace only the existing environment API.
        Alternatives: Mock the capture function. Quality Attributes: Isolation.
        Trade-offs: Does not validate actual simulator restore behavior.
    Evidence: Internal: Lifecycle assertions. External: None. Findings: None.
        Metrics: Closed flags and reset calls. Confidence: Unit boundary.
    Constraints: Tests only. Reconsider When: Environment API changes.
    References: capture_profile_inputs. ADR: None.
    """
    def __init__(self, instances, invalid=False, **kwargs):
        """Retain constructor settings and append this owned fake scene to its test log.

        Args: instances/invalid/kwargs: class inputs. Returns: None. Raises: None.
        Design: Context: Both scene lifetimes matter. Concern: Cleanup after failure.
            Decision: Log each instance. Alternatives: No lifecycle visibility.
            Quality Attributes: Testability. Trade-offs: Fixture state only.
        Evidence: Internal: Closed assertions. External: None. Findings: None.
            Metrics: Instance count. Confidence: Direct instrumentation.
        Constraints: CPU only. Reconsider When: API changes.
        References: FakeEnv. ADR: None.
        """
        self.settings = kwargs
        self.invalid = invalid
        self.closed = False
        self.metadata = {"prompt": "fixture prompt", "render_mode": kwargs["render_mode"],
                         "settle_steps": 10, "control_hz": kwargs["control_freq"]}
        instances.append(self)

    def reset(self, init_state_index, seed):
        """Remember the requested fixed reset without advancing hardware.

        Args: init_state_index/seed: reset identity. Returns: Raw fixture. Raises: None.
        Design: Context: Capture uses published states. Concern: Wrong seed/index wiring.
            Decision: Record arguments. Alternatives: Ignore reset.
            Quality Attributes: Observability. Trade-offs: Synthetic dynamics.
        Evidence: Internal: Capture assertions. External: None. Findings: None.
            Metrics: Reset identity. Confidence: Fixture.
        Constraints: CPU only. Reconsider When: Reset API changes.
        References: FakeEnv. ADR: None.
        """
        self.index, self.seed = init_state_index, seed
        return raw_sensors(self.settings["resolution"], self.index)

    def snapshot(self, tick, episode_id):
        """Produce a fake flattened physical snapshot with the same transport fields.

        Args: tick/episode_id: snapshot identity. Returns: Snapshot mapping. Raises: None.
        Design: Context: Camera restoration must use source physics.
            Concern: Reading unrelated camera reset sensing.
            Decision: Encode state index into snapshot. Alternatives: Fixed snapshot.
            Quality Attributes: Test sensitivity. Trade-offs: No MuJoCo state.
        Evidence: Internal: Restored index assertions. External: None. Findings: None.
            Metrics: Tick and simulation time. Confidence: Fixture.
        Constraints: CPU only. Reconsider When: Snapshot API changes.
        References: FakeEnv.observe_snapshot. ADR: None.
        """
        return {"physics_state": np.array([self.index, self.seed], dtype=np.float64),
                "tick": tick, "episode_id": episode_id, "simulation_time": .5}

    def observe_snapshot(self, snapshot):
        """Decode source-state identity into synchronized fake raw sensing.

        Args: snapshot: source physics mapping. Returns: Raw fixture, optionally malformed.
        Raises: None. Design: Context: Capture uses restored sensing.
            Concern: Reading stale cached sensors. Decision: Derive index from snapshot.
            Alternatives: Return reset cache. Quality Attributes: Test sensitivity.
            Trade-offs: Synthetic restore. Evidence: Internal: Byte comparisons.
            External: None. Findings: None. Metrics: Sensor hashes. Confidence: Fixture.
        Constraints: CPU only. Reconsider When: Restore API changes.
        References: FakeEnv.snapshot. ADR: None.
        """
        observation = raw_sensors(self.settings["resolution"], int(snapshot["physics_state"][0]))
        if self.invalid:
            observation["robot0_eef_pos"] = np.zeros(2)
        return observation

    def close(self):
        """Expose explicit scene cleanup to the test without real graphics resources.

        Args: None. Returns: None. Raises: None.
        Design: Context: Profiling follows capture. Concern: Lingering scene resources.
            Decision: Record closure. Alternatives: Rely on object collection.
            Quality Attributes: Testability. Trade-offs: No real context cleanup.
        Evidence: Internal: Closed flags. External: None. Findings: None.
            Metrics: Closure state. Confidence: Fixture.
        Constraints: CPU only. Reconsider When: Close API changes.
        References: FakeEnv. ADR: None.
        """
        self.closed = True


def test_capture_roundtrip(tmp_path):
    """Save two fixed samples with exact bytes, no truth, verified hashes and closed scenes.

    Args: tmp_path: isolated destination root. Returns: None. Raises: AssertionError.
    Design: Context: All model profilers reuse these files. Concern: Silent input divergence.
        Decision: Compare original/reloaded bytes and NPZ/manifest hashes.
        Alternatives: Shape-only tests miss rotation or casting.
        Quality Attributes: Integrity. Trade-offs: Fake simulator.
    Evidence: Internal: Real NPZ/JSON IO. External: None. Findings: Measured by assertions.
        Metrics: Byte equality for ten arrays. Confidence: CPU contract.
    Constraints: No GPU. Reconsider When: Schema changes.
    References: capture_profile_inputs. ADR: None.
    """
    instances = []
    output = tmp_path / "inputs"
    manifest = capture.capture_profile_inputs(output, [1, 2], resolution=4,
                                              env_factory=partial(FakeEnv, instances))
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert not (output / ".manifest.json.tmp").exists()
    assert manifest["simulators_closed_before_manifest"] and all(env.closed for env in instances)
    assert manifest["environment"] == {"task_suite": "libero_spatial", "task_id": 0,
                                       "control_hz": 20, "resolution": 4, "settle_steps": 10,
                                       "seed": 20261102, "initial_states": [1, 2]}
    assert all(env.seed == 20261102 for env in instances)
    assert [env.settings["render_mode"] for env in instances] == ["physics", "sync"] * 2
    for sample in manifest["samples"]:
        path = output / sample["file"]
        saved = capture.load_profile_input(path, resolution=4)
        expected = raw_sensors(4, sample["init_state_index"])
        assert set(saved) == set(capture.MODEL_INPUT_KEYS)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample["sha256"]
        assert capture.sensor_hashes(saved, 4) == sample["sensor_hashes"]
        for key, value in saved.items():
            assert value.dtype == expected[key].dtype
            assert value.tobytes() == expected[key].tobytes()
            assert value.flags.owndata and value.flags.c_contiguous


@pytest.mark.parametrize("problem", ["missing", "shape", "image_dtype", "nonfinite", "integer_state"])
def test_sensor_validation_rejects_malformed_inputs(problem):
    """Reject malformed raw sensors before serialization or model preparation.

    Args: problem: selected malformed sensor. Returns: None. Raises: AssertionError.
    Design: Context: Sensor shape/dtype defines policy inputs. Concern: Silent coercion.
        Decision: Exercise each contract category. Alternatives: Coerce malformed data.
        Quality Attributes: Integrity. Trade-offs: Invalid capture must be restarted.
    Evidence: Internal: ValueError assertions. External: None. Findings: None.
        Metrics: Five rejection categories. Confidence: CPU contract.
    Constraints: No GPU. Reconsider When: Allowed dtype/schema changes.
    References: model_input. ADR: None.
    """
    raw = raw_sensors()
    if problem == "missing":
        raw.pop("robot0_eef_quat")
    elif problem == "shape":
        raw["robot0_eef_pos"] = np.zeros(4)
    elif problem == "image_dtype":
        raw["agentview_image"] = raw["agentview_image"].astype(float)
    elif problem == "nonfinite":
        raw["robot0_eef_pos"][0] = np.nan
    else:
        raw["robot0_eef_pos"] = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError):
        capture.model_input(raw, 4)


@pytest.mark.parametrize("problem", ["extra", "object"])
def test_loader_forbids_truth_and_pickle(tmp_path, problem):
    """Reject additional evaluator fields and object arrays at the NPZ loading boundary.

    Args: tmp_path: isolated files; problem: extra field or pickle payload.
    Returns: None. Raises: AssertionError.
    Design: Context: Files may come from another capture version.
        Concern: Expanded inputs or object pickle loading.
        Decision: Exact allowlist and allow_pickle=False must reject both.
        Alternatives: Filter extra archive fields silently.
        Quality Attributes: Input integrity. Trade-offs: Strict schema.
    Evidence: Internal: Real np.savez/np.load. External: None. Findings: None.
        Metrics: Two rejected archives. Confidence: CPU contract.
    Constraints: Object payload is harmless fixture text. Reconsider When: Schema versions grow.
    References: load_profile_input. ADR: None.
    """
    sensors = capture.model_input(raw_sensors(), 4)
    if problem == "extra":
        sensors["success"] = np.array(True)
    else:
        sensors["robot0_eef_pos"] = np.array(["pickle", "is", "forbidden"], dtype=object)
    path = tmp_path / "bad.npz"
    np.savez(path, **sensors)
    with pytest.raises(ValueError):
        capture.load_profile_input(path, 4)


def test_existing_output_refused_before_environment_start(tmp_path):
    """Preserve even an empty existing output directory without constructing any scene.

    Args: tmp_path: existing destination. Returns: None. Raises: AssertionError.
    Design: Context: Artifacts must stay immutable. Concern: Accidental overwrite.
        Decision: Exclusive mkdir precedes factory use. Alternatives: Reuse empty folders.
        Quality Attributes: Reliability. Trade-offs: Caller supplies a new destination.
    Evidence: Internal: Instance log stays empty. External: None. Findings: None.
        Metrics: Zero constructed environments. Confidence: CPU lifecycle test.
    Constraints: No simulator. Reconsider When: Explicit resumable capture is designed.
    References: capture_profile_inputs. ADR: None.
    """
    instances = []
    with pytest.raises(FileExistsError):
        capture.capture_profile_inputs(tmp_path, env_factory=partial(FakeEnv, instances))
    assert not instances
    assert not list(tmp_path.iterdir())


def test_failed_capture_closes_scenes_without_publishing_manifest(tmp_path):
    """Release both scenes after malformed sensing and leave no completed manifest.

    Args: tmp_path: isolated files. Returns: None. Raises: AssertionError.
    Design: Context: Later profiling must only consume completed captures.
        Concern: A failure can leak scenes or advertise incomplete inputs.
        Decision: Finally cleanup and deferred atomic manifest publication.
        Alternatives: Write completed metadata before capture finishes.
        Quality Attributes: Reliability. Trade-offs: Partial directory remains for inspection.
    Evidence: Internal: Fake closed flags and manifest absence. External: None.
        Findings: None. Metrics: Both scenes closed. Confidence: CPU failure path.
    Constraints: No GPU. Reconsider When: Resume semantics are added.
    References: capture_profile_inputs. ADR: None.
    """
    instances = []
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="invalid shape"):
        capture.capture_profile_inputs(output, [1], resolution=4,
                                       env_factory=partial(FakeEnv, instances, invalid=True))
    assert len(instances) == 2 and all(env.closed for env in instances)
    assert not (output / "manifest.json").exists()
