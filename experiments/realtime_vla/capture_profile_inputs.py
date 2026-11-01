"""Save shared, raw LIBERO sensors before any frozen-model profiling process starts.

Args: CLI output directory, published initial-state indices, seed and RGB resolution.
Returns: Five-key NPZ samples and an atomically completed manifest.json.
Raises: FileExistsError for an existing destination; sensor/simulator failures propagate.
Design:
    Context: SmolVLA, pi0.5 and pi0-FAST need the same observation bytes for timing.
    Concern: Rendering or simulation during inference confounds model timing and memory.
    Decision: Reset physics, restore its snapshot in a separate synchronous sensor scene,
        save only raw cameras/proprioception, then close both scenes before completion.
    Alternatives: Cached reset observations use a different sensor sampling phase.
    Quality Attributes: Reproducibility, isolation. Trade-offs: Capture has its own startup cost.
Evidence:
    Internal: FrozenLiberoEnv.observe_snapshot and CameraWorker use the same restore API.
    External: LIBERO regenerate_obs_from_state, invoked through the existing wrapper.
    Findings: Raw proprioception dtype is preserved so historical byte hashes can match.
    Metrics: Per-sensor SHA256, NPZ SHA256 and historical initial-sensor byte comparison.
    Confidence: CPU contract tests; actual renderer parity is reported at capture time.
Constraints: No model imports, training, camera rotation or object truth in saved model input.
Reconsider When: Official policy raw observation keys or sensor-phase semantics change.
References: libero_env.py; camera_worker.py; artifacts/realtime-vla/formal-smolvla-e10.
ADR: None.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_INPUT_KEYS = (
    "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
    "robot0_eef_quat", "robot0_gripper_qpos",
)
PRIOR_BASELINE = ROOT / "artifacts/realtime-vla/formal-smolvla-e10"


def model_input(observation, resolution=256):
    """Copy only validated raw camera/proprioception arrays from a simulator observation.

    Args: observation: raw mapping; resolution: square RGB width and height.
    Returns: Exactly five independently owned, C-contiguous numeric arrays.
    Raises: ValueError for missing/invalid sensors or nonfinite proprioception.
    Design: Context: Simulator observations also contain privileged evaluator fields.
        Concern: Accidental object-state leakage or dtype-changing serialization.
        Decision: Explicit allowlist, exact uint8 RGB and preserved floating proprio dtype.
        Alternatives: Saving the entire observation would require pickle and expose truth.
        Quality Attributes: Integrity, comparability. Trade-offs: No joint-state aliases.
    Evidence: Internal: FrozenLiberoEnv raw sensor contract. External: None.
        Findings: Existing initial.json hashes use raw array bytes without conversion.
        Metrics: Shapes, dtypes and finite-value checks. Confidence: CPU contract tests.
    Constraints: Images remain unrotated; prompt is stored separately in the manifest.
    Reconsider When: A checkpoint requires additional raw sensors.
    References: load_profile_input. ADR: None.
    """
    if type(resolution) is not int or resolution < 1:
        raise ValueError("resolution must be a positive integer")
    sensors = {}
    shapes = ((resolution, resolution, 3), (resolution, resolution, 3), (3,), (4,), (2,))
    for key, shape in zip(MODEL_INPUT_KEYS, shapes):
        if key not in observation:
            raise ValueError(f"missing model sensor: {key}")
        array = np.asarray(observation[key])
        if array.shape != shape:
            raise ValueError(f"invalid shape for {key}: {array.shape}, expected {shape}")
        if key.endswith("image"):
            if array.dtype != np.uint8:
                raise ValueError(f"{key} must be uint8 RGB")
        elif array.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(array).all():
            raise ValueError(f"{key} must be finite float32 or float64")
        sensors[key] = np.array(array, copy=True, order="C")
    return sensors


def sensor_hashes(observation, resolution=256):
    """Hash the five raw sensor byte streams using the existing benchmark convention.

    Args: observation: sensor mapping; resolution: expected square camera size.
    Returns: Sensor-key to hexadecimal SHA256 mapping.
    Raises: ValueError for invalid sensors.
    Design: Context: Saved samples must establish exact cross-model input identity.
        Concern: Images alone miss proprioception and implicit float casts.
        Decision: Hash each validated array's C-order bytes separately.
        Alternatives: One aggregate hash obscures which sensor changed.
        Quality Attributes: Auditability. Trade-offs: Shape/dtype are recorded separately.
    Evidence: Internal: evaluate.py initial_sensing. External: SHA256.
        Findings: This matches historical initial.json byte hashing.
        Metrics: Five hashes per sample. Confidence: CPU roundtrip tests.
    Constraints: No normalization, rotation or float conversion before hashing.
    Reconsider When: Historical byte-hash conventions change.
    References: model_input. ADR: None.
    """
    return {key: hashlib.sha256(array.tobytes()).hexdigest()
            for key, array in model_input(observation, resolution).items()}


def load_profile_input(path, resolution=256):
    """Read a saved sample with pickle disabled and reject any additional model fields.

    Args: path: NPZ file; resolution: expected square RGB size.
    Returns: Five-key raw sensor dict with independent arrays and original numeric dtype.
    Raises: ValueError for extra/missing keys, object arrays or malformed sensors; IO errors.
    Design: Context: Three profilers independently reload the same immutable artifact.
        Concern: Pickle or extra evaluator fields undermine the shared-input boundary.
        Decision: np.load allow_pickle=False plus exact key-set and sensor validation.
        Alternatives: Object-dict NPZ serialization requires unsafe pickle loading.
        Quality Attributes: Isolation, reproducibility. Trade-offs: NPZ metadata lives in JSON.
    Evidence: Internal: CPU object-array/allowlist and roundtrip tests. External: NumPy NPZ.
        Findings: Returned arrays stay valid after archive close.
        Metrics: Exact array byte equality. Confidence: CPU contract tests.
    Constraints: The caller obtains task prompt from manifest.json, never object truth.
    Reconsider When: The saved input schema is versioned beyond version1.
    References: model_input. ADR: None.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        if len(archive.files) != len(MODEL_INPUT_KEYS) or set(archive.files) != set(MODEL_INPUT_KEYS):
            raise ValueError("profile NPZ must contain exactly the five model sensor keys")
        return model_input({key: archive[key] for key in MODEL_INPUT_KEYS}, resolution)


def _historical_comparison(index, hashes, seed, resolution):
    """Record historical sensor parity without treating old observations as new input.

    Args: index: initial state; hashes: new sensors; seed/resolution: capture settings.
    Returns: File provenance, per-key match flags and explicit comparable/unavailable status.
    Raises: IO/JSON errors for an existing malformed baseline.
    Design: Context: Prior formal realtime SmolVLA bootstrap hashes already exist.
        Concern: Direct synchronous reset sensing can differ from scene-clone restoration.
        Decision: Compare all raw sensor hashes and record mismatch honestly.
        Alternatives: Claiming historical identity from matching task names is insufficient.
        Quality Attributes: Traceability. Trade-offs: No historical arrays are reconstructed.
    Evidence: Internal: formal-smolvla-e10/realtime-init*/initial.json. External: None.
        Findings: Results are evaluated at capture time.
        Metrics: Per-sensor exact SHA equality. Confidence: Byte comparison only.
    Constraints: Baseline files are read-only; seed/resolution differences mark noncomparability.
    Reconsider When: Another canonical historical baseline is selected.
    References: capture_profile_inputs. ADR: None.
    """
    path = PRIOR_BASELINE / f"realtime-init{index}" / "initial.json"
    if not path.is_file():
        return {"status": "unavailable", "file": str(path.relative_to(ROOT))}
    content = path.read_bytes()
    baseline = json.loads(content)
    matches = {key: baseline.get("sensor_hashes", {}).get(key) == value for key, value in hashes.items()}
    comparable = baseline.get("seed") == seed and resolution == 256
    return {"status": "compared" if comparable else "settings_differ",
            "file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(content).hexdigest(),
            "baseline_seed": baseline.get("seed"), "baseline_resolution": 256,
            "sensor_hashes_equal": matches, "all_sensor_hashes_equal": all(matches.values())}


def capture_profile_inputs(output, initial_states=(1, 2, 3, 4, 5), seed=20261102,
                           resolution=256, env_factory=None):
    """Capture fixed task0 observations and close every simulator before finalizing metadata.

    Args: output: fresh directory; initial_states: unique nonnegative official indices;
        seed: reset seed; resolution: RGB size; env_factory: optional CPU test substitute.
    Returns: Completed manifest mapping; files are exclusively written under output.
    Raises: FileExistsError for existing output; ValueError for bad settings/sensors;
        simulator or IO errors propagate with no completed manifest.
    Design: Context: Capture is a separate executable stage before model profiling.
        Concern: Live simulator GPU contexts would contaminate inference resource measures.
        Decision: Fresh physics and camera scenes per initial state, use official ten-step
            settling and snapshot restoration, then close scenes before manifest publication.
        Alternatives: CameraWorker subprocess adds scheduling without changing restoration.
        Quality Attributes: Fairness, resource hygiene. Trade-offs: Two scene startups per state.
    Evidence: Internal: CameraWorker._render_process and FrozenLiberoEnv.reset.
        External: Official LIBERO task0 initial states. Findings: Capture executes no policy.
        Metrics: NPZ/sensor SHA256, capture duration and historical sensor parity.
        Confidence: Fake-environment lifecycle tests; actual hashes measured by root capture.
    Constraints: Canonical20Hz, settle10, raw sensor allowlist; never overwrite any destination.
    Reconsider When: Multiple task suites become part of the profiling protocol.
    References: load_profile_input; camera_worker.py. ADR: None.
    """
    indices = tuple(initial_states)
    if not indices or any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("initial_states must be unique nonnegative integer indices")
    if type(seed) is not int or type(resolution) is not int or resolution < 1:
        raise ValueError("seed must be an integer and resolution must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if env_factory is None:
        if __package__:
            from .libero_env import FrozenLiberoEnv
        else:
            from libero_env import FrozenLiberoEnv
        env_factory = FrozenLiberoEnv
    started = time.perf_counter()
    samples, provenance = [], []
    prompt = None
    settings = {"task_suite": "libero_spatial", "task_id": 0, "horizon": 280,
                "resolution": resolution, "control_freq": 20, "output_dir": None}
    for index in indices:
        physics = camera = None
        try:
            physics = env_factory(**settings, render_mode="physics")
            camera = env_factory(**settings, render_mode="sync")
            camera.reset(init_state_index=index, seed=seed)
            physics.reset(init_state_index=index, seed=seed)
            snapshot = physics.snapshot(tick=0, episode_id=f"profile-init{index}")
            observation = camera.observe_snapshot(snapshot)
            sensors = model_input(observation, resolution)
            sample_prompt = physics.metadata["prompt"]
            if not isinstance(sample_prompt, str) or not sample_prompt.strip():
                raise ValueError("task prompt must be a nonempty string")
            if prompt is not None and sample_prompt != prompt:
                raise ValueError("task prompt changed between initial states")
            prompt = sample_prompt
            sample_id = f"init{index}"
            filename = f"{sample_id}.npz"
            path = output / filename
            with path.open("xb") as handle:
                np.savez(handle, **sensors)
            hashes = sensor_hashes(sensors, resolution)
            reloaded = load_profile_input(path, resolution)
            if sensor_hashes(reloaded, resolution) != hashes:
                raise ValueError("saved sensor byte verification failed")
            samples.append({"id": sample_id, "file": filename,
                            "init_state_index": index, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "sensor_hashes": hashes,
                            "sensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                        for key, value in sensors.items()}})
            provenance.append({"id": sample_id, "physics_environment": dict(physics.metadata),
                               "camera_environment": dict(camera.metadata),
                               "snapshot": {"tick": int(snapshot["tick"]),
                                            "simulation_time": float(snapshot["simulation_time"]),
                                            "physics_state_sha256": hashlib.sha256(np.asarray(snapshot["physics_state"]).tobytes()).hexdigest()},
                               "historical_initial_comparison": _historical_comparison(index, hashes, seed, resolution)})
        finally:
            try:
                if camera is not None:
                    camera.close()
            finally:
                if physics is not None:
                    physics.close()
    source_paths = (Path(__file__).resolve(), ROOT / "experiments/realtime_vla/libero_env.py",
                    ROOT / "experiments/realtime_vla/camera_worker.py")
    manifest = {"version": 1, "status": "complete", "prompt": prompt, "samples": samples,
                "environment": {"task_suite": "libero_spatial", "task_id": 0,
                                "control_hz": 20, "resolution": resolution, "settle_steps": 10,
                                "seed": seed, "initial_states": list(indices)},
                "model_input_keys": list(MODEL_INPUT_KEYS),
                "images": "raw unrotated HWC RGB uint8; official model processor rotates once",
                "source": {"method": "physics reset + snapshot -> separate sync scene observe_snapshot",
                           "model_loaded": False, "control_or_camera_workers_started": False,
                           "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in source_paths}},
                "provenance": provenance, "capture_wall_seconds": time.perf_counter() - started,
                "simulators_closed_before_manifest": True}
    temporary = output / ".manifest.json.tmp"
    with temporary.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(output / "manifest.json")
    return manifest


def main(argv=None):
    """Run the standalone capture CLI without importing any model runtime.

    Args: argv: optional argument sequence, otherwise process arguments.
    Returns: None; prints compact completion metadata.
    Raises: Capture/argument errors; existing output always fails before simulator startup.
    Design: Context: Root schedules this only after its shared GPU lease is released.
        Concern: Capture renderer must exit before inference profiling starts.
        Decision: One foreground executable stage with explicit fresh output directory.
        Alternatives: Lazy capture inside each model profiler duplicates and contaminates input.
        Quality Attributes: Isolation. Trade-offs: GPU availability is coordinated externally.
    Evidence: Internal: Capture lifecycle tests. External: None.
        Findings: --help initializes no simulator. Metrics: Sample count and elapsed capture time.
        Confidence: CPU CLI smoke.
    Constraints: No assets download, model loading or profiling in this command.
    Reconsider When: An orchestration service owns the GPU lease.
    References: capture_profile_inputs. ADR: None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-states", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args(argv)
    manifest = capture_profile_inputs(args.output, args.initial_states, args.seed, args.resolution)
    print(json.dumps({"status": manifest["status"], "output": str(args.output),
                      "samples": len(manifest["samples"]),
                      "capture_wall_seconds": manifest["capture_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
