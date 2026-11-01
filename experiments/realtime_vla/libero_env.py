"""Expose canonical LIBERO sensors and physics without inference waits or automatic reset.

Args: Task suite/id, fixed benchmark initial state, RGB resolution and output directory.
Returns: Raw two-camera observations, canonical eight-value state and evaluator metrics.
Raises: ValueError for invalid task/action/state; runtime errors for missing dependencies/assets.
Design:
    Context: Real-time scheduling belongs to the parent runner, not the environment.
    Concern: Previous lockstep wrappers froze physics while a policy was computing.
    Decision: Reuse LIBERO OffScreenRenderEnv, canonical reset states and ten settling steps.
    Alternatives: A custom scene would invalidate pretrained policy compatibility.
    Quality Attributes: Reproducibility, compatibility. Trade-offs: Known benchmark only.
Evidence:
    Internal: Matched SmolVLA and pi0-FAST LIBERO input contracts.
    External: LeRobot envs/libero.py and LiberoProcessorStep.
    Findings: Input cameras are raw; official policy processors rotate them once.
    Metrics: Goal XY error, OSC tracking error and physical step latency.
    Confidence: Functional tests and recorded bare-simulator smoke.
Constraints: No sleep, model calls, object teleportation during step or terminal auto-reset.
Reconsider When: Task assets or policy observation contract changes.
References: LeRobot fixed source commit3f797de5952432284a7c77ad2760218cb74e2f26.
ADR: None.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ASSET_REVISION='0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'


def configure_rendering():
    """Select the existing WSL GPU driver before OpenGL modules are imported.

    Args: None. Returns: None. Raises: None; renderer initialization reports failures.
    Design:
        Context: RTX5080 CUDA availability does not imply accelerated OpenGL.
        Concern: Mesa otherwise selected llvmpipe despite an available D3D12 device.
        Decision: On WSL only, default to its installed D3D12 driver and NVIDIA adapter.
        Alternatives: Changing system drivers would affect unrelated applications.
        Quality Attributes: Performance, isolation. Trade-offs: Process-local defaults.
    Evidence:
        Internal: environment-smoke/profile.pstat assigns88percent of software time to rendering.
        External: Mesa D3D12 GALLIUM_DRIVER and adapter-selection environment variables.
        Findings: Explicit D3D12 produces GL_RENDERER NVIDIA GeForce RTX5080.
        Metrics: Actual renderer is recorded in environment metadata.
        Confidence: Verified EGL context and two-camera environment initialization.
    Constraints: Preserve explicit user overrides and non-WSL configurations.
    Reconsider When: Native Linux NVIDIA EGL is used.
    References: artifacts/realtime-vla/environment-smoke/hardware-render-results.json.
    ADR: None.
    """
    if Path('/dev/dxg').exists() and Path('/usr/lib/wsl/lib/libd3d12.so').exists():
        os.environ.setdefault('GALLIUM_DRIVER','d3d12')
        os.environ.setdefault('MESA_D3D12_DEFAULT_ADAPTER_NAME','NVIDIA')
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')


def canonical_state(raw):
    """Convert raw robot sensing to LeRobot's position/axis-angle/gripper eight-vector.

    Args: raw: eef position, XYZW quaternion and two finger positions.
    Returns: Float32 state[8]. Raises: ValueError for invalid or nonfinite input.
    Design:
        Context: Both pretrained policies share LIBERO state semantics.
        Concern: Quaternion and axis-angle are not interchangeable.
        Decision: Match official LiberoProcessorStep's clipped quaternion conversion.
        Alternatives: Passing quaternion creates a nine-value incompatible state.
        Quality Attributes: Compatibility. Trade-offs: State is an optional convenience alias.
    Evidence:
        Internal: Models still run their own canonical processors from raw sensing.
        External: LeRobot LiberoProcessorStep._quat2axisangle.
        Findings: Quaternions are XYZW and camera rotation is separate.
        Metrics: State shape/finite assertions. Confidence: Unit-tested conversion.
    Constraints: Does not expose object state.
    Reconsider When: Model state schema changes.
    References: Module contract.
    ADR: None.
    """
    pos=np.asarray(raw['robot0_eef_pos'],dtype=np.float32)
    quat=np.asarray(raw['robot0_eef_quat'],dtype=np.float32)
    grip=np.asarray(raw['robot0_gripper_qpos'],dtype=np.float32)
    if pos.shape!=(3,) or quat.shape!=(4,) or grip.shape!=(2,) or not np.isfinite(np.r_[pos,quat,grip]).all():
        raise ValueError('invalid proprioceptive sensing')
    w=np.clip(quat[3],-1.,1.);den=np.sqrt(max(0.,1.-float(w*w)))
    axis=np.zeros(3,dtype=np.float32) if den<=1e-10 else quat[:3]/den*(2*np.arccos(w))
    return np.concatenate([pos,axis,grip]).astype(np.float32)


def validate_action(action):
    """Require finite native seven-value relative-pose/gripper control.

    Args: action: array-like seven values. Returns: Float32 bounded command.
    Raises: ValueError for invalid shape or nonfinite commands.
    Design:
        Context: Common simulator control must be fair across both policies.
        Concern: Malformed chunks must not advance physical state.
        Decision: Validate finite7D before env.step and apply the existing controller bounds.
        Alternatives: Rejecting ordinary VLA flow tails would differ from canonical controller behavior.
        Quality Attributes: Integrity. Trade-offs: Invalid inference aborts the episode.
    Evidence:
        Internal: LIBERO action dimension7 and bounds[-1,1].
        External: LeRobot LiberoEnv.action_space.
        Findings: Relative pose commands must not be reused as position-hold targets.
        Metrics: Shape/range checks. Confidence: Unit-tested.
    Constraints: No action rescaling or gripper sign changes here.
    Reconsider When: Controller representation changes.
    References: Module contract.
    ADR: None.
    """
    array=np.asarray(action,dtype=np.float32)
    if array.shape!=(7,) or not np.isfinite(array).all():
        raise ValueError('expected finite native action[7]')
    return np.clip(array,-1,1)


def configure_libero_paths():
    """Keep LIBERO configuration and fixed community assets inside the workspace.

    Args: None. Returns: Package directory. Raises: FileNotFoundError if assets absent.
    Design:
        Context: LIBERO otherwise prompts and writes to a home-directory configuration.
        Concern: Setup must be noninteractive and preserve external user files.
        Decision: Workspace config plus an environment-local assets symlink.
        Alternatives: Monkeypatching object XML paths would alter upstream behavior.
        Quality Attributes: Isolation. Trade-offs: Depends on dedicated virtual environment.
    Evidence:
        Internal: Fixed assets revision saved in manifest.
        External: hf-libero get_libero_path/get_assets_path.
        Findings: Local assets are preferred over automatic Hub download.
        Metrics: Path existence. Confidence: Filesystem verification.
    Constraints: No base virtual environment changes.
    Reconsider When: Package layout changes.
    References: scripts/setup_realtime_vla.sh.
    ADR: None.
    """
    import yaml
    package=Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/libero/libero'
    assets=ROOT/'.cache/libero-assets'
    if not (assets/'stable_scanned_objects').is_dir():raise FileNotFoundError('Run setup_realtime_vla.sh to obtain fixed LIBERO assets')
    target=package/'assets'
    if not target.exists():
        try:target.symlink_to(assets,target_is_directory=True)
        except FileExistsError:pass
    if target.resolve()!=assets.resolve():raise RuntimeError('dedicated LIBERO package assets do not match the pinned workspace snapshot')
    config=ROOT/'.cache/libero-realtime-config';config.mkdir(parents=True,exist_ok=True)
    os.environ['LIBERO_CONFIG_PATH']=str(config)
    paths={'benchmark_root':str(package),'bddl_files':str(package/'bddl_files'),'init_states':str(package/'init_files'),'datasets':str(ROOT/'.cache/libero-datasets'),'assets':str(assets)}
    temporary=config/f'config-{os.getpid()}.tmp'
    temporary.write_text(yaml.safe_dump(paths));temporary.replace(config/'config.yaml')
    return package


class FrozenLiberoEnv:
    """Own a single benchmark episode and expose fast wall-clock-schedulable steps.

    Args: task_suite, task_id, horizon, resolution, output_dir.
    Returns: Environment instance. Raises: ValueError for invalid suite/task.
    Design:
        Context: Root scheduler controls timing and asynchronous policy queues.
        Concern: Auto-reset or policy calls inside physics would hide latency.
        Decision: Direct official OffScreenRenderEnv with fixed benchmark state snapshots.
        Alternatives: LeRobot Gym wrapper auto-resets terminal observations.
        Quality Attributes: Traceability. Trade-offs: Single task/scene at a time.
    Evidence:
        Internal: Official task assets and success predicate are unchanged.
        External: LeRobot LiberoEnv reset convention.
        Findings: Ten dummy actions settle initial contacts before timed evaluation.
        Metrics: Success, goal error and step wall time.
        Confidence: Measured by bare-environment smoke.
    Constraints: Raw unflipped images; no policy-visible object truth.
    Reconsider When: Policy checkpoint requires another camera layout.
    References: Module contract.
    ADR: None.
    """
    def __init__(self,task_suite='libero_spatial',task_id=0,horizon=280,resolution=256,output_dir=None,control_freq=20,render_mode='sync'):
        """Load the selected official task and its fifty published initial states.

        Args: See class; render_mode: sync cameras or physics without rendering.
        Returns: None. Raises: Missing asset/runtime errors or invalid rendering mode.
        Design:
            Context: Model comparisons require identical initial physical states.
            Concern: Independent random reset calls can differ across implementations.
            Decision: Restore exact published benchmark states by index; optionally omit sensors for a separate renderer.
            Alternatives: Uncontrolled randomization confounds model comparisons.
            Quality Attributes: Reproducibility. Trade-offs: Benchmark distribution only.
        Evidence:
            Internal: State file SHA recorded in metadata.
            External: LIBERO benchmark pruned_init files.
            Findings: Explicit trusted torch load supports numpy state metadata.
            Metrics: Initial state SHA and task identity. Confidence: Verified by tests.
        Constraints: Canonical20Hz or explicitly reported10Hz controller; no sleeping inside environment.
        Reconsider When: Task distribution changes.
        References: Module contract.
        ADR: None.
        """
        configure_rendering()
        package=configure_libero_paths()
        import torch
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.env_wrapper import ControlEnv
        suites=benchmark.get_benchmark_dict()
        if task_suite not in suites:raise ValueError('unknown LIBERO task suite')
        self.suite=suites[task_suite]();self.task=self.suite.get_task(int(task_id))
        self.task_suite=task_suite;self.task_id=int(task_id);self.horizon=int(horizon);self.control_freq=int(control_freq)
        if render_mode not in ('sync','physics'):raise ValueError('render_mode must be sync or physics')
        self.render_mode=render_mode
        if self.control_freq not in (10,20):raise ValueError('validated control frequencies are10 or20Hz')
        if self.horizon<1:raise ValueError('horizon must be positive')
        state_file=package/'init_files'/self.task.problem_folder/self.task.init_states_file
        self.initial_states=torch.load(state_file,map_location='cpu',weights_only=False)
        env_type=OffScreenRenderEnv if render_mode=='sync' else ControlEnv
        self.env=env_type(bddl_file_name=str(package/'bddl_files'/self.task.problem_folder/self.task.bddl_file),camera_heights=resolution,camera_widths=resolution,control_freq=self.control_freq,use_camera_obs=render_mode=='sync',has_offscreen_renderer=render_mode=='sync')
        renderer=None
        if render_mode=='sync':
            from OpenGL import GL
            renderer=GL.glGetString(GL.GL_RENDERER)
        self.step_count=0;self.done=True;self.last_raw=None;self.last_action=np.array([0,0,0,0,0,0,-1],dtype=np.float32)
        self.output_dir=Path(output_dir) if output_dir else None
        self.metadata={'task_suite':task_suite,'task_id':self.task_id,'task_name':self.task.name,'prompt':self.task.language,'initial_states':len(self.initial_states),'initial_states_file_sha256':hashlib.sha256(state_file.read_bytes()).hexdigest(),'assets_revision':ASSET_REVISION,'control_hz':self.control_freq,'canonical_control_hz':20,'dataset_metadata_fps':10,'physics_dt':float(self.env.sim.model.opt.timestep),'settle_steps':10,'images':'raw HWC RGB; policy official processor rotates180 once','action':'native LIBERO relative XYZ/axisangle/gripper7'}
        self.metadata.update(render_mode=render_mode,gl_renderer=renderer.decode() if renderer else None,render_platform=os.environ.get('MUJOCO_GL'),gallium_driver=os.environ.get('GALLIUM_DRIVER'))
        if self.output_dir:self.output_dir.mkdir(parents=True,exist_ok=True);(self.output_dir/'environment-metadata.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

    def _observation(self,raw,latency=0.):
        """Return model-compatible raw sensing and segregated evaluator measurements.

        Args: raw: official LIBERO sensing; latency: latest step wall seconds.
        Returns: Sensor dict with images aliases/state/prompt/success/metrics.
        Raises: ValueError for malformed camera/state.
        Design:
            Context: Existing policy adapters expect official raw keys.
            Concern: Evaluator object truth must not masquerade as model inputs.
            Decision: Allowlist raw robot sensing; nest true object measurements in metrics.
            Alternatives: Forwarding raw object-state encourages accidental leakage.
            Quality Attributes: Integrity. Trade-offs: Policy adapter must honor allowlist.
        Evidence:
            Internal: Both model adapters prepare input from raw robot sensors.
            External: Official LIBERO image/state conventions.
            Findings: Success and goal error come from the simulator evaluator.
            Metrics: Goal XY error, OSC tracking error, wall step time.
            Confidence: Physical sensor/evaluator measurements.
        Constraints: No image rotation, resizing or normalization here.
        Reconsider When: Input schema changes.
        References: canonical_state.
        ADR: None.
        """
        keys=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','robot0_joint_pos','robot0_joint_vel')
        obs={key:np.asarray(raw[key]).copy() for key in keys if key in raw}
        for camera in ('agentview_image','robot0_eye_in_hand_image'):
            if self.render_mode=='sync' and (camera not in obs or obs[camera].ndim!=3 or obs[camera].shape[2]!=3 or obs[camera].dtype!=np.uint8):raise ValueError('invalid RGB camera')
        success=bool(self.env.check_success());metrics={'step_wall_seconds':latency,'native_clip_max_excess':getattr(self,'last_clip_excess',0.)}
        raw_env=self.env.env
        body_ids=getattr(raw_env,'obj_body_id',{})
        bowl=next((name for name in body_ids if 'black_bowl' in name),None)
        plate=next((name for name in body_ids if name.startswith('plate')),None)
        if bowl and plate:
            b=np.asarray(self.env.sim.data.body_xpos[body_ids[bowl]]);p=np.asarray(self.env.sim.data.body_xpos[body_ids[plate]])
            metrics.update(goal_xy_error_m=float(np.linalg.norm(b[:2]-p[:2])),bowl_position=b.tolist(),plate_position=p.tolist())
        controller=raw_env.robots[0].controller
        goal=getattr(controller,'goal_pos',None)
        if goal is not None:metrics['eef_tracking_error_m']=float(np.linalg.norm(np.asarray(goal)-obs['robot0_eef_pos']))
        metrics['actuator_control']=np.asarray(self.env.sim.data.ctrl).tolist()
        images={name:obs[key] for name,key in (('agentview','agentview_image'),('wrist','robot0_eye_in_hand_image')) if key in obs}
        obs.update(images=images,state=canonical_state(obs),prompt=self.task.language,step=self.step_count,success=success,done=bool(success or self.step_count>=self.horizon),metrics=metrics)
        return obs

    def snapshot(self,tick=None,episode_id=None):
        """Copy physical state for an independent renderer without waiting for a camera.

        Args: tick: control-boundary index; episode_id: reset identity carried to sensors.
        Returns: physics_state, capture_time, tick, episode_id and simulation_time mapping.
        Raises: ValueError for invalid tick.
        Design:
            Context: Sensor rendering must not stop the physics loop.
            Concern: Image receipt time hides the age of the physical state depicted.
            Decision: Timestamp the state copy at capture and carry it through rendering.
            Alternatives: Replacing capture time with render completion understates latency.
            Quality Attributes: Timing integrity. Trade-offs: One small state copy per request.
        Evidence:
            Internal: CameraWorker restores this exact state in the same official scene.
            External: LIBERO get_sim_state and regenerate_obs_from_state.
            Findings: Robot and object poses are synchronized by one MuJoCo state.
            Metrics: Capture tick/time versus rendering completion.
            Confidence: State-restoration integration checks.
        Constraints: The snapshot is evaluator transport, never a model observation.
        Reconsider When: Task includes external state outside MuJoCo.
        References: camera_worker.py.
        ADR: None.
        """
        tick=self.step_count if tick is None else tick
        if type(tick) is not int or tick<0:raise ValueError('invalid snapshot tick')
        capture_time=time.monotonic()
        state=np.asarray(self.env.get_sim_state()).copy()
        return {'physics_state':state,'capture_time':capture_time,'tick':tick,'episode_id':episode_id,'simulation_time':float(self.env.sim.data.time)}

    def observe_snapshot(self,snapshot):
        """Render raw RGB and proprioception from one synchronized physical snapshot.

        Args: snapshot: output of snapshot(). Returns: Raw sensor mapping with capture time/tick.
        Raises: ValueError when called on the physics-only environment or invalid state.
        Design:
            Context: A separate process owns the canonical two-camera scene.
            Concern: Combining current proprioception with delayed images creates a false observation.
            Decision: Restore state and force every observable from that same snapshot.
            Alternatives: Joining latest image with current state breaks sensor synchronization.
            Quality Attributes: Compatibility, traceability. Trade-offs: Renderer has a scene clone.
        Evidence:
            Internal: Initial state and scene assets match the physics environment.
            External: Official LIBERO regenerate_obs_from_state implementation.
            Findings: No action or simulator step is executed by rendering.
            Metrics: Render duration, capture age, raw sensor hashes.
            Confidence: Integration smoke verifies state remains unchanged.
        Constraints: Does not expose the flattened snapshot to the policy.
        Reconsider When: A dynamic non-MuJoCo sensor is introduced.
        References: CameraWorker.
        ADR: None.
        """
        if self.render_mode!='sync':raise ValueError('snapshot observation requires a sensor scene')
        state=np.asarray(snapshot['physics_state'])
        if state.ndim!=1 or not np.isfinite(state).all():raise ValueError('invalid physics snapshot')
        start=time.monotonic()
        raw=self.env.regenerate_obs_from_state(state)
        self.step_count=int(snapshot['tick'])
        obs=self._observation(raw)
        obs.update(capture_time=float(snapshot['capture_time']),tick=int(snapshot['tick']),episode_id=snapshot.get('episode_id'),simulation_time=float(snapshot['simulation_time']),render_started_time=start,render_finished_time=time.monotonic())
        return obs

    def reset(self,init_state_index=0,seed=0):
        """Restore a fixed benchmark state and settle before timed execution.

        Args: init_state_index: published state index; seed: environment RNG seed.
        Returns: Initial sensing. Raises: ValueError for invalid state index.
        Design:
            Context: Paired models must start from identical physical configurations.
            Concern: Reset settling must not count as inference-driven motion.
            Decision: Same ten canonical no-op steps for every condition.
            Alternatives: Random placement adds an avoidable confound.
            Quality Attributes: Reproducibility. Trade-offs: Timer begins after settling.
        Evidence:
            Internal: Exact state and image hashes can be recorded by root runner.
            External: LeRobot LiberoEnv.reset.
            Findings: No model inference is executed here.
            Metrics: Reset state identity. Confidence: Measured.
        Constraints: Nonnegative valid state index; no implicit modulo.
        Reconsider When: Benchmark state distribution changes.
        References: Module contract.
        ADR: None.
        """
        if type(init_state_index) is not int or not 0<=init_state_index<len(self.initial_states):raise ValueError('invalid initial state index')
        self.env.seed(int(seed));self.env.reset();raw=self.env.set_init_state(self.initial_states[init_state_index])
        for _ in range(10):raw,_,_,_=self.env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32))
        self.step_count=0;self.done=False;self.last_clip_excess=0.;self.last_raw=raw;self.last_action=np.array([0,0,0,0,0,0,-1],dtype=np.float32)
        return self._observation(raw)

    def step(self,action):
        """Apply one finite native controller action and return fresh sensors immediately.

        Args: action: native seven-vector. Returns: New sensing/evaluation mapping.
        Raises: RuntimeError after terminal; ValueError before malformed actuation.
        Design:
            Context: Wall-clock runner calls this independently of ongoing inference.
            Concern: Automatic reset or blocking inference hides deadline misses.
            Decision: Exactly one official physics step; terminal remains observable.
            Alternatives: LeRobot vector wrapper resets on success.
            Quality Attributes: Timing transparency. Trade-offs: Caller handles episode lifecycle.
        Evidence:
            Internal: Step count and latency are recorded after env.step.
            External: LIBERO OffScreenRenderEnv.step.
            Findings: Each call advances one configured native controller interval.
            Metrics: Wall step latency, success and goal error.
            Confidence: Physical execution.
        Constraints: No sleeps or model invocations.
        Reconsider When: Manual camera sampling is required for throughput.
        References: validate_action.
        ADR: None.
        """
        if self.done:raise RuntimeError('reset required before step')
        command=validate_action(action);self.last_clip_excess=float(max(0.,np.abs(np.asarray(action)).max()-1.));start=time.perf_counter();raw,_,native_done,_=self.env.step(command)
        self.step_count+=1;self.last_raw=raw;self.last_action=command.copy()
        obs=self._observation(raw,time.perf_counter()-start);self.done=bool(native_done or obs['done']);obs['done']=self.done
        return obs

    def hold_action(self):
        """Return zero relative motion while preserving the last gripper command.

        Args: None. Returns: Seven-vector. Raises: None.
        Design:
            Context: Empty action queues must not repeat a movement delta.
            Concern: Holding the previous relative action causes drift.
            Decision: Zero XYZ/axisangle and retained gripper state.
            Alternatives: Repeating the entire last action is not position hold.
            Quality Attributes: Predictable fallback. Trade-offs: OSC still compensates dynamics.
        Evidence:
            Internal: Native LIBERO controls are relative pose increments.
            External: OSC_POSE use_delta contract.
            Findings: Repeating zeros refreshes the current pose target.
            Metrics: Queue starvation is recorded separately by scheduler.
            Confidence: Controller semantics, not hardware safety proof.
        Constraints: Simulation only; not a certified stop behavior.
        Reconsider When: Absolute-position control is selected.
        References: Module contract.
        ADR: None.
        """
        action=np.zeros(7,dtype=np.float32);action[-1]=self.last_action[-1];return action

    def close(self):
        """Release the owned simulator graphics/physics resources.

        Args: None. Returns: None. Raises: Upstream close errors.
        Design:
            Context: Models are evaluated serially on one16GBGPU.
            Concern: Graphics resources must not accumulate across runs.
            Decision: Delegate close to official environment.
            Alternatives: Relying on process exit complicates repeated tests.
            Quality Attributes: Resource hygiene. Trade-offs: None.
        Evidence:
            Internal: Owned OffScreenRenderEnv instance.
            External: LIBERO close API.
            Findings: Caller retains evaluation artifacts.
            Metrics: None. Confidence: API contract.
        Constraints: Do not reuse a closed instance.
        Reconsider When: Multiple scenes are pooled.
        References: Module contract.
        ADR: None.
        """
        self.env.close()
