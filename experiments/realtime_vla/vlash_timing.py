"""Keep VLASH state-time alignment and a unit-consistent robot-only forecast explicit.

Args: Measured robot sensors, known native action prefix, control ticks and chunks.
Returns: Forecast robot sensors or bounded state-origin action scheduling.
Raises: Invalid sensor/action/tick/queue errors.
Design: Context: VLASH training aligns future state and action targets to one offset.
    Concern: Treating every chunk as image-origin or adding action7 to state8 is incorrect.
    Decision: Separate image capture, measured proprioception and represented state origin;
        forecast free Cartesian critical damping with native scales and world-left rotations.
    Alternatives: Future oracle state or dimensionally invalid action accumulation.
    Quality Attributes: Causal traceability. Trade-offs: Contact and full manipulator dynamics are omitted.
Evidence: Internal: robosuite OSC kp150/kd2sqrt150/scales .05m and .5rad.
    External: Author VLASHDataset shifts state and targets together.
    Findings: Predicted state can represent a future tick; early chunks must wait without blocking control.
    Metrics: Origin delay, pending waits, dropped prefix and prediction horizon.
    Confidence: Mechanics tested; forecast quality requires measured diagnostics.
Constraints: No object truth, future measurements, learned predictor or simulator clone lookahead.
Reconsider When: Controller gains/action scales or orientation convention change.
References: docs/reference/vlash-libero-assets.md. ADR: None.
"""
import math
import numpy as np
try:
    from .chunk_queue import ChunkQueue
except ImportError:
    from chunk_queue import ChunkQueue


def quaternion_product(left,right):
    """Compose XYZW quaternions using world-left rotation order.

    Args: left/right: finite quaternion4. Returns: Quaternion4. Raises: NumPy shape errors.
    Design: Context: Native OSC uses Rdelta@Rcurrent. Concern: Euler addition changes rotation order.
        Decision: Hamilton product in XYZW convention. Alternatives: Axis-angle vector addition.
        Quality Attributes: Geometric fidelity. Trade-offs: Inputs are validated by caller.
    Evidence: Internal: robosuite set_goal_orientation. External: Quaternion algebra.
        Findings: Left product applies world rotation. Metrics: Unit checks. Confidence: Exact algebra.
    Constraints: XYZW only. Reconsider When: Robot frame convention changes.
    References: forecast_robot_sensors. ADR: None.
    """
    left=np.asarray(left,dtype=np.float64);right=np.asarray(right,dtype=np.float64)
    return np.r_[left[3]*right[:3]+right[3]*left[:3]+np.cross(left[:3],right[:3]),
                 left[3]*right[3]-np.dot(left[:3],right[:3])]


def quaternion_log(quaternion):
    """Return the shortest world rotation vector of a finite XYZW quaternion.

    Args: quaternion: quaternion4. Returns: Rotation vector3 radians. Raises: Invalid norm errors.
    Design: Context: Measured angular velocity. Concern: q and -q must be equivalent.
        Decision: Normalize and select nonnegative scalar hemisphere before logarithm.
        Alternatives: Differentiate raw quaternion entries. Quality Attributes: Stability. Trade-offs: Pi branch ambiguity.
    Evidence: Internal: SO3 forecast tests. External: Quaternion exponential/log identities.
        Findings: Identity returns zero. Metrics: Round-trip checks. Confidence: Analytic.
    Constraints: Finite nonzero quaternion. Reconsider When: Rotations cross pi between ticks.
    References: quaternion_exp. ADR: None.
    """
    q=np.asarray(quaternion,dtype=np.float64);norm=float(np.linalg.norm(q))
    if q.shape!=(4,) or not np.isfinite(q).all() or norm<1e-12:raise ValueError('invalid XYZW quaternion')
    q=q/norm
    if q[3]<0:q=-q
    size=float(np.linalg.norm(q[:3]))
    return 2*q[:3] if size<1e-10 else q[:3]*(2*math.atan2(size,float(q[3]))/size)


def quaternion_exp(rotation):
    """Convert a rotation vector in radians to a normalized XYZW quaternion.

    Args: rotation: finite vector3. Returns: Quaternion4. Raises: ValueError on invalid vector.
    Design: Context: Geometric forecast. Concern: Rotations must stay on SO3.
        Decision: Exponential map with small-angle branch. Alternatives: Add quaternion entries.
        Quality Attributes: Stability. Trade-offs: None for supported vectors.
    Evidence: Internal: Round-trip tests. External: Axis-angle quaternion identity.
        Findings: Zero rotation gives identity. Metrics: Norm. Confidence: Analytic.
    Constraints: Radians. Reconsider When: Inputs use another convention.
    References: quaternion_log. ADR: None.
    """
    rotation=np.asarray(rotation,dtype=np.float64)
    if rotation.shape!=(3,) or not np.isfinite(rotation).all():raise ValueError('invalid rotation vector')
    angle=float(np.linalg.norm(rotation))
    q=np.r_[rotation*(.5 if angle<1e-10 else math.sin(angle/2)/angle),math.cos(angle/2)]
    return q/np.linalg.norm(q)


def critical_step(delta,velocity,dt,kp=150.):
    """Integrate one free critically damped Cartesian target step in SI units.

    Args: delta: target-current in m/rad; velocity: m/s or rad/s; dt: seconds; kp: s^-2.
    Returns: Displacement and terminal velocity vectors. Raises: Invalid gains/time/vector errors.
    Design: Context: OSC uses mass-compensated kp150/kd2sqrt150 and piecewise targets.
        Concern: Raw delta action is neither a joint angle nor immediate physical displacement.
        Decision: Closed form e=(e0+(v0+w*e0)t)exp(-wt), e0=-delta.
        Alternatives: Treat normalized action as state increment.
        Quality Attributes: Unit consistency. Trade-offs: Free Cartesian approximation omits contact/saturation.
    Evidence: Internal: Installed robosuite OSC defaults and analytic tests.
        External: Critically damped second-order dynamics. Findings: Zero dt leaves state unchanged.
        Metrics: Displacement/velocity units. Confidence: Analytic approximation.
    Constraints: Positive kp/nonnegative dt. Reconsider When: Impedance/controller changes.
    References: forecast_robot_sensors. ADR: None.
    """
    delta=np.asarray(delta,dtype=np.float64);velocity=np.asarray(velocity,dtype=np.float64)
    if delta.shape!=(3,) or velocity.shape!=(3,) or not np.isfinite(np.r_[delta,velocity,dt,kp]).all() or dt<0 or kp<=0:
        raise ValueError('invalid critical-damping inputs')
    omega=math.sqrt(kp);initial_error=-delta;term=velocity+omega*initial_error;decay=math.exp(-omega*dt)
    error=(initial_error+term*dt)*decay;terminal_velocity=(velocity-omega*term*dt)*decay
    return delta+error,terminal_velocity


def forecast_robot_sensors(current,previous,prefix,horizon_ticks,control_hz=20.,method='pd'):
    """Forecast only measured EEF pose, holding measured finger positions conservatively.

    Args: current/previous: {tick,raw} measured proprioception; previous may be None;
        prefix: known queued native actions[T,7]; horizon_ticks: causal future steps;
        control_hz: physical time base; method: pd or constant-velocity.
    Returns: Raw sensor override and explicit predictor metadata.
    Raises: Invalid sensing/action/horizon errors.
    Design: Context: Predicted state represents expected inference completion, not an observed future.
        Concern: Pose and action dimensions/units differ; contact truth must not leak.
        Decision: Measured finite-difference velocity and native-scale free OSC integration;
            orientation uses world-left SO3 and native sensor w>=0 quaternion representation;
            fingers remain measured.
        Alternatives: Future simulator lookahead or action-sum state proxy.
        Quality Attributes: Causality, inspectability. Trade-offs: No contact, torque saturation or full Jacobian dynamics.
    Evidence: Internal: Native OSC scales .05m/.5rad and kp150,20Hz command interval.
        External: robosuite set_goal_orientation. Findings: Planned commands are clipped as at actuation.
        Metrics: Horizon, velocity and padded hold count. Confidence: Model approximation, not oracle.
    Constraints: No object sensors. Zero rotational command is approximated by a current-pose target;
        native persistent rotational goals can differ. No learned correction.
    Reconsider When: Measured prediction errors invalidate the approximation.
    References: critical_step. ADR: None.
    """
    if type(horizon_ticks) is not int or horizon_ticks<0 or not math.isfinite(control_hz) or control_hz<=0 or method not in ('pd','constant-velocity'):
        raise ValueError('invalid forecast horizon/method')
    raw=current['raw'];position=np.asarray(raw['robot0_eef_pos'],dtype=np.float64).copy()
    quaternion=np.asarray(raw['robot0_eef_quat'],dtype=np.float64).copy();quaternion_log(quaternion);quaternion/=np.linalg.norm(quaternion)
    fingers=np.asarray(raw['robot0_gripper_qpos'],dtype=np.float64).copy()
    if position.shape!=(3,) or fingers.shape!=(2,) or not np.isfinite(np.r_[position,fingers]).all():raise ValueError('invalid measured proprioception')
    velocity=np.zeros(3);angular=np.zeros(3);velocity_dt=None
    if previous is not None:
        ticks=int(current['tick'])-int(previous['tick'])
        if ticks<=0:raise ValueError('noncausal measured history')
        velocity_dt=ticks/control_hz;old=previous['raw']
        velocity=(position-np.asarray(old['robot0_eef_pos'],dtype=np.float64))/velocity_dt
        oldq=np.asarray(old['robot0_eef_quat'],dtype=np.float64);quaternion_log(oldq);oldq=oldq/np.linalg.norm(oldq)
        inverse=np.r_[-oldq[:3],oldq[3]];angular=quaternion_log(quaternion_product(quaternion,inverse))/velocity_dt
    measured_velocity=velocity.copy();measured_angular=angular.copy()
    actions=np.asarray(prefix,dtype=np.float64).reshape(-1,7)
    if not np.isfinite(actions).all():raise ValueError('nonfinite known actuator prefix')
    clipped=int(np.count_nonzero(np.abs(actions)>1));actions=np.clip(actions,-1,1);dt=1/control_hz
    if method=='constant-velocity':
        position+=velocity*horizon_ticks*dt;quaternion=quaternion_product(quaternion_exp(angular*horizon_ticks*dt),quaternion)
    else:
        for tick in range(horizon_ticks):
            command=actions[tick] if tick<len(actions) else np.zeros(7)
            displacement,velocity=critical_step(command[:3]*.05,velocity,dt)
            rotation_delta=command[3:6]*.5
            rotational_displacement,angular=critical_step(rotation_delta,angular,dt)
            target=quaternion_product(quaternion_exp(rotation_delta),quaternion)
            quaternion=quaternion_product(quaternion_exp(rotational_displacement-rotation_delta),target)
            position+=displacement
            quaternion/=np.linalg.norm(quaternion)
    # Match robosuite mat2quat's scalar-nonnegative sensor convention. Equivalent
    # q/-q values otherwise produce different canonical axis-angle state tokens.
    if quaternion[3]<0:quaternion=-quaternion
    overrides={'robot0_eef_pos':position.astype(np.float32),'robot0_eef_quat':quaternion.astype(np.float32),
               'robot0_gripper_qpos':fingers.astype(np.float32)}
    metadata={'method':method,'measured_state_tick':int(current['tick']),'target_state_tick':int(current['tick'])+horizon_ticks,
              'horizon_ticks':horizon_ticks,'horizon_seconds':horizon_ticks/control_hz,'control_hz':control_hz,
              'linear_velocity_m_s':measured_velocity.tolist(),'world_angular_velocity_rad_s':measured_angular.tolist(),
              'velocity_dt_sim_seconds':velocity_dt,'known_prefix_length':len(actions),
              'zero_delta_padding_ticks':max(0,horizon_ticks-len(actions)),'clipped_prefix_components':clipped,
              'position_scale_m':.05,'rotation_scale_rad':.5,'kp_s_minus2':150.,'kd_s_minus1':2*math.sqrt(150.),
              'output_quaternion_convention':'XYZW, scalar w>=0 as robosuite mat2quat',
              'gripper_prediction':'hold measured two finger positions','state_is_prediction':True,
              'limitations':['free Cartesian approximation','no contact/torque saturation/full manipulator dynamics',
                             'zero rotational action approximated as current-pose target']}
    return overrides,metadata


class StateOriginQueue:
    """Align chunks to their represented state tick and defer early predictions nonblockingly.

    Args: horizon: native generated horizon. Returns: Bounded scheduler instance.
    Raises: Chunk/tick or pending-capacity errors.
    Design: Context: Coherent/current/predicted VLASH inputs represent different times.
        Concern: Image-origin expiration is wrong for current/future state conditioning.
        Decision: Reuse common aligned queue at explicit state origin and retain at most one early chunk.
        Alternatives: Run a future-state action before its target tick or freeze physical control.
        Quality Attributes: Causal alignment. Trade-offs: New requests wait for pending activation.
    Evidence: Internal: ChunkQueue tested aligned/hold behavior. External: VLASH offset semantics.
        Findings: Waiting for target means continuing the old actuator queue, not blocking.
        Metrics: Pending waits and discarded prefix. Confidence: Unit-tested mechanics.
    Constraints: One model query and one pending chunk maximum. Reconsider When: Multiquery serving is used.
    References: accept; advance. ADR: None.
    """
    def __init__(self,horizon=50):
        """Create the reused aligned queue with one explicit pending slot.

        Args: horizon: positive integer. Returns: None. Raises: Invalid horizon errors.
        Design: Context: Native H50. Concern: Unbounded early results. Decision: One pending slot.
            Alternatives: Multiple future chunks. Quality Attributes: Bounded memory. Trade-offs: Serial generation.
        Evidence: Internal: Unit tests. External: None. Findings: Existing hold semantics preserved.
            Metrics: Zero pending initially. Confidence: Deterministic.
        Constraints: Caller owns inference concurrency. Reconsider When: Serving changes.
        References: StateOriginQueue. ADR: None.
        """
        self.buffer=ChunkQueue(horizon,'aligned');self.pending=None;self.pending_wait_ticks=0

    def __len__(self):
        """Return executable old/current actions, excluding a future pending chunk.

        Args: None. Returns: Queue length. Raises: None.
        Design: Context: R30 triggers. Concern: Pending is not executable. Decision: Expose only current buffer.
            Alternatives: Count future actions as ready. Quality Attributes: Clarity. Trade-offs: None.
        Evidence: Internal: Pending tests. External: None. Findings: No activation side effect.
            Metrics: Length. Confidence: Deterministic.
        Constraints: Read-only. Reconsider When: Queue API changes. References: advance. ADR: None.
        """
        return len(self.buffer)

    def accept(self,message,ready_tick):
        """Accept a result now or store it until its represented state time arrives.

        Args: message: actions/request/source ticks and time; ready_tick: control now.
        Returns: Applied/deferred event. Raises: Invalid chunk, causal time or occupied-slot errors.
        Design: Context: Latency predictor may overestimate completion. Concern: Premature future-state actions.
            Decision: Defer without touching old queue; activation drops only actual state-origin lateness.
            Alternatives: Relabel origin as now. Quality Attributes: Causality. Trade-offs: Pending slot.
        Evidence: Internal: Early/late queue tests. External: VLASH state offset semantics.
            Findings: All cases preserve image capture metadata. Metrics: Prefix expiration.
            Confidence: Deterministic mechanics.
        Constraints: At most one pending. Reconsider When: Asynchronous policy streaming changes.
        References: advance. ADR: None.
        """
        if self.pending is not None:raise RuntimeError('VLASH pending slot already occupied')
        actions=np.asarray(message['actions'])
        if actions.shape!=(50,7) or not np.isfinite(actions).all():raise ValueError('expected finite native H50x7')
        origin=int(message['state_origin_tick']);image=int(message['observation_tick'])
        if image<0 or image>ready_tick or origin<0:raise ValueError('invalid causal source tick')
        if origin>ready_tick:
            self.pending=dict(message);return {'request_id':message['request_id'],'deferred':True,'ready_tick':ready_tick,
                                              'state_origin_tick':origin,'activation_tick':origin,'queued_actions':0,'expired_prefix_actions':0}
        count=self.buffer.accept(actions,origin,message['observation_time'],ready_tick,message['request_id'])
        for item in self.buffer.queue:
            item.update(image_capture_tick=image,state_origin_tick=origin,measured_proprio_tick=message['measured_proprio_tick'],
                        measured_proprio_time=message.get('measured_proprio_time',message['observation_time']),
                        state_condition=message['state_condition'],state_is_prediction=message['state_condition']=='predicted')
        return {'request_id':message['request_id'],'deferred':False,'ready_tick':ready_tick,'activation_tick':ready_tick,
                'state_origin_tick':origin,'queued_actions':count,'expired_prefix_actions':min(50,max(0,ready_tick-origin))}

    def advance(self,tick):
        """Activate an early chunk at its origin while old control continues before then.

        Args: tick: current control tick. Returns: Activation event or None. Raises: Underlying schema errors.
        Design: Context: Nonblocking early arrival. Concern: Negative-prefix expiration.
            Decision: Explicit tick gate. Alternatives: sleep until future state time.
            Quality Attributes: Timeliness. Trade-offs: One-tick activation granularity.
        Evidence: Internal: Early activation test. External: None. Findings: No waits or model calls.
            Metrics: Pending wait ticks. Confidence: Deterministic.
        Constraints: Called once per tick. Reconsider When: Subtick control is introduced.
        References: accept. ADR: None.
        """
        if self.pending is None:return None
        if tick<int(self.pending['state_origin_tick']):self.pending_wait_ticks+=1;return None
        message=self.pending;self.pending=None;return self.accept(message,tick)

    def pop(self):
        """Consume one native action or the common zero-delta/gripper-preserving hold.

        Args: None. Returns: Action metadata. Raises: None.
        Design: Context: Physical control always advances. Concern: Empty queue repeats relative motion.
            Decision: Reuse tested hold implementation. Alternatives: Repeat last delta.
            Quality Attributes: Predictability. Trade-offs: Holds may reduce progress.
        Evidence: Internal: ChunkQueue tests. External: None. Findings: Gripper state retained.
            Metrics: Hold counter. Confidence: Deterministic.
        Constraints: Exactly one consumption per physical tick. Reconsider When: Absolute control is used.
        References: ChunkQueue.pop. ADR: None.
        """
        return self.buffer.pop()
