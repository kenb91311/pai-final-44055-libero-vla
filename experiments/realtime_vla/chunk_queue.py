"""Common action-chunk timing rules for two frozen VLA architectures.

Args: Generated actions, observation identity and current control tick.
Returns: Actions with source metadata, or a delta-pose hold when empty.
Raises: ValueError for malformed/nonfinite chunks and invalid queue parameters.
Design:
    Context: Chunk inference runs asynchronously while physical control continues.
    Concern: Relative commands must not be repeated on queue exhaustion.
    Decision: Bounded common FIFO or time-aligned replacement and explicit zero-delta hold.
    Alternatives: Repeating the last delta action causes additional motion.
    Quality Attributes: Fairness, traceability. Trade-offs: Generic merging is not RTC guidance.
Evidence:
    Internal: Unit tests define expiration, overlap and hold behavior.
    External: LeRobot remote-inference distinguishes append and aligned strategies.
    Findings: Model size alone does not determine closed-loop performance.
    Metrics: Holds, discarded actions, observation age. Confidence: Contract-level only.
Constraints: Seven relative pose/gripper actions, consumed prefix length shared across models.
Reconsider When: Absolute joint controllers or model-specific RTC are used.
References: docs/architecture/realtime-vla.md.
ADR: None.
"""
from collections import deque
import numpy as np


class ChunkQueue:
    """Manage a common bounded executable prefix independently of model inference.

    Args: execute_steps: positive prefix count; mode: fifo or aligned.
    Returns: Queue instance. Raises: ValueError on unsupported configuration.
    Design:
        Context: Model-native generation horizons differ.
        Concern: A longer generated horizon must not silently grant a different execution budget.
        Decision: Use the same executable prefix count; record truncation and expiration.
        Alternatives: Native50 vs10 consumption confounds model and scheduler.
        Quality Attributes: Comparability. Trade-offs: Native-horizon studies are separate.
    Evidence:
        Internal: Config and counters are serialized by the evaluator.
        External: None. Findings: Prefix length is an experimental factor.
        Metrics: Queue depth and discarded prefix. Confidence: High for mechanics.
    Constraints: One in-flight request is managed by caller.
    Reconsider When: Multi-request streaming is needed.
    References: Module docstring.
    ADR: None.
    """
    def __init__(self,execute_steps=10,mode='fifo'):
        """Initialize a bounded action buffer and explicit gripper hold state.

        Args: execute_steps/mode: See class. Returns: None. Raises: ValueError.
        Design:
            Context: Empty queue is a meaningful control state.
            Concern: Uninitialized gripper state can accidentally close on startup.
            Decision: Begin with LIBERO open-gripper command -1.
            Alternatives: Zero gripper command can change controller behavior.
            Quality Attributes: Predictability. Trade-offs: LIBERO-specific action convention.
        Evidence:
            Internal: Official get_libero_dummy_action uses last coordinate -1.
            External: LeRobot LIBERO wrapper.
            Findings: First six coordinates are relative pose increments.
            Metrics: Hold vector. Confidence: High.
        Constraints: Seven-dimensional relative action space.
        Reconsider When: Gripper convention changes.
        References: Module docstring.
        ADR: None.
        """
        if type(execute_steps) is not int or execute_steps<1 or mode not in ('fifo','aligned'):
            raise ValueError('invalid execution prefix or merge mode')
        self.execute_steps=execute_steps;self.mode=mode;self.queue=deque()
        self.gripper=-1.;self.discarded_expired=0;self.discarded_overlap=0;self.holds=0;self.max_depth=0

    def accept(self,actions,observation_tick,observation_time,ready_tick,request_id):
        """Merge one finished chunk using the predeclared common strategy.

        Args: actions: T×7 array; observation_tick/time: source; ready_tick: control now;
            request_id: unique inference identity.
        Returns: Number of newly queued executable actions.
        Raises: ValueError on invalid dimensions/values or future observations.
        Design:
            Context: A chunk can arrive after several control ticks.
            Concern: Late actions cannot be relabeled as fresh sensor feedback.
            Decision: FIFO retains source age; aligned drops elapsed prefix and replaces overlap.
            Alternatives: Hidden retiming would conceal delays.
            Quality Attributes: Auditability. Trade-offs: Aligned strategy can expire an entire chunk.
        Evidence:
            Internal: Expiration tests include delay longer than prefix.
            External: Generic asynchronous action alignment.
            Findings: Strategies are shared across both models.
            Metrics: Expiration and depth. Confidence: High for mechanics.
        Constraints: Native controller clips bounded commands; raw excess is logged upstream.
        Reconsider When: RTC-trained prefix conditioning is available to both models.
        References: Module docstring.
        ADR: None.
        """
        a=np.asarray(actions,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=7 or a.shape[0]<1 or not np.isfinite(a).all():raise ValueError('invalid action chunk')
        if ready_tick<observation_tick:raise ValueError('future observation tick')
        a=np.clip(a[:self.execute_steps],-1,1)
        offset=0
        if self.mode=='aligned':
            offset=min(ready_tick-observation_tick,len(a));self.discarded_expired+=offset
            self.discarded_overlap+=len(self.queue);self.queue.clear()
        for index in range(offset,len(a)):
            self.queue.append({'action':a[index].copy(),'observation_tick':int(observation_tick),
                               'observation_time':float(observation_time),'chunk_index':index,'request_id':request_id})
        if len(self.queue)>2*self.execute_steps:raise ValueError('queue exceeded bounded double-prefix capacity')
        self.max_depth=max(self.max_depth,len(self.queue))
        return len(a)-offset

    def pop(self):
        """Consume one action or hold zero relative pose while preserving gripper.

        Args: None. Returns: Action metadata including held flag. Raises: None.
        Design:
            Context: Physics must advance even when inference is late.
            Concern: Repeating a previous relative command changes position.
            Decision: Empty queue returns six zeros and last gripper target.
            Alternatives: Pausing physics invalidates real-time comparison.
            Quality Attributes: Fairness. Trade-offs: Waiting can still affect task dynamics.
        Evidence:
            Internal: Hold tests verify no repeated translation/rotation.
            External: Relative control semantics.
            Findings: Hold is not an expert corrective policy.
            Metrics: Hold ticks. Confidence: High.
        Constraints: All models use this identical rule.
        Reconsider When: Controller uses absolute targets.
        References: Module docstring.
        ADR: None.
        """
        if self.queue:
            item=self.queue.popleft();self.gripper=float(item['action'][-1]);item['held']=False;return item
        self.holds+=1
        return {'action':np.array([0.,0.,0.,0.,0.,0.,self.gripper]),'held':True,
                'request_id':None,'observation_tick':None,'observation_time':None,'chunk_index':None}

    def __len__(self):
        """Return the remaining action count without exposing mutable buffer entries.

        Args: None. Returns: int. Raises: None.
        Design:
            Context: Caller triggers inference by remaining depth.
            Concern: Queue internals should not be mutated externally.
            Decision: Read-only count. Alternatives: Public deque mutation bypasses invariants.
            Quality Attributes: Maintainability. Trade-offs: Minimal API.
        Evidence:
            Internal: Runner uses one in-flight request. External: None.
            Findings: Count is deterministic. Metrics: Depth. Confidence: High.
        Constraints: Single control-thread access.
        Reconsider When: Multiple consumers are added.
        References: Module docstring.
        ADR: None.
        """
        return len(self.queue)
