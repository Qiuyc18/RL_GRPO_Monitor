from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from monitor.events import PhaseEvent

from .base import emit_event, resolve_physical_gpu_id

if TYPE_CHECKING:
    from monitor.rollout_stats import RolloutStatsRecorder
else:  # pragma: no cover
    RolloutStatsRecorder = Any


@dataclass
class VerlMonitorBridge:
    """给 veRL 或其他自定义训练循环使用的轻量事件桥。"""

    monitor: object | None
    gpu_id: int
    mode: str = "verl"
    trainer_role: str = "trainer"
    rollout_role: str = "rollout"
    rollout_recorder: RolloutStatsRecorder | None = None

    @classmethod
    def from_local_rank(
        cls,
        monitor,
        local_rank: int,
        *,
        mode: str = "verl",
        rollout_recorder: RolloutStatsRecorder | None = None,
    ):
        return cls(
            monitor=monitor,
            gpu_id=resolve_physical_gpu_id(local_rank),
            mode=mode,
            rollout_recorder=rollout_recorder,
        )

    def emit(self, event_type: PhaseEvent, *, step=None, role: str | None = None):
        emit_event(
            self.monitor,
            event_type,
            gpu_id=self.gpu_id,
            mode=self.mode,
            role=role or self.trainer_role,
            step=step,
        )

    def step_start(self, step=None):
        self.emit(PhaseEvent.STEP_START, step=step)

    def step_end(self, step=None):
        self.emit(PhaseEvent.STEP_END, step=step)

    def rollout_start(self, step=None):
        self.emit(PhaseEvent.ROLLOUT_PHASE_START, step=step, role=self.rollout_role)

    def rollout_end(self, step=None):
        self.emit(PhaseEvent.ROLLOUT_PHASE_END, step=step, role=self.rollout_role)

    def batch_prep_start(self, step=None):
        self.emit(PhaseEvent.BATCH_PREP_START, step=step)

    def batch_prep_end(self, step=None):
        self.emit(PhaseEvent.BATCH_PREP_END, step=step)

    def reward_start(self, step=None):
        self.emit(PhaseEvent.REWARD_CALC_START, step=step)

    def reward_end(self, step=None):
        self.emit(PhaseEvent.REWARD_CALC_END, step=step)

    def forward_start(self, step=None):
        self.emit(PhaseEvent.FORWARD_START, step=step)

    def forward_end(self, step=None):
        self.emit(PhaseEvent.FORWARD_END, step=step)

    def backward_start(self, step=None):
        self.emit(PhaseEvent.BACKWARD_START, step=step)

    def backward_end(self, step=None):
        self.emit(PhaseEvent.BACKWARD_END, step=step)

    def optim_step_start(self, step=None):
        self.emit(PhaseEvent.OPTIM_STEP_START, step=step)

    def optim_step_end(self, step=None):
        self.emit(PhaseEvent.OPTIM_STEP_END, step=step)

    def weight_sync_start(self, step=None):
        self.emit(PhaseEvent.WEIGHT_SYNC_START, step=step)

    def weight_sync_end(self, step=None):
        self.emit(PhaseEvent.WEIGHT_SYNC_END, step=step)

    def record_rollout_lengths(
        self,
        *,
        step_id: int,
        lengths: list[int],
        num_generations: int,
    ):
        if self.rollout_recorder is None:
            return
        self.rollout_recorder.record(
            step_id=step_id,
            lengths=lengths,
            num_generations=num_generations,
            mode=self.mode,
        )
