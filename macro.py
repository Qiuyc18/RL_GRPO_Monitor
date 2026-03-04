import enum


class PhaseEvent(enum.Enum):
    """统一 GRPO 阶段事件，与 Colocate/External 模式解耦。"""
    STEP_START = 0
    STEP_END = 1
    ROLLOUT_PHASE_START = 2
    ROLLOUT_PHASE_END = 3
    REWARD_CALC_START = 4
    REWARD_CALC_END = 5
    BATCH_PREP_START = 6
    BATCH_PREP_END = 7
    FORWARD_START = 8
    FORWARD_END = 9
    BACKWARD_START = 10
    BACKWARD_END = 11
    OPTIM_STEP_START = 12
    OPTIM_STEP_END = 13
    WEIGHT_SYNC_START = 14
    WEIGHT_SYNC_END = 15


# 兼容旧代码
Event = PhaseEvent
