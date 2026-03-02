import enum


class RolloutEvent(enum.Enum):
    """Rollout 侧（如 vLLM）GPU 阶段事件。"""
    # 生成阶段
    GENERATE_START = 0
    GENERATE_END = 1
    # 可扩展：PREPARE_START, PREPARE_END 等


class TrainerEvent(enum.Enum):
    """Trainer 侧 GPU 阶段事件。"""
    VLLM_WAIT_START = 0
    VLLM_WAIT_END = 1
    TOKENIZE_START = 2
    TOKENIZE_END = 3
    REWARD_CALC_START = 4
    REWARD_CALC_END = 5
    FORWARD_START = 6
    FORWARD_END = 7
    BACKWARD_START = 8
    BACKWARD_END = 9


# 兼容旧代码
Event = TrainerEvent
