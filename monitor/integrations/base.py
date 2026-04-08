import os


def resolve_physical_gpu_id(local_rank: int, env_var: str = "CUDA_VISIBLE_DEVICES") -> int:
    """从 visible devices 和 local rank 推导物理 GPU 编号。"""
    if local_rank < 0:
        local_rank = 0

    visible = os.environ.get(env_var, "")
    if not visible:
        return local_rank

    parts = [part.strip() for part in visible.split(",") if part.strip()]
    if local_rank >= len(parts):
        return local_rank

    try:
        return int(parts[local_rank])
    except ValueError:
        return local_rank


def emit_event(monitor, event_type, gpu_id, mode=None, role="trainer", step=None):
    if monitor is None:
        return
    monitor.add_event(
        event_type,
        step=step,
        gpu_id=gpu_id,
        mode=mode,
        role=role,
    )
