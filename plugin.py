import os
import functools
from macro import PhaseEvent
from monitor.monitor import Monitor
from swift.trainers.rlhf_trainer import GRPOTrainer
from swift.utils.env import get_dist_setting


def get_physical_gpu_id(local_rank: int) -> int:
    """从 CUDA_VISIBLE_DEVICES + local_rank 得到物理 GPU 编号，供 TensorBoard/事件与 NVML 一致。"""
    if local_rank < 0:
        local_rank = 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return local_rank
    parts = [p.strip() for p in visible.split(",") if p.strip()]
    if local_rank < len(parts):
        return int(parts[local_rank])
    return local_rank


def _ev(monitor, event, gpu_id, mode=None, role="trainer"):
    if monitor:
        monitor.add_event(event, gpu_id=gpu_id, mode=mode, role=role)


def monkey_patch_common(monitor: Monitor | None, physical_gpu_id: int, mode: str | None = None):
    """Common trainer events: STEP, REWARD_CALC, BATCH_PREP_END, FORWARD, BACKWARD."""
    if not hasattr(GRPOTrainer, "_original_compute_loss"):
        GRPOTrainer._original_compute_loss = GRPOTrainer.compute_loss  # type: ignore

    @functools.wraps(GRPOTrainer._original_compute_loss)  # type: ignore
    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        _ev(monitor, PhaseEvent.BATCH_PREP_END, physical_gpu_id, mode)
        _ev(monitor, PhaseEvent.FORWARD_START, physical_gpu_id, mode)
        result = GRPOTrainer._original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)  # type: ignore
        _ev(monitor, PhaseEvent.FORWARD_END, physical_gpu_id, mode)
        _ev(monitor, PhaseEvent.BACKWARD_START, physical_gpu_id, mode)
        return result

    GRPOTrainer.compute_loss = patched_compute_loss

    target_score_method = "_score_completions"
    if hasattr(GRPOTrainer, target_score_method):
        original_score_method = getattr(GRPOTrainer, target_score_method)

        @functools.wraps(original_score_method)
        def patched_score_completions(self, *args, **kwargs):
            _ev(monitor, PhaseEvent.REWARD_CALC_START, physical_gpu_id, mode)
            result = original_score_method(self, *args, **kwargs)
            _ev(monitor, PhaseEvent.REWARD_CALC_END, physical_gpu_id, mode)
            return result

        setattr(GRPOTrainer, target_score_method, patched_score_completions)

    if not hasattr(GRPOTrainer, "_original_training_step"):
        GRPOTrainer._original_training_step = GRPOTrainer.training_step  # type: ignore

    @functools.wraps(GRPOTrainer._original_training_step)  # type: ignore
    def patched_training_step(self, *args, **kwargs):
        _ev(monitor, PhaseEvent.STEP_START, physical_gpu_id, mode)
        res = GRPOTrainer._original_training_step(self, *args, **kwargs)  # type: ignore
        _ev(monitor, PhaseEvent.BACKWARD_END, physical_gpu_id, mode)
        _ev(monitor, PhaseEvent.STEP_END, physical_gpu_id, mode)
        return res

    GRPOTrainer.training_step = patched_training_step
    print("[MonkeyPatch] Common trainer events: STEP, REWARD_CALC, BATCH_PREP_END, FORWARD, BACKWARD")


def monkey_patch_external_rollout(monitor: Monitor | None, physical_gpu_id: int):
    """External mode: trainer 等待 vLLM 生成 + BATCH_PREP_START."""
    target_generate_method = "_generate_and_score_completions"
    original_generate_method = getattr(GRPOTrainer, target_generate_method)

    @functools.wraps(original_generate_method)
    def patched_generate_and_score(self, *args, **kwargs):
        _ev(monitor, PhaseEvent.ROLLOUT_PHASE_START, physical_gpu_id, mode="external")
        result = original_generate_method(self, *args, **kwargs)
        _ev(monitor, PhaseEvent.ROLLOUT_PHASE_END, physical_gpu_id, mode="external")
        _ev(monitor, PhaseEvent.BATCH_PREP_START, physical_gpu_id, mode="external")
        return result

    setattr(GRPOTrainer, target_generate_method, patched_generate_and_score)
    print("[MonkeyPatch] External rollout events: ROLLOUT_PHASE_*, BATCH_PREP_START")


def monkey_patch_colocate(monitor: Monitor | None, physical_gpu_id: int):
    """Colocate 模式事件注入。TODO: 待 Colocate 实现时补全。"""
    # TODO: Colocate 下 rollout 与 trainer 同进程，在此注入 Colocate 特有事件
    print("[MonkeyPatch] Colocate events: TODO")


def monkey_patch(monitor: Monitor | None):
    """Wrap GRPOTrainer：common + external rollout，保持 External 模式行为不变。"""
    _, local_rank, _, _ = get_dist_setting()
    physical_gpu_id = get_physical_gpu_id(local_rank)
    monkey_patch_common(monitor, physical_gpu_id, mode="external")
    monkey_patch_external_rollout(monitor, physical_gpu_id)
    monkey_patch_colocate(monitor, physical_gpu_id)


if __name__ == "__main__":
    monitor = Monitor(
        output_file_path="logs/test/gpu_metrics.csv",
        events_file_path="logs/test/gpu_events.csv",
        interval=1,
        enable_metrics=True,
        platform="nvidia",
    )
    monitor.start()
    monkey_patch(monitor)
    monitor.stop()
