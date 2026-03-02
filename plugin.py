import os
import functools
from macro import Event
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


def monkey_patch(monitor: Monitor | None):
    """Wrap GRPOTrainer to inject detailed monitor events."""
    
    # --- 1. Patch Rollout (VLLM Interaction) ---
    # ms-swift 的 GRPO 逻辑通常在 _generate_and_score_completions 中
    # 尝试截获 vLLM 的调用（通常是 self.llm.generate 或类似）

    target_generate_method = "_generate_and_score_completions"
    original_generate_method = getattr(GRPOTrainer, target_generate_method)
    
    _, local_rank, _, _ = get_dist_setting()
    physical_gpu_id = get_physical_gpu_id(local_rank)

    @functools.wraps(original_generate_method)
    def patched_generate_and_score(self, *args, **kwargs):
        # [Event: 开始等待 vLLM 生成]
        if monitor:
            monitor.add_event(Event.VLLM_WAIT_START, gpu_id=physical_gpu_id)

        result = original_generate_method(self, *args, **kwargs)

        if monitor:
            monitor.add_event(Event.VLLM_WAIT_END, gpu_id=physical_gpu_id)
            monitor.add_event(Event.TOKENIZE_START, gpu_id=physical_gpu_id)

        return result

    setattr(GRPOTrainer, target_generate_method, patched_generate_and_score)

    # --- 1.1 Patch Reward 计算 (_score_completions 在 _generate_and_score_completions 内、generate 之后调用) ---
    target_score_method = "_score_completions"
    if hasattr(GRPOTrainer, target_score_method):
        original_score_method = getattr(GRPOTrainer, target_score_method)

        @functools.wraps(original_score_method)
        def patched_score_completions(self, *args, **kwargs):
            if monitor:
                monitor.add_event(Event.REWARD_CALC_START, gpu_id=physical_gpu_id)
            result = original_score_method(self, *args, **kwargs)
            if monitor:
                monitor.add_event(Event.REWARD_CALC_END, gpu_id=physical_gpu_id)
            return result

        setattr(GRPOTrainer, target_score_method, patched_score_completions)

    # --- 2. Patch Compute Loss (Forward Pass) ---
    # compute_loss 是 Training Loop 中 Forward 阶段的核心
    
    if not hasattr(GRPOTrainer, "_original_compute_loss"):
        GRPOTrainer._original_compute_loss = GRPOTrainer.compute_loss # type: ignore

    @functools.wraps(GRPOTrainer._original_compute_loss) # type: ignore
    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # [Event: Tokenize 结束 (因为进入 compute_loss 前数据必须这就绪)]
        # [Event: Forward 开始]
        if monitor:
            monitor.add_event(Event.TOKENIZE_END, gpu_id=physical_gpu_id)
            monitor.add_event(Event.FORWARD_START, gpu_id=physical_gpu_id)

        result = GRPOTrainer._original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch) # type: ignore

        if monitor:
            monitor.add_event(Event.FORWARD_END, gpu_id=physical_gpu_id)
            monitor.add_event(Event.BACKWARD_START, gpu_id=physical_gpu_id)
            
        return result

    GRPOTrainer.compute_loss = patched_compute_loss

    # --- 3. Patch Training Step (Total Loop & Backward End) ---
    
    if not hasattr(GRPOTrainer, "_original_training_step"):
        GRPOTrainer._original_training_step = GRPOTrainer.training_step # type: ignore

    @functools.wraps(GRPOTrainer._original_training_step) # type: ignore
    def patched_training_step(self, *args, **kwargs):
        # 注意：training_step 内部调用了 compute_loss (Forward) 和 backward
        # 所以这里的开始其实是整个 step 的开始
        
        res = GRPOTrainer._original_training_step(self, *args, **kwargs) # type: ignore
        
        if monitor:
            monitor.add_event(Event.BACKWARD_END, gpu_id=physical_gpu_id)
            
        return res

    GRPOTrainer.training_step = patched_training_step
    
    print(f"[MonkeyPatch] Successfully injected probes: Generate, ComputeLoss, TrainingStep")


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