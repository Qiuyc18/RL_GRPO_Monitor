import functools
from macro import Event
from monitor.monitor import Monitor
from swift.trainers.rlhf_trainer import GRPOTrainer
from swift.utils.env import get_dist_setting

def monkey_patch(monitor: Monitor | None):
    """Wrap GRPOTrainer to inject detailed monitor events."""
    
    # --- 1. Patch Rollout (VLLM Interaction) ---
    # ms-swift 的 GRPO 逻辑通常在 _generate_and_score_completions 中
    # 尝试截获 vLLM 的调用（通常是 self.llm.generate 或类似）

    target_generate_method = "_generate_and_score_completions"
    original_generate_method = getattr(GRPOTrainer, target_generate_method)
    
    _, local_rank, _, _ = get_dist_setting()

    @functools.wraps(original_generate_method)
    def patched_generate_and_score(self, *args, **kwargs):
        # [Event: 开始等待 vLLM 生成]
        # 注意：这里其实包含了 "生成" + "评分" 两个动作
        # 如果要拆得更细，需要看 ms-swift 源码中具体的 llm.generate 调用
        if monitor:
            monitor.add_event(Event.VLLM_WAIT_START, gpu_id=local_rank)

        # 执行原始生成与评分
        result = original_generate_method(self, *args, **kwargs)

        # [Event: 生成与评分结束]
        if monitor:
            monitor.add_event(Event.VLLM_WAIT_END, gpu_id=local_rank)
            # 在这里，数据已经回到 Training GPU，准备开始处理
            # 我们可以标记一个 "数据准备/Tokenize" 的虚拟阶段，
            # 因为 GRPO Trainer 紧接着就会把这些 text 转成 tensor
            monitor.add_event(Event.TOKENIZE_START, gpu_id=local_rank)

        return result

    setattr(GRPOTrainer, target_generate_method, patched_generate_and_score)

    # --- 2. Patch Compute Loss (Forward Pass) ---
    # compute_loss 是 Training Loop 中 Forward 阶段的核心
    
    if not hasattr(GRPOTrainer, "_original_compute_loss"):
        GRPOTrainer._original_compute_loss = GRPOTrainer.compute_loss # type: ignore

    @functools.wraps(GRPOTrainer._original_compute_loss) # type: ignore
    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # [Event: Tokenize 结束 (因为进入 compute_loss 前数据必须这就绪)]
        # [Event: Forward 开始]
        if monitor:
            monitor.add_event(Event.TOKENIZE_END, gpu_id=local_rank) 
            monitor.add_event(Event.FORWARD_START, gpu_id=local_rank)
        
        # 执行 Forward
        result = GRPOTrainer._original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch) # type: ignore
        
        # [Event: Forward 结束]
        if monitor:
            monitor.add_event(Event.FORWARD_END, gpu_id=local_rank)
            # Forward 结束紧接着就是 Backward，我们在这里标记 Backward 开始
            monitor.add_event(Event.BACKWARD_START, gpu_id=local_rank)
            
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
            # 当 training_step 返回时，Backward 和 Optimizer step 都做完了
            monitor.add_event(Event.BACKWARD_END, gpu_id=local_rank)
            
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