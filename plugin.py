import os
import functools
from macro import PhaseEvent
from monitor.monitor import Monitor
from rollout_stats import RolloutStatsRecorder
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


def _extract_completion_length(inp: dict, trainer) -> int:
    """从单条 completion input 中提取 token 长度。

    优先级：
      1. response_token_ids（最准确，直接就是 token 列表）
      2. messages[-1]['content'] 为 list/dict 时按 token_ids 计算
      3. messages[-1]['content'] 为 str 时用 tokenizer 重新编码
      4. 全部失败返回 0
    """
    # 1) response_token_ids —— _generate_completions 执行后一般都有
    token_ids = inp.get("response_token_ids")
    if token_ids:
        if isinstance(token_ids, list):
            if token_ids and isinstance(token_ids[0], list):
                # 多轮对话：嵌套 list
                return sum(len(turn) for turn in token_ids)
            return len(token_ids)

    # 2) messages[-1]['content'] 可能直接就是 token_ids
    try:
        content = inp["messages"][-1]["content"]
        if isinstance(content, list):
            return len(content)
        if isinstance(content, dict) and "token_ids" in content:
            return len(content["token_ids"])
        # 3) content 是纯文本，用 tokenizer 编码
        if isinstance(content, str):
            tokenizer = getattr(trainer, "processing_class", None) or getattr(
                trainer, "tokenizer", None
            )
            if tokenizer:
                return len(tokenizer.encode(content))
            # 最后手段：字符数
            return len(content)
    except (KeyError, IndexError, TypeError):
        pass

    return 0


def monkey_patch_rollout_stats(
    recorder: RolloutStatsRecorder | None,
    physical_gpu_id: int,
    mode: str = "external",
):
    """Patch _generate_completions 以记录 completion 长度统计。

    如果当前 ms-swift 版本没有 _generate_completions，退回到
    _generate_and_score_completions 返回后处理（需要已有 external rollout patch）。
    """
    if recorder is None:
        print("[MonkeyPatch] RolloutStats: recorder is None, skipped")
        return

    # ---- 优先 patch _generate_completions ----
    target = "_generate_completions"
    if hasattr(GRPOTrainer, target):
        if not hasattr(GRPOTrainer, "_original_generate_completions"):
            GRPOTrainer._original_generate_completions = getattr(GRPOTrainer, target)

        @functools.wraps(GRPOTrainer._original_generate_completions)
        def patched_generate_completions(self, inputs):
            result = GRPOTrainer._original_generate_completions(self, inputs)
            try:
                lengths = [_extract_completion_length(inp, self) for inp in result]
                step_id = getattr(self, "_step", -1)
                num_gen = getattr(self, "num_generations", 1)
                recorder.record(step_id, lengths, num_gen, mode=mode)
            except Exception as e:
                print(f"[RolloutStats] Error recording from {target}: {e}")
            return result

        setattr(GRPOTrainer, target, patched_generate_completions)
        print(f"[MonkeyPatch] RolloutStats: patched {target}")
        return

    # ---- fallback: 在 _generate_and_score_completions 外层追加 ----
    fallback_target = "_generate_and_score_completions"
    # 取当前（可能已被 phase event 包过一层）的版本
    current_method = getattr(GRPOTrainer, fallback_target)
    if not hasattr(GRPOTrainer, "_rollout_stats_fallback_applied"):

        @functools.wraps(current_method)
        def patched_with_stats(self, inputs):
            result = current_method(self, inputs)
            # result 是 batch_encoded_inputs（list of mini-batches），
            # 此时 inputs 已被就地修改，可以直接从 inputs 取长度
            try:
                lengths = [_extract_completion_length(inp, self) for inp in inputs]
                step_id = getattr(self, "_step", -1)
                num_gen = getattr(self, "num_generations", 1)
                recorder.record(step_id, lengths, num_gen, mode=mode)
            except Exception as e:
                print(f"[RolloutStats] Error recording from {fallback_target}: {e}")
            return result

        setattr(GRPOTrainer, fallback_target, patched_with_stats)
        GRPOTrainer._rollout_stats_fallback_applied = True
        print(
            f"[MonkeyPatch] RolloutStats: {target} not found, "
            f"fallback to wrapping {fallback_target}"
        )


def monkey_patch(monitor: Monitor | None, rollout_recorder: RolloutStatsRecorder | None = None):
    """Wrap GRPOTrainer：common + external rollout + rollout stats，保持 External 模式行为不变。"""
    _, local_rank, _, _ = get_dist_setting()
    physical_gpu_id = get_physical_gpu_id(local_rank)
    monkey_patch_common(monitor, physical_gpu_id, mode="external")
    monkey_patch_external_rollout(monitor, physical_gpu_id)
    monkey_patch_rollout_stats(rollout_recorder, physical_gpu_id, mode="external")
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
