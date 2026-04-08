import functools

from monitor.events import PhaseEvent
from monitor.monitor import Monitor

from .base import emit_event, resolve_physical_gpu_id

try:
    from monitor.rollout_stats import RolloutStatsRecorder
except ImportError:  # pragma: no cover
    RolloutStatsRecorder = None  # type: ignore


def _require_grpo_trainer():
    try:
        from swift.trainers.rlhf_trainer import GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ms-swift 适配层需要先安装 `swift`，当前仓库本身已不再内置 ms-swift 子模块。"
        ) from exc
    return GRPOTrainer


def _require_dist_setting():
    try:
        from swift.utils.env import get_dist_setting
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ms-swift 适配层需要 `swift.utils.env.get_dist_setting`。"
        ) from exc
    return get_dist_setting


def _require_rollout_deploy():
    try:
        from swift.llm.infer.rollout import SwiftRolloutDeploy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ms-swift rollout 适配层需要 `swift.llm.infer.rollout.SwiftRolloutDeploy`。"
        ) from exc
    return SwiftRolloutDeploy


def get_physical_gpu_id(local_rank: int) -> int:
    return resolve_physical_gpu_id(local_rank)


def monkey_patch_common(
    monitor: Monitor | None,
    physical_gpu_id: int,
    mode: str | None = None,
):
    """Common trainer events: STEP, REWARD_CALC, BATCH_PREP_END, FORWARD, BACKWARD."""
    trainer_cls = _require_grpo_trainer()

    if not hasattr(trainer_cls, "_original_compute_loss"):
        trainer_cls._original_compute_loss = trainer_cls.compute_loss  # type: ignore[attr-defined]

    @functools.wraps(trainer_cls._original_compute_loss)  # type: ignore[attr-defined]
    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        emit_event(monitor, PhaseEvent.BATCH_PREP_END, physical_gpu_id, mode=mode)
        emit_event(monitor, PhaseEvent.FORWARD_START, physical_gpu_id, mode=mode)
        result = trainer_cls._original_compute_loss(  # type: ignore[attr-defined]
            self,
            model,
            inputs,
            return_outputs,
            num_items_in_batch,
        )
        emit_event(monitor, PhaseEvent.FORWARD_END, physical_gpu_id, mode=mode)
        emit_event(monitor, PhaseEvent.BACKWARD_START, physical_gpu_id, mode=mode)
        return result

    trainer_cls.compute_loss = patched_compute_loss

    target_score_method = "_score_completions"
    if hasattr(trainer_cls, target_score_method):
        original_score_method = getattr(trainer_cls, target_score_method)

        @functools.wraps(original_score_method)
        def patched_score_completions(self, *args, **kwargs):
            emit_event(monitor, PhaseEvent.REWARD_CALC_START, physical_gpu_id, mode=mode)
            result = original_score_method(self, *args, **kwargs)
            emit_event(monitor, PhaseEvent.REWARD_CALC_END, physical_gpu_id, mode=mode)
            return result

        setattr(trainer_cls, target_score_method, patched_score_completions)

    if not hasattr(trainer_cls, "_original_training_step"):
        trainer_cls._original_training_step = trainer_cls.training_step  # type: ignore[attr-defined]

    @functools.wraps(trainer_cls._original_training_step)  # type: ignore[attr-defined]
    def patched_training_step(self, *args, **kwargs):
        emit_event(monitor, PhaseEvent.STEP_START, physical_gpu_id, mode=mode)
        result = trainer_cls._original_training_step(self, *args, **kwargs)  # type: ignore[attr-defined]
        emit_event(monitor, PhaseEvent.BACKWARD_END, physical_gpu_id, mode=mode)
        emit_event(monitor, PhaseEvent.STEP_END, physical_gpu_id, mode=mode)
        return result

    trainer_cls.training_step = patched_training_step
    print("[MonkeyPatch] Common trainer events: STEP, REWARD_CALC, BATCH_PREP_END, FORWARD, BACKWARD")


def monkey_patch_external_rollout(monitor: Monitor | None, physical_gpu_id: int):
    """External mode: trainer 等待 rollout 完成，然后开始 batch prepare。"""
    trainer_cls = _require_grpo_trainer()
    target_generate_method = "_generate_and_score_completions"
    original_generate_method = getattr(trainer_cls, target_generate_method)

    @functools.wraps(original_generate_method)
    def patched_generate_and_score(self, *args, **kwargs):
        emit_event(
            monitor,
            PhaseEvent.ROLLOUT_PHASE_START,
            physical_gpu_id,
            mode="external",
        )
        result = original_generate_method(self, *args, **kwargs)
        emit_event(
            monitor,
            PhaseEvent.ROLLOUT_PHASE_END,
            physical_gpu_id,
            mode="external",
        )
        emit_event(
            monitor,
            PhaseEvent.BATCH_PREP_START,
            physical_gpu_id,
            mode="external",
        )
        return result

    setattr(trainer_cls, target_generate_method, patched_generate_and_score)
    print("[MonkeyPatch] External rollout events: ROLLOUT_PHASE_*, BATCH_PREP_START")


def monkey_patch_colocate(monitor: Monitor | None, physical_gpu_id: int):
    """Colocate 模式事件注入。"""
    _ = monitor
    _ = physical_gpu_id
    print("[MonkeyPatch] Colocate events: TODO")


def _extract_completion_length(inp: dict, trainer) -> int:
    token_ids = inp.get("response_token_ids")
    if token_ids:
        if isinstance(token_ids, list):
            if token_ids and isinstance(token_ids[0], list):
                return sum(len(turn) for turn in token_ids)
            return len(token_ids)

    try:
        content = inp["messages"][-1]["content"]
        if isinstance(content, list):
            return len(content)
        if isinstance(content, dict) and "token_ids" in content:
            return len(content["token_ids"])
        if isinstance(content, str):
            tokenizer = getattr(trainer, "processing_class", None) or getattr(
                trainer,
                "tokenizer",
                None,
            )
            if tokenizer:
                return len(tokenizer.encode(content))
            return len(content)
    except (KeyError, IndexError, TypeError):
        pass

    return 0


def monkey_patch_rollout_stats(
    recorder: RolloutStatsRecorder | None,
    physical_gpu_id: int,
    mode: str = "external",
):
    if recorder is None:
        print("[MonkeyPatch] RolloutStats: recorder is None, skipped")
        return

    trainer_cls = _require_grpo_trainer()
    target = "_generate_completions"
    if hasattr(trainer_cls, target):
        if not hasattr(trainer_cls, "_original_generate_completions"):
            trainer_cls._original_generate_completions = getattr(trainer_cls, target)  # type: ignore[attr-defined]

        @functools.wraps(trainer_cls._original_generate_completions)  # type: ignore[attr-defined]
        def patched_generate_completions(self, inputs):
            result = trainer_cls._original_generate_completions(self, inputs)  # type: ignore[attr-defined]
            try:
                lengths = [_extract_completion_length(inp, self) for inp in result]
                step_id = getattr(self, "_step", -1)
                num_gen = getattr(self, "num_generations", 1)
                recorder.record(step_id, lengths, num_gen, mode=mode)
            except Exception as error:  # pragma: no cover
                print(f"[RolloutStats] Error recording from {target}: {error}")
            return result

        setattr(trainer_cls, target, patched_generate_completions)
        print(f"[MonkeyPatch] RolloutStats: patched {target} on GPU {physical_gpu_id}")
        return

    fallback_target = "_generate_and_score_completions"
    current_method = getattr(trainer_cls, fallback_target)
    if hasattr(trainer_cls, "_rollout_stats_fallback_applied"):
        return

    @functools.wraps(current_method)
    def patched_with_stats(self, inputs):
        result = current_method(self, inputs)
        try:
            lengths = [_extract_completion_length(inp, self) for inp in inputs]
            step_id = getattr(self, "_step", -1)
            num_gen = getattr(self, "num_generations", 1)
            recorder.record(step_id, lengths, num_gen, mode=mode)
        except Exception as error:  # pragma: no cover
            print(f"[RolloutStats] Error recording from {fallback_target}: {error}")
        return result

    setattr(trainer_cls, fallback_target, patched_with_stats)
    trainer_cls._rollout_stats_fallback_applied = True  # type: ignore[attr-defined]
    print(
        f"[MonkeyPatch] RolloutStats: {target} not found, fallback to wrapping {fallback_target}"
    )


def monkey_patch(
    monitor: Monitor | None,
    rollout_recorder: RolloutStatsRecorder | None = None,
):
    """Wrap GRPOTrainer：common + external rollout + rollout stats。"""
    get_dist_setting = _require_dist_setting()
    _, local_rank, _, _ = get_dist_setting()
    physical_gpu_id = get_physical_gpu_id(local_rank)
    monkey_patch_common(monitor, physical_gpu_id, mode="external")
    monkey_patch_external_rollout(monitor, physical_gpu_id)
    monkey_patch_rollout_stats(rollout_recorder, physical_gpu_id, mode="external")
    monkey_patch_colocate(monitor, physical_gpu_id)


def monkey_patch_rollout(monitor: Monitor | None, rollout_gpu_id: int = 0):
    """包装 SwiftRolloutDeploy.infer，在每次生成前后打 rollout 事件。"""
    rollout_deploy_cls = _require_rollout_deploy()
    original_infer = rollout_deploy_cls.infer

    @functools.wraps(original_infer)
    async def patched_infer(self, infer_requests, request_config=None, *, use_tqdm=None):
        emit_event(
            monitor,
            PhaseEvent.ROLLOUT_PHASE_START,
            gpu_id=rollout_gpu_id,
            mode="external",
            role="rollout",
        )
        try:
            return await original_infer(
                self,
                infer_requests,
                request_config=request_config,
                use_tqdm=use_tqdm,
            )
        finally:
            emit_event(
                monitor,
                PhaseEvent.ROLLOUT_PHASE_END,
                gpu_id=rollout_gpu_id,
                mode="external",
                role="rollout",
            )

    rollout_deploy_cls.infer = patched_infer
    print("[MonkeyPatch] Rollout infer -> PhaseEvent.ROLLOUT_PHASE_START/END (role=rollout)")
