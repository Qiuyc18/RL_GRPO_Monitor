"""
veRL 训练插桩插件 — 将 GPU 监控事件注入 veRL 的 RayPPOTrainer 训练循环。

veRL 的训练循环在 RayPPOTrainer.fit() 中，通过 marked_timer 标记了各阶段：
  gen → reward → old_log_prob → RefPolicy → adv → update_critic → update_actor → testing

本插件通过 monkey-patch RayPPOTrainer.fit()，在每个 marked_timer 阶段前后
插入 Monitor 事件，实现 GPU 利用率与训练阶段的精确对齐。

注意：veRL 的 fit() 运行在 Ray actor 内部（TaskRunner），因此 monkey-patch
需要在 TaskRunner.run() 调用 trainer.fit() 之前执行。

用法（在容器内修改 verl 源码）:
    # 方法 1: 修改 main_ppo.py 的 TaskRunner.run()
    在 trainer.fit() 之前插入:
        from plugin_verl import patch_verl_trainer
        patch_verl_trainer(trainer, platform="amd")

    # 方法 2: 使用 patch_main_ppo() 自动注入
    在训练脚本启动前:
        python3 -c "from monitor.plugin_verl import install; install()"
"""

import functools
import os
import sys
from pathlib import Path

# 确保 monitor 包可导入
_monitor_root = Path(__file__).resolve().parent
if str(_monitor_root) not in sys.path:
    sys.path.insert(0, str(_monitor_root))

from macro import PhaseEvent
from monitor.monitor import Monitor


# veRL marked_timer 阶段名 → Monitor PhaseEvent 映射
PHASE_MAP = {
    "gen":             (PhaseEvent.ROLLOUT_PHASE_START, PhaseEvent.ROLLOUT_PHASE_END),
    "reward":          (PhaseEvent.REWARD_CALC_START,   PhaseEvent.REWARD_CALC_END),
    "update_actor":    (PhaseEvent.FORWARD_START,       PhaseEvent.BACKWARD_END),
    "update_critic":   (PhaseEvent.OPTIM_STEP_START,    PhaseEvent.OPTIM_STEP_END),
    "adv":             (PhaseEvent.BATCH_PREP_START,    PhaseEvent.BATCH_PREP_END),
    "testing":         (PhaseEvent.REWARD_CALC_START,   PhaseEvent.REWARD_CALC_END),
}


def _emit(monitor: Monitor, event: PhaseEvent, gpu_id: int = 0, phase: str = ""):
    """向 Monitor 发送事件。"""
    if monitor:
        monitor.add_event(event, gpu_id=gpu_id, mode="verl", role=phase)


def patch_verl_trainer(trainer, platform: str = "amd", output_dir: str = "logs/verl"):
    """Monkey-patch RayPPOTrainer.fit() 以注入 GPU 监控事件。

    Args:
        trainer: RayPPOTrainer 实例
        platform: GPU 平台 ("amd" 或 "nvidia")
        output_dir: 监控日志输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    monitor = Monitor(
        platform=platform,
        output_file_path=os.path.join(output_dir, "gpu_metrics.csv"),
        events_file_path=os.path.join(output_dir, "gpu_events.csv"),
        interval=0.5,
        buffer_seconds=10,
        write_interval=2.0,
    )

    original_fit = trainer.fit

    @functools.wraps(original_fit)
    def patched_fit():
        monitor.start()
        _emit(monitor, PhaseEvent.STEP_START, phase="training_start")
        try:
            # Patch marked_timer 以注入事件
            _patch_marked_timer(monitor)
            return original_fit()
        finally:
            _emit(monitor, PhaseEvent.STEP_END, phase="training_end")
            monitor.stop()
            print(f"[plugin_verl] GPU monitor logs saved to {output_dir}/")

    trainer.fit = patched_fit
    print(f"[plugin_verl] Patched RayPPOTrainer.fit() with GPU monitor (platform={platform})")
    return monitor


def _patch_marked_timer(monitor: Monitor):
    """Patch veRL 的 marked_timer context manager 以在阶段边界发送事件。"""
    try:
        import verl.utils.tracking as tracking_module
        if not hasattr(tracking_module, "_original_marked_timer"):
            from contextlib import contextmanager

            # veRL 的 marked_timer 定义在 ray_trainer.py 或 tracking.py 中
            # 我们需要找到它并包装
            from verl.trainer.ppo.ray_trainer import marked_timer as original_marked_timer

            @contextmanager
            def patched_marked_timer(name, timing_raw, color=None):
                phase_events = PHASE_MAP.get(name)
                if phase_events:
                    _emit(monitor, phase_events[0], phase=name)

                with original_marked_timer(name, timing_raw, color=color) as val:
                    yield val

                if phase_events:
                    _emit(monitor, phase_events[1], phase=name)

            # 替换全局引用
            import verl.trainer.ppo.ray_trainer as ray_trainer_module
            ray_trainer_module.marked_timer = patched_marked_timer
            tracking_module._original_marked_timer = original_marked_timer
            print("[plugin_verl] Patched marked_timer for phase event injection")
    except (ImportError, AttributeError) as e:
        print(f"[plugin_verl] Warning: Could not patch marked_timer: {e}")
        print("[plugin_verl] GPU metrics will still be collected, but without phase alignment")


def patch_task_runner():
    """Patch TaskRunner.run() 以自动注入 GPU 监控。

    在训练脚本中 import verl 之前调用此函数。
    """
    import verl.trainer.main_ppo as main_ppo_module

    OriginalTaskRunner = main_ppo_module.TaskRunner

    class PatchedTaskRunner(OriginalTaskRunner):
        def run(self, config):
            # 复用原始 run() 的逻辑，但在 trainer.fit() 前注入监控
            # 由于原始 run() 直接调用 trainer.fit()，我们需要 override 整个方法
            # 这里通过 monkey-patch RayPPOTrainer 类来实现
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer

            original_init = RayPPOTrainer.__init__

            @functools.wraps(original_init)
            def patched_init(self_trainer, *args, **kwargs):
                original_init(self_trainer, *args, **kwargs)
                platform = os.environ.get("GPU_PLATFORM", "amd")
                output_dir = os.environ.get("GPU_MONITOR_OUTPUT", "logs/verl")
                patch_verl_trainer(self_trainer, platform=platform, output_dir=output_dir)

            RayPPOTrainer.__init__ = patched_init

            try:
                return super().run(config)
            finally:
                # 恢复原始 __init__
                RayPPOTrainer.__init__ = original_init

    main_ppo_module.TaskRunner = PatchedTaskRunner
    print("[plugin_verl] Patched TaskRunner for automatic GPU monitoring")


def install():
    """便捷安装入口，自动 patch TaskRunner。

    在训练脚本中使用:
        python3 -c "from monitor.plugin_verl import install; install()"
    或在训练脚本开头:
        export VERL_GPU_MONITOR=1
    """
    patch_task_runner()
