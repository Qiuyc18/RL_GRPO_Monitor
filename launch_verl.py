#!/usr/bin/env python3
"""
veRL 训练启动器 — 在 verl.trainer.main_ppo 启动前注入 GPU 监控插桩。

替代 `python3 -m verl.trainer.main_ppo`，用法完全一致：
    python3 monitor/launch_verl.py data.train_files=... actor_rollout_ref.model.path=...

等价于:
    1. patch_task_runner()  → 注入 GPU 监控到 TaskRunner
    2. verl.trainer.main_ppo.main()  → 正常启动 Hydra + veRL 训练
"""
import os
import sys


def main():
    # 注入 GPU 监控（如果环境变量启用）
    if os.environ.get("GPU_PLATFORM"):
        try:
            from plugin_verl import patch_task_runner
            patch_task_runner()
        except ImportError as e:
            print(f"[launch_verl] Warning: GPU monitor plugin not available: {e}")
            print("[launch_verl] Continuing without GPU monitoring...")

    # 启动 veRL 训练（Hydra 入口）
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
