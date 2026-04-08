import argparse
import os
import random
import time
from pathlib import Path

from monitor import Monitor
from monitor.integrations.verl import VerlMonitorBridge
from monitor.rollout_stats import RolloutStatsRecorder

LOG_DIRECTORY = "logs"
TB_TIME_ANCHOR_PATH = os.path.join(LOG_DIRECTORY, ".time_anchor")


def parse_args():
    parser = argparse.ArgumentParser(
        description="通用训练事件示例入口，可用于验证 GPU 指标与阶段事件是否正常落盘。",
    )
    parser.add_argument("--platform", default=os.getenv("PLATFORM", "nvidia"))
    parser.add_argument("--log-dir", default=f"{LOG_DIRECTORY}/demo_run")
    parser.add_argument("--interval", type=float, default=0.1, help="监控采样间隔（秒）")
    parser.add_argument("--steps", type=int, default=3, help="模拟训练 step 数")
    parser.add_argument("--local-rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")))
    parser.add_argument("--mode", default="verl", help="事件模式标记，默认使用 verl")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--base-length", type=int, default=128)
    parser.add_argument("--rollout-seconds", type=float, default=0.5)
    parser.add_argument("--batch-prep-seconds", type=float, default=0.15)
    parser.add_argument("--reward-seconds", type=float, default=0.1)
    parser.add_argument("--forward-seconds", type=float, default=0.2)
    parser.add_argument("--backward-seconds", type=float, default=0.25)
    parser.add_argument("--optim-seconds", type=float, default=0.1)
    return parser.parse_args()


def build_monitor(args):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(
        platform=args.platform,
        output_file_path=str(log_dir / "gpu_metrics.csv"),
        events_file_path=str(log_dir / "gpu_events.csv"),
        interval=args.interval,
        enable_metrics=True,
        tb_time_anchor_path=TB_TIME_ANCHOR_PATH,
    )


def build_rollout_lengths(step: int, num_generations: int, base_length: int) -> list[int]:
    random.seed(step)
    return [
        base_length + step * 8 + idx * 16 + random.randint(0, 12)
        for idx in range(num_generations)
    ]


def simulate_step(step: int, bridge: VerlMonitorBridge, args):
    bridge.step_start(step=step)

    bridge.rollout_start(step=step)
    time.sleep(args.rollout_seconds)
    bridge.rollout_end(step=step)
    bridge.record_rollout_lengths(
        step_id=step,
        lengths=build_rollout_lengths(step, args.num_generations, args.base_length),
        num_generations=args.num_generations,
    )

    bridge.batch_prep_start(step=step)
    time.sleep(args.batch_prep_seconds)
    bridge.batch_prep_end(step=step)

    bridge.reward_start(step=step)
    time.sleep(args.reward_seconds)
    bridge.reward_end(step=step)

    bridge.forward_start(step=step)
    time.sleep(args.forward_seconds)
    bridge.forward_end(step=step)

    bridge.backward_start(step=step)
    time.sleep(args.backward_seconds)
    bridge.backward_end(step=step)

    bridge.optim_step_start(step=step)
    time.sleep(args.optim_seconds)
    bridge.optim_step_end(step=step)
    bridge.step_end(step=step)


def main():
    args = parse_args()
    monitor = build_monitor(args)
    recorder = RolloutStatsRecorder(output_dir=args.log_dir)
    bridge = VerlMonitorBridge.from_local_rank(
        monitor,
        local_rank=args.local_rank,
        mode=args.mode,
        rollout_recorder=recorder,
    )

    print(f"[Run] 输出目录: {args.log_dir}")
    print(f"[Run] 平台: {args.platform}")
    print(f"[Run] local_rank={args.local_rank}, mapped_gpu={bridge.gpu_id}")

    monitor.start()
    try:
        for step in range(args.steps):
            simulate_step(step, bridge, args)
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
