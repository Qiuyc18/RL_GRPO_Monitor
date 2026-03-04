import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.abspath("./ms-swift"))

# Setup GPU environment based on command line arguments
def setup_gpu_env():
    if "--rollout" in sys.argv:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("[Setup] Configured for ROLLOUT: GPU 0")
    elif "--grpo" in sys.argv:
        # Default: except GPU 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3' 
        print("[Setup] Configured for TRAINING: GPU 1,2,3")

setup_gpu_env()


from swift.llm import RLHFArguments, RolloutArguments, rlhf_main, rollout_main
from swift.trainers.rlhf_trainer import GRPOTrainer
from swift.utils.env import get_dist_setting

from macro import Event
from monitor.monitor import Monitor
from plugin import get_physical_gpu_id, monkey_patch
from plugin_rollout import monkey_patch_rollout
from rollout_stats import RolloutStatsRecorder

# --- Configuration ---
PLATFORM = os.getenv("PLATFORM", "nvidia")
LOG_DIRECTORY = "logs"
GPU_LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "gpu_metrics.csv")
EVENTS_LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "gpu_events.csv")
MONITOR_INTERVAL_MS = 10
UI_REFRESH_INTERVAL_SECONDS = 0.5
BUFFER_SECONDS = 3600
DEFAULT_PLOT_WINDOW_SECONDS = 30
DEFAULT_MAX_GPU_CHOICES = 8
DEFAULT_EVENT_LIMIT = 200
DEFAULT_MONITOR_INTERVAL_SECONDS = MONITOR_INTERVAL_MS / 1000
TB_TIME_ANCHOR_PATH = os.path.join(LOG_DIRECTORY, ".time_anchor")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Configuration ---
MODEL_PATH = "models/Qwen/Qwen3-0.6B"
# MODEL_PATH = "models/Qwen/Qwen2.5-1.5B-Instruct"
VLLM_SERVER_PORT = 8000
VLLM_SERVER_HOST = "127.0.0.1"


def start_rollout_server(log_dir="logs/rollout_experiment"):
    """启动 vLLM rollout 服务器"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rollout_log_dir = log_dir
    os.makedirs(rollout_log_dir, exist_ok=True)
    rollout_metrics_file = os.path.join(rollout_log_dir, "gpu_metrics.csv")
    rollout_events_file = os.path.join(rollout_log_dir, "gpu_events.csv")

    monitor = Monitor(
        platform="nvidia",
        output_file_path=rollout_metrics_file,
        events_file_path=rollout_events_file,
        interval=DEFAULT_MONITOR_INTERVAL_SECONDS,
        enable_metrics=True,
        my_physical_gpu_id=0,
        is_main_for_tb=True,
        write_metrics_csv=True,
        rollout_gpu_ids=(0,),
        tb_time_anchor_path=TB_TIME_ANCHOR_PATH,
    )
    monitor.start()
    monkey_patch_rollout(monitor)

    print(f"[System] Starting vLLM rollout server on GPU 0...")
    print(f"[System] Model: {MODEL_PATH}")
    print(f"[System] Port: {VLLM_SERVER_PORT}")
    print(f"[System] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[System] Rollout Monitor -> {rollout_log_dir}")

    rollout_args = RolloutArguments(
        model=MODEL_PATH,
        port=VLLM_SERVER_PORT,
        torch_dtype="float16",
    )

    try:
        rollout_main(rollout_args)
    finally:
        monitor.stop()


def start_grpo_training(log_dir="logs/grpo_experiment"):
    """启动 GRPO 训练"""
    
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f"[System] Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}, Local World Size: {local_world_size}")
    physical_gpu_id = get_physical_gpu_id(local_rank)

    # Start monitoring
    # 注意：Monitor 会监控所有物理 GPU（不受 CUDA_VISIBLE_DEVICES 影响）
    # 它会通过 NVML 直接访问所有 GPU，所以可以看到 GPU 0（rollout 使用）和其他 GPU（训练使用）
    monitor = None

    output_dir = log_dir

    os.makedirs(log_dir, exist_ok=True)
    metrics_file = os.path.join(log_dir, f"gpu_metrics_rank.csv")
    events_file = os.path.join(log_dir, f"gpu_events_rank_{physical_gpu_id}.csv")
    if os.path.exists(events_file):
        raise FileExistsError(f"[Error] Events file {events_file} already exists, please use a different log directory")

    print(f"[System][Physical GPU {physical_gpu_id}] Starting Monitor -> {events_file}")
    do_enable_metrics = True

    monitor = Monitor(
        platform="nvidia",
        output_file_path=metrics_file,
        events_file_path=events_file,
        interval=DEFAULT_MONITOR_INTERVAL_SECONDS,
        enable_metrics=do_enable_metrics,
        my_physical_gpu_id=physical_gpu_id,
        is_main_for_tb=(rank == 0),
        write_metrics_csv=(rank == 0),
        rollout_gpu_ids=(0,),
        tb_time_anchor_path=TB_TIME_ANCHOR_PATH,
    )
    monitor.start()

    # Rollout 长度统计（独立于 PhaseEvent，输出到单独 CSV）
    rollout_recorder = RolloutStatsRecorder(output_dir=log_dir, gpu_id=physical_gpu_id)

    monkey_patch(monitor, rollout_recorder=rollout_recorder)

    # --- Define DeepSpeed ZeRO-2 configuration ---
    ds_config_dict = {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_clipping": "auto",
        "fp16": {"enabled": "auto"},
        "bf16": {"enabled": "auto"},
    }

    # Build training arguments
    args = RLHFArguments(
        rlhf_type="grpo",
        model=MODEL_PATH,
        reward_funcs=["accuracy"],
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=[VLLM_SERVER_HOST],
        vllm_server_port=[VLLM_SERVER_PORT],
        train_type="full",
        torch_dtype="float16",
        device_map=None,
        dataset=["AI-MO/NuminaMath-TIR#1000"],
        output_dir=output_dir,
        # Training hyperparameters
        load_from_cache_file=True,
        split_dataset_ratio=0,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-6,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        save_total_limit=2,
        logging_steps=1,
        warmup_ratio=0.05,
        dataloader_num_workers=4,
        dataset_num_proc=4,
        report_to=["tensorboard"],
        beta=0.04,
        # GRPO specific parameters
        num_generations=2,  # Group Size
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        # Memory optimization
        max_completion_length=512,
        max_length=1024,
        deepspeed=ds_config_dict,
        # deepspeed="zero2",
        num_iterations=1,
    )

    # Start training
    if rank == 0:
        print(f"[System] Connecting to vLLM server at {VLLM_SERVER_HOST}:{VLLM_SERVER_PORT}...")
        print(f"[System] Starting GRPO training...")

    try:
        rlhf_main(args)
    except Exception as e:
        print(f"[Error] Training Failed: {e}")
        raise e
    finally:
        if monitor:
            print(f"[System] Stopping Monitor on GPU: {local_rank}...")
            monitor.stop()


def main():
    import datetime
    parser = argparse.ArgumentParser(
        description="GRPO Training Script with vLLM Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Terminal 1: Start vLLM server
  python run.py --rollout

  # Terminal 2: Start training
  python run.py --grpo
        """,
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Start vLLM rollout server (run in a separate terminal)",
    )
    parser.add_argument(
        "--grpo",
        action="store_true",
        help="Start GRPO training (requires vLLM server to be running)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=f"logs/grpo_experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory to save GRPO training logs",
    )

    args = parser.parse_args()

    if args.rollout:
        start_rollout_server(log_dir=args.log_dir)
    elif args.grpo:
        start_grpo_training(log_dir=args.log_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
