import functools
import os
import sys

sys.path.insert(0, os.path.abspath("./ms-swift"))

from swift.llm import RLHFArguments, rlhf_main
from swift.trainers.rlhf_trainer import GRPOTrainer
from swift.utils.env import get_dist_setting

from macro import Event
from monitor.monitor import Monitor

# --- Configuration ---
PLATFORM = os.getenv("PLATFORM", "nvidia")
LOG_DIRECTORY = "logs"
GPU_LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "gpu_metrics.csv")
EVENTS_LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "gpu_events.csv")
MONITOR_INTERVAL_MS = 100
UI_REFRESH_INTERVAL_SECONDS = 0.5
BUFFER_SECONDS = 3600
DEFAULT_PLOT_WINDOW_SECONDS = 30
DEFAULT_MAX_GPU_CHOICES = 8
DEFAULT_EVENT_LIMIT = 200
DEFAULT_MONITOR_INTERVAL_SECONDS = MONITOR_INTERVAL_MS / 1000


def monkey_patch(monitor: Monitor | None):
    """Wrap GRPOTrainer to inject monitor."""
    target_method = "_generate_and_score_completions"

    original_method = getattr(GRPOTrainer, target_method)
    setattr(GRPOTrainer, f"_original_{target_method}", original_method)

    @functools.wraps(getattr(GRPOTrainer, f"_original_{target_method}"))
    def patched_generate(self, *args, **kwargs):
        # [EVENT: 生成开始]
        if monitor:
            monitor.add_event(Event.INFERENCE_START)

        result = getattr(GRPOTrainer, f"_original_{target_method}")(
            self, *args, **kwargs
        )

        # [EVENT: 生成结束]
        if monitor:
            monitor.add_event(Event.INFERENCE_END)
            monitor.add_event(Event.TRAINING_START)

        return result

    setattr(GRPOTrainer, target_method, patched_generate)
    print(f"[MonkeyPatch] Successfully patched GRPOTrainer.{target_method}")

    if not hasattr(GRPOTrainer, "_original_training_step"):
        GRPOTrainer._original_training_step = GRPOTrainer.training_step  # pyright: ignore[reportAttributeAccessIssue]

    @functools.wraps(GRPOTrainer._original_training_step)  # pyright: ignore[reportAttributeAccessIssue]
    def patched_step(self, *args, **kwargs):
        res = GRPOTrainer._original_training_step(self, *args, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]
        if monitor:
            monitor.add_event(Event.TRAINING_END)
        return res

    GRPOTrainer.training_step = patched_step


def main():
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    is_master = rank == 0  # Use GPU:0 for monitoring

    # Start monitoring
    monitor = None
    if is_master:
        output_dir = "logs/grpo_experiment_v1"
        os.makedirs(output_dir, exist_ok=True)

        print(f"[System] Starting Custom GPU Monitor on GPU: {local_rank}...")
        monitor = Monitor(
            platform="nvidia",
            output_file_path=os.path.join(output_dir, "gpu_metrics.csv"),
            events_file_path=os.path.join(output_dir, "gpu_events.csv"),
            interval=DEFAULT_MONITOR_INTERVAL_SECONDS,
            enable_metrics=True,
        )
        monitor.start()

    if is_master:
        monkey_patch(monitor)
    else:
        monkey_patch(None)

    # --- Define DeepSpeed ZeRO-2 configuration ---
    ds_config_dict = {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_clipping": "auto",
        "fp16": {"enabled": "auto"},
        "bf16": {"enabled": "auto"},
    }

    # Build arguments
    args = RLHFArguments(
        rlhf_type="grpo",
        model="models/Qwen/Qwen2.5-1.5B-Instruct",
        reward_funcs="accuracy",
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="127.0.0.1",
        vllm_server_port=8000,
        train_type="full",
        torch_dtype="bfloat16",
        dataset="AI-MO/NuminaMath-TIR#1000",
        output_dir="output/grpo_experiment_v1",
        # Training hyperparameters
        load_from_cache_file=True,
        split_dataset_ratio=0,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=1e-6,
        gradient_accumulation_steps=2,
        save_total_limit=2,
        logging_steps=1,
        warmup_ratio=0.05,
        dataloader_num_workers=4,
        dataset_num_proc=4,
        report_to="tensorboard",
        beta=0.04,
        # GRPO specific parameters
        num_generations=8,  # Group Size
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        # Memory optimization
        max_completion_length=2048,
        deepspeed=ds_config_dict,
        num_iterations=1,
    )

    # Start training
    try:
        rlhf_main(args)
    except Exception as e:
        print(f"[Error] Training Failed: {e}")
        raise e
    finally:
        if monitor:
            print(f"[System] Stopping Monitor on GPU: {local_rank}...")
            monitor.stop()


if __name__ == "__main__":
    main()
