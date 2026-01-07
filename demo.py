#!/usr/bin/env python3
import argparse
import os

from swift.llm import (  # pyright: ignore[reportMissingImports]
    InferArguments,
    InferRequest,
)
from swift.llm.infer import SwiftInfer  # pyright: ignore[reportMissingImports]

from monitor_gpu import GpuMonitor

LOG_DIRECTORY = "logs"
EVENTS_LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "gpu_events.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ms-swift inference demo (generate-only stage)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID or local path, e.g. /model/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--prompt",
        default="用一句话解释什么是 GRPO。",
        help="Prompt to generate from.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to transformers (e.g., auto, cpu).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens to stdout during generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    event_monitor = GpuMonitor(
        output_file_path=os.path.join(LOG_DIRECTORY, "gpu_metrics.csv"),
        events_file_path=EVENTS_LOG_FILE_PATH,
        enable_metrics=False,
    )
    event_monitor.add_event("demo_start")
    infer_args = InferArguments(
        model=args.model,
        infer_backend="pt",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        stream=args.stream,
    )
    infer = SwiftInfer(infer_args)
    event_monitor.add_event("model_loaded")
    request_config = infer_args.get_request_config()
    request = InferRequest(messages=[{"role": "user", "content": args.prompt}])
    event_monitor.add_event("generate_start")
    infer.infer_single(request, request_config=request_config)  # pyright: ignore[reportArgumentType]
    event_monitor.add_event("generate_end")


if __name__ == "__main__":
    main()
