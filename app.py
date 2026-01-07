import atexit
import os

import gradio as gr
import pandas as pd

from monitor_gpu import GpuMonitor

# --- Configuration ---
LOG_DIRECTORY = "data"
GPU_LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "gpu_metrics.csv")
MONITOR_INTERVAL_MS = 100
UI_REFRESH_INTERVAL_SECONDS = 0.2
BUFFER_SECONDS = 60
DEFAULT_PLOT_WINDOW_POINTS = 300
DEFAULT_MAX_GPUS = 8

# --- Setup ---
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Instantiate and start the GPU monitor in the background.
monitor = GpuMonitor(
    output_file_path=GPU_LOG_FILE_PATH,
    interval=MONITOR_INTERVAL_MS / 1000,
    buffer_seconds=BUFFER_SECONDS,
    write_interval=1.0,
)
monitor.start()

# Register the monitor's stop method to be called when the application exits.
atexit.register(monitor.stop)


def read_monitoring_data(window_points, max_gpus):
    """
    Reads the GPU log data and prepares it for the Gradio dashboard.
    """
    try:
        rows = monitor.get_recent_rows(BUFFER_SECONDS)
        if not rows:
            return (
                pd.DataFrame(
                    columns=[
                        "timestamp",
                        "gpu_id",
                        "gpu_utilization",
                        "memory_utilization",
                        "temperature",
                    ]
                ),
                pd.DataFrame(columns=["timestamp", "gpu_id", "gpu_utilization"]),
                pd.DataFrame(columns=["timestamp", "gpu_id", "memory_utilization"]),
            )

        rows_by_gpu = {}
        for row in rows:
            rows_by_gpu.setdefault(row["gpu_id"], []).append(row)

        gpu_ids = sorted(rows_by_gpu.keys())
        if max_gpus is not None and max_gpus > 0:
            gpu_ids = gpu_ids[: int(max_gpus)]

        sampled_rows = []
        for gpu_id in gpu_ids:
            gpu_rows = rows_by_gpu[gpu_id]
            if window_points is not None and window_points > 0:
                gpu_rows = gpu_rows[-int(window_points) :]
            sampled_rows.extend(gpu_rows)

        if not sampled_rows:
            return (
                pd.DataFrame(
                    columns=[
                        "timestamp",
                        "gpu_id",
                        "gpu_utilization",
                        "memory_utilization",
                        "temperature",
                    ]
                ),
                pd.DataFrame(columns=["timestamp", "gpu_id", "gpu_utilization"]),
                pd.DataFrame(columns=["timestamp", "gpu_id", "memory_utilization"]),
            )

        df = pd.DataFrame(sampled_rows)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="s", format="%Y-%m-%d %H:%M:%S.%2f"
        )
        df["gpu_id"] = df["gpu_id"].astype(str)

        latest_rows = monitor.get_latest_snapshot()
        if latest_rows:
            summary_df = (
                pd.DataFrame(latest_rows)
                .sort_values("gpu_id")
                .query("gpu_id in @gpu_ids")
            )
            summary_df["timestamp"] = pd.to_datetime(summary_df["timestamp"], unit="s")
            summary_df["gpu_id"] = summary_df["gpu_id"].astype(str)
        else:
            summary_df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "gpu_id",
                    "gpu_utilization",
                    "memory_utilization",
                    "temperature",
                ]
            )

        gpu_df = df[["timestamp", "gpu_id", "gpu_utilization"]]
        mem_df = df[["timestamp", "gpu_id", "memory_utilization"]]

        return summary_df, gpu_df, mem_df

    except Exception:
        return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("## GPU 实时监控")

    with gr.Row():
        window_points = gr.Slider(
            minimum=50,
            maximum=5000,
            value=DEFAULT_PLOT_WINDOW_POINTS,
            step=50,
            label="绘图窗口（每 GPU 点数）",
        )
        max_gpus = gr.Slider(
            minimum=1,
            maximum=16,
            value=DEFAULT_MAX_GPUS,
            step=1,
            label="最大显示 GPU 数",
        )
        refresh_interval = gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=UI_REFRESH_INTERVAL_SECONDS,
            step=0.1,
            label="刷新间隔（秒）",
        )

    summary_table = gr.Dataframe(
        headers=[
            "时间",
            "GPU ID",
            "GPU 使用率",
            "显存使用率",
            "温度",
        ],
        datatype=["datetime", "str", "number", "number", "number"],
        label="当前 GPU 指标（每 GPU）",
        interactive=False,
    )

    gpu_history_plot = gr.LinePlot(
        x="timestamp",
        y="gpu_utilization",
        color="gpu_id",
        title="GPU 使用率历史",
        y_lim=[0, 100],
        tooltip=["timestamp", "gpu_id", "gpu_utilization"],
    )
    mem_history_plot = gr.LinePlot(
        x="timestamp",
        y="memory_utilization",
        color="gpu_id",
        title="显存使用率历史",
        y_lim=[0, 100],
        tooltip=["timestamp", "gpu_id", "memory_utilization"],
    )

    # Use gr.Timer for periodic refresh
    timer = gr.Timer(value=UI_REFRESH_INTERVAL_SECONDS)
    timer.tick(
        fn=read_monitoring_data,
        inputs=[window_points, max_gpus],
        outputs=[summary_table, gpu_history_plot, mem_history_plot],
    )
    refresh_interval.change(
        fn=lambda interval: interval,
        inputs=refresh_interval,
        outputs=timer,
    )

if __name__ == "__main__":
    # Ensure localhost bypasses any proxy that could break Gradio's startup checks.
    existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
    localhost_no_proxy = "127.0.0.1,localhost"
    if existing_no_proxy:
        if "127.0.0.1" not in existing_no_proxy or "localhost" not in existing_no_proxy:
            merged_no_proxy = f"{existing_no_proxy},{localhost_no_proxy}"
        else:
            merged_no_proxy = existing_no_proxy
    else:
        merged_no_proxy = localhost_no_proxy
    os.environ["NO_PROXY"] = merged_no_proxy
    os.environ["no_proxy"] = merged_no_proxy

    demo.launch(debug=True, server_name="127.0.0.1", server_port=7860)
