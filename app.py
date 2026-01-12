import atexit
import csv
import logging
import os
import dotenv

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from monitor import Monitor

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

# --- Setup ---
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Instantiate and start the GPU monitor in the background.
monitor = Monitor(
    platform=PLATFORM,
    output_file_path=GPU_LOG_FILE_PATH,
    interval=MONITOR_INTERVAL_MS / 1000,
    buffer_seconds=BUFFER_SECONDS,
    write_interval=1.0,
    events_file_path=EVENTS_LOG_FILE_PATH,
)
monitor.start()

# Register the monitor's stop method to be called when the application exits.
atexit.register(monitor.stop)


def _tail_lines(path, max_lines):
    if max_lines <= 0:
        return []
    with open(path, "rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        block = 4096
        data = b""
        while size > 0 and data.count(b"\n") <= max_lines:
            step = min(block, size)
            file.seek(size - step)
            data = file.read(step) + data
            size -= step
    return data.splitlines()[-max_lines:]


def _read_recent_events(limit):
    if not os.path.exists(EVENTS_LOG_FILE_PATH):
        return pd.DataFrame(columns=["timestamp", "event_type"])  # type: ignore
    lines = _tail_lines(EVENTS_LOG_FILE_PATH, limit + 1)
    if not lines:
        return pd.DataFrame(columns=["timestamp", "event_type"])  # type: ignore
    if lines[0].startswith(b"timestamp"):
        lines = lines[1:]
    rows = []
    for row in csv.reader([line.decode("utf-8") for line in lines]):
        if len(row) >= 2:
            try:
                ts = float(row[0])
            except ValueError:
                continue
            rows.append({"timestamp": ts, "event_type": row[1]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.round("ms")
    return df


def _build_usage_figure(df, y_key, title, events_df):
    fig = go.Figure()
    if not df.empty:
        for gpu_id in sorted(df["gpu_id"].unique()):
            gpu_df = df[df["gpu_id"] == gpu_id]
            fig.add_trace(
                go.Scatter(
                    x=gpu_df["timestamp"],
                    y=gpu_df[y_key],
                    mode="lines",
                    name=f"GPU {gpu_id}",
                )
            )

    if events_df is not None and not events_df.empty:
        palette = [
            "#e74c3c",
            "#3498db",
            "#2ecc71",
            "#f39c12",
            "#9b59b6",
            "#1abc9c",
            "#e67e22",
            "#34495e",
        ]
        event_types = sorted(events_df["event_type"].unique())
        color_map = {
            event_type: palette[i % len(palette)]
            for i, event_type in enumerate(event_types)
        }
        for event_type in event_types:
            event_rows = events_df[events_df["event_type"] == event_type]
            x_vals = []
            y_vals = []
            for ts in event_rows["timestamp"]:
                x_vals.extend([ts, ts, None])
                y_vals.extend([0, 100, None])
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=color_map[event_type], dash="dot", width=1),
                    name=event_type,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="占用率（%）",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h"),
        width=1200,
        height=300,
    )
    return fig


def _empty_payload():
    return (
        pd.DataFrame(
            columns=[  # type: ignore
                "timestamp",
                "gpu_id",
                "gpu_utilization",
                "memory_utilization",
                "temperature",
            ]
        ),
        pd.DataFrame(columns=["timestamp", "gpu_id", "gpu_utilization"]),  # type: ignore
        pd.DataFrame(columns=["timestamp", "gpu_id", "memory_utilization"]),  # type: ignore
    )


def _normalize_gpu_ids(selected_gpu_ids):
    if selected_gpu_ids is None:
        return None
    if not selected_gpu_ids:
        return set()
    return {str(gpu_id) for gpu_id in selected_gpu_ids}


def _get_gpu_choices():
    gpu_list = monitor.get_gpu_choices()
    if not gpu_list:
        return [str(i) for i in range(DEFAULT_MAX_GPU_CHOICES)]
    return gpu_list


def read_monitoring_data(window_seconds, selected_gpu_ids, paused, last_state):
    """
    Reads the GPU log and prepares it for the Gradio dashboard.
    """
    if paused and last_state is not None:
        return (*last_state, last_state)
    try:
        rows = monitor.get_recent_rows(window_seconds)
        if not rows:
            empty_payload = _empty_payload()
            return (*empty_payload, empty_payload)

        rows_by_gpu = {}
        for row in rows:
            rows_by_gpu.setdefault(str(row["gpu_id"]), []).append(row)

        selected_set = _normalize_gpu_ids(selected_gpu_ids)
        if selected_set is None:
            gpu_ids = sorted(rows_by_gpu.keys())
        else:
            gpu_ids = [
                gpu_id
                for gpu_id in sorted(rows_by_gpu.keys())
                if gpu_id in selected_set
            ]

        sampled_rows = []
        for gpu_id in gpu_ids:
            sampled_rows.extend(rows_by_gpu[gpu_id])

        if not sampled_rows:
            empty_payload = _empty_payload()
            return (*empty_payload, empty_payload)

        df = pd.DataFrame(sampled_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.round("ms")
        df["gpu_id"] = df["gpu_id"].astype(str)

        latest_rows = monitor.get_latest_snapshot()
        if latest_rows:
            summary_df = pd.DataFrame(latest_rows).sort_values("gpu_id")
            summary_df["timestamp"] = pd.to_datetime(
                summary_df["timestamp"], unit="s"
            ).dt.round("ms")
            summary_df["gpu_id"] = summary_df["gpu_id"].astype(str)
            if gpu_ids:
                summary_df = summary_df[summary_df["gpu_id"].isin(gpu_ids)]
        else:
            summary_df = pd.DataFrame(
                columns=[  # type: ignore
                    "timestamp",
                    "gpu_id",
                    "gpu_utilization",
                    "memory_utilization",
                    "temperature",
                ]
            )

        events_df = _read_recent_events(DEFAULT_EVENT_LIMIT)
        if not events_df.empty and not df.empty:
            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()
            events_df = events_df[
                (events_df["timestamp"] >= min_ts) & (events_df["timestamp"] <= max_ts)
            ]

        gpu_fig = _build_usage_figure(df, "gpu_utilization", "GPU 占用率", events_df)
        mem_fig = _build_usage_figure(df, "memory_utilization", "显存占用率", events_df)
        payload = (summary_df, gpu_fig, mem_fig)
        return (*payload, payload)

    except Exception:
        if last_state is not None:
            return (*last_state, last_state)
        empty_payload = _empty_payload()
        return (*empty_payload, empty_payload)


with gr.Blocks() as demo:
    gr.Markdown("## GPU 实时监控")

    with gr.Row():
        with gr.Column(scale=1, min_width=240):
            window_seconds = gr.Slider(
                minimum=5,
                maximum=BUFFER_SECONDS,
                value=DEFAULT_PLOT_WINDOW_SECONDS,
                step=1,
                label="最大绘图窗口（秒）",
            )
            refresh_interval = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=UI_REFRESH_INTERVAL_SECONDS,
                step=0.1,
                label="刷新间隔（秒）",
            )
            monitor_interval = gr.Slider(
                minimum=0.05,
                maximum=2.0,
                value=DEFAULT_MONITOR_INTERVAL_SECONDS,
                step=0.05,
                label="采样间隔（秒）",
            )
            pause_state = gr.State(value=False)
            pause_button = gr.Button(value="暂停刷新")
            gpu_selector = gr.CheckboxGroup(
                choices=_get_gpu_choices(),
                value=_get_gpu_choices(),
                label="GPU 列表",
            )

        with gr.Column(scale=4):
            summary_table = gr.Dataframe(
                headers=[
                    "时间",
                    "GPU ID",
                    "GPU 占用率",
                    "显存占用率",
                    "温度",
                ],
                datatype=["date", "str", "number", "number", "number"],
                label="当前 GPU 指标（每 GPU）",
                interactive=False,
            )

            gpu_history_plot = gr.Plot(label="GPU 占用率历史")
            mem_history_plot = gr.Plot(label="显存占用率历史")

    last_state = gr.State(value=None)

    # events_table = gr.Dataframe(
    #     headers=["timestamp", "event_type"],
    #     datatype=["date", "str"],
    #     label="事件（最近N条）",
    #     interactive=False,
    # )

    # Use gr.Timer for periodic refresh
    timer = gr.Timer(value=UI_REFRESH_INTERVAL_SECONDS)
    timer.tick(
        fn=read_monitoring_data,
        inputs=[window_seconds, gpu_selector, pause_state, last_state],
        outputs=[
            summary_table,
            gpu_history_plot,
            mem_history_plot,
            #  events_table
            last_state,
        ],
    )
    pause_button.click(
        fn=lambda paused: (
            not paused,
            gr.update(value="恢复刷新" if not paused else "暂停刷新"),
        ),
        inputs=pause_state,
        outputs=[pause_state, pause_button],
    )
    refresh_interval.change(
        fn=lambda interval: interval,
        inputs=refresh_interval,
        outputs=timer,
    )
    monitor_interval.change(
        fn=lambda interval: monitor.update_interval(interval),
        inputs=monitor_interval,
        outputs=[],
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
