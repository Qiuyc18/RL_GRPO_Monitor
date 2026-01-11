from monitor.monitor import Monitor
import os
import time

def _run_smoke_test():
    platform = os.environ.get("GPU_PLATFORM", "amd")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.environ.get(
        "GPU_MONITOR_OUTPUT",
        os.path.join(output_dir, f"gpu_metrics_smoke_{int(time.time())}.csv"),
    )

    monitor = Monitor(
        platform=platform,
        output_file_path=output_path,
        interval=0.2,
        buffer_seconds=10,
        write_interval=1.0,
    )
    monitor.start()
    monitor.add_event("monitor_start")
    time.sleep(5)
    monitor.add_event("monitor_stop")
    monitor.stop()

    print("latest_snapshot:", monitor.get_latest_snapshot())
    print("recent_rows:", len(monitor.get_recent_rows()))
    print("recent_events:", monitor.get_recent_events(limit=5))


if __name__ == "__main__":
    _run_smoke_test()
