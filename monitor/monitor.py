import csv
import os
import threading
import time
from collections import deque

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from monitor.monitor_gpu_amd import AMDMonitor
from monitor.monitor_gpu_nvidia import NVIDIAMonitor


class Monitor:
    def __init__(
        self,
        platform,
        output_file_path,
        interval=0.01,
        buffer_seconds=3600,
        write_interval=1.0,
        events_file_path=None,
        enable_metrics=True,
    ):
        if platform == "amd":
            self._collector = AMDMonitor()
        elif platform == "nvidia":
            self._collector = NVIDIAMonitor()
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        self._output_file_path = output_file_path
        self._interval = interval
        self._buffer_seconds = buffer_seconds
        self._write_interval = write_interval
        self._events_file_path = events_file_path
        self._enable_metrics = enable_metrics
        self._is_running = False
        self._thread = None
        self._lock = threading.Lock()
        buffer_maxlen = int(self._buffer_seconds / self._interval)
        self._metrics_buffer = deque(maxlen=buffer_maxlen)
        self._events_buffer = deque(maxlen=buffer_maxlen)

        self._latest_by_gpu = {}
        self._pending_rows = []
        self._pending_events = []
        self._last_flush_time = 0.0
        if self._events_file_path is None:
            if self._output_file_path is None:
                raise ValueError(
                    "events_file_path is required when output_file_path is None."
                )
            self._events_file_path = os.path.join(
                os.path.dirname(self._output_file_path), "gpu_events.csv"
            )
        if self._enable_metrics:
            self._initialize_log_file()
        self._initialize_events_log_file()

    def _initialize_log_file(self):
        with open(self._output_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "gpu_id",
                    "gpu_utilization",
                    "memory_utilization",
                    "temperature",
                ]
            )

    def _initialize_events_log_file(self):
        assert self._events_file_path is not None
        if os.path.exists(self._events_file_path):
            return
        with open(self._events_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "event_type"])

    def _monitor_loop(self):
        if not self._enable_metrics:
            return
        try:
            self._collector.initialize()
        except Exception as error:
            print(f"GPU collector init failed: {error}. Stopping monitor.")
            self._is_running = False
            return

        try:
            while self._is_running:
                timestamp = time.time()
                rows = self._collector.read_metrics()
                for row in rows:
                    row["timestamp"] = timestamp

                with self._lock:
                    self._metrics_buffer.extend(rows)
                    for row in rows:
                        self._latest_by_gpu[row["gpu_id"]] = row
                    cutoff = timestamp - self._buffer_seconds
                    while (
                        self._metrics_buffer
                        and self._metrics_buffer[0]["timestamp"] < cutoff
                    ):
                        self._metrics_buffer.popleft()

                    if self._write_interval is not None:
                        self._pending_rows.extend(rows)
                        if timestamp - self._last_flush_time >= self._write_interval:
                            self._flush_pending_rows()
                            self._flush_pending_events()

                time.sleep(self._interval)
        except Exception as error:
            print(f"An unexpected error occurred: {error}. Stopping monitor.")
            self._is_running = False
        finally:
            self._collector.shutdown()

    def _flush_pending_rows(self):
        if not self._pending_rows:
            return
        with open(self._output_file_path, "a", newline="") as file:
            writer = csv.writer(file)
            for row in self._pending_rows:
                writer.writerow(
                    [
                        row["timestamp"],
                        row["gpu_id"],
                        row["gpu_utilization"],
                        row["memory_utilization"],
                        row["temperature"],
                    ]
                )
        self._pending_rows.clear()
        self._last_flush_time = time.time()

    def _flush_pending_events(self):
        if not self._pending_events:
            return
        assert self._events_file_path is not None
        with open(self._events_file_path, "a", newline="") as file:
            writer = csv.writer(file)
            for row in self._pending_events:
                writer.writerow([row["timestamp"], row["event_type"]])
        self._pending_events.clear()

    def start(self):
        if not self._is_running and self._enable_metrics:
            self._is_running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def update_interval(self, interval):
        if interval is None or interval <= 0:
            return
        with self._lock:
            self._interval = interval
            buffer_maxlen = int(self._buffer_seconds / self._interval)
            if buffer_maxlen <= 0:
                buffer_maxlen = 1
            self._metrics_buffer = deque(self._metrics_buffer, maxlen=buffer_maxlen)
            self._events_buffer = deque(self._events_buffer, maxlen=buffer_maxlen)

    def stop(self):
        if self._is_running:
            print("Stopping GPU monitor...")
            self._is_running = False
            if self._thread:
                self._thread.join()
            with self._lock:
                self._flush_pending_rows()
                self._flush_pending_events()
            print("GPU monitor stopped.")

    def get_recent_rows(self, window_seconds=None):
        if window_seconds is None:
            window_seconds = self._buffer_seconds
        cutoff = time.time() - window_seconds
        with self._lock:
            return [
                row.copy() for row in self._metrics_buffer if row["timestamp"] >= cutoff
            ]

    def get_latest_snapshot(self):
        with self._lock:
            return [row.copy() for row in self._latest_by_gpu.values()]

    def add_event(self, event_type):
        ts = time.time()
        with self._lock:
            self._events_buffer.append(
                {
                    "timestamp": ts,
                    "event_type": event_type,
                }
            )
            self._pending_events.append(
                {
                    "timestamp": ts,
                    "event_type": event_type,
                }
            )
        self._flush_pending_events()

    def get_recent_events(self, window_seconds=None, limit=None):
        if window_seconds is None:
            window_seconds = self._buffer_seconds
        cutoff = time.time() - window_seconds
        with self._lock:
            rows = [
                row.copy() for row in self._events_buffer if row["timestamp"] >= cutoff
            ]
        if limit is not None and limit > 0:
            return rows[-int(limit) :]
        return rows

