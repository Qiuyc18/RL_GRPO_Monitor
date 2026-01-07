import csv
import os
import threading
import time
from collections import deque

from pynvml import (
    NVML_TEMPERATURE_GPU,
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


class GpuMonitor:
    """
    A class to monitor GPU status in a background thread and log it to a CSV file.
    """

    def __init__(
        self,
        output_file_path,
        interval=0.01,
        buffer_seconds=3600,
        write_interval=1.0,
        events_file_path=None,
        enable_metrics=True,
    ):
        """
        Args:
            output_file_path: The path to the CSV file to log the GPU metrics.
            interval: The interval in seconds to monitor the GPU metrics.
            buffer_seconds: The number of seconds to buffer the GPU metrics in memory.
            write_interval: The interval in seconds to write the GPU metrics to the CSV file.
        """
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
        """Creates the log file and writes the header if it doesn't exist."""
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
        """Creates the events log file and writes the header if it doesn't exist."""
        assert self._events_file_path is not None
        if os.path.exists(self._events_file_path):
            return
        with open(self._events_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "event_type"])

    def _monitor_loop(self):
        """The main loop for monitoring and logging GPU stats."""
        if not self._enable_metrics:
            return
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        handles = [nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]

        while self._is_running:
            try:
                timestamp = time.time()
                rows = []
                for gpu_id, handle in enumerate(handles):
                    utilization = nvmlDeviceGetUtilizationRates(handle)
                    memory_info = nvmlDeviceGetMemoryInfo(handle)
                    temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

                    gpu_utilization = utilization.gpu
                    memory_utilization = round(
                        (float(memory_info.used) / float(memory_info.total)) * 100, 2
                    )
                    row = {
                        "timestamp": timestamp,
                        "gpu_id": gpu_id,
                        "gpu_utilization": gpu_utilization,
                        "memory_utilization": memory_utilization,
                        "temperature": temperature,
                    }
                    rows.append(row)

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
            except NVMLError as error:
                print(f"NVMLError: {error}. Stopping monitor.")
                self._is_running = False
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Stopping monitor.")
                self._is_running = False

        nvmlShutdown()

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
        """Starts the background monitoring thread."""
        if not self._is_running and self._enable_metrics:
            self._is_running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def update_interval(self, interval):
        """Updates the sampling interval and resizes buffers safely."""
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
        """Stops the background monitoring thread."""
        if self._is_running:
            print("Stopping GPU monitor...")
            self._is_running = False
            # Wait for the thread to finish
            if self._thread:
                self._thread.join()
            with self._lock:
                self._flush_pending_rows()
                self._flush_pending_events()
            print("GPU monitor stopped.")

    def get_recent_rows(self, window_seconds=None):
        """"""
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
