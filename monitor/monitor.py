import csv
import os
import threading
import time
from collections import deque, defaultdict
import re

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("[Monitor] Warning: torch.utils.tensorboard not found. TensorBoard logging disabled.")
    TENSORBOARD_AVAILABLE = False

if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dotenv
dotenv.load_dotenv()

PLATFORM = os.getenv("GPU_PLATFORM") or os.getenv("PLATFORM", "nvidia")
if PLATFORM == "amd":
    from monitor.monitor_gpu_amd import AMDMonitor
elif PLATFORM == "nvidia":
    from monitor.monitor_gpu_nvidia import NVIDIAMonitor
else:
    raise ValueError(f"Unsupported platform: {PLATFORM}")


LOG_CSV_HEADERS = ["timestamp", "gpu_id", "gpu_utilization", "memory_utilization", "temperature"]
EVENT_CSV_HEADERS = ["timestamp", "gpu_id", "step", "event_type", "mode", "role"]

ROLLOUT_STATE_MAP = {
    "IDLE": 0,
    "ROLLOUT_PHASE": 1,
    "PREPARE": 2,
}
TRAINER_STATE_MAP = {
    "IDLE": 0,
    "ROLLOUT_PHASE": 1,
    "BATCH_PREP": 2,
    "REWARD": 3,
    "FORWARD": 4,
    "BACKWARD": 5,
}

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
        tb_log_dir=None,
        my_physical_gpu_id=None,
        is_main_for_tb=False,
        write_metrics_csv=True,
        rollout_gpu_ids=(0,),
        tb_time_anchor_path=None,
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
        self._my_physical_gpu_id = my_physical_gpu_id
        self._is_main_for_tb = is_main_for_tb
        self._write_metrics_csv = write_metrics_csv
        self._rollout_gpu_ids = tuple(rollout_gpu_ids) if rollout_gpu_ids else ()
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

        # --- TensorBoard ---
        self._tb_writer = None
        self._start_time = time.time()
        self._tb_time_anchor = self._start_time
        if tb_time_anchor_path:
            anchor_dir = os.path.dirname(tb_time_anchor_path)
            if anchor_dir:
                os.makedirs(anchor_dir, exist_ok=True)
            try:
                with open(tb_time_anchor_path) as f:
                    self._tb_time_anchor = float(f.read().strip())
            except (ValueError, OSError, FileNotFoundError):
                try:
                    with open(tb_time_anchor_path, "x") as f:
                        f.write(str(self._start_time))
                    self._tb_time_anchor = self._start_time
                except FileExistsError:
                    with open(tb_time_anchor_path) as f:
                        self._tb_time_anchor = float(f.read().strip())
                except OSError:
                    pass
        self._rollout_state = defaultdict(int)
        self._trainer_state = defaultdict(int)
        
        if TENSORBOARD_AVAILABLE:
            if tb_log_dir:
                self._tb_writer = SummaryWriter(log_dir=tb_log_dir)
            elif output_file_path:
                log_dir = os.path.join(os.path.dirname(output_file_path), "runs")
                self._tb_writer = SummaryWriter(log_dir=log_dir)
                print(f"[Monitor] TensorBoard logging enabled at: {log_dir}")

        if self._events_file_path is None:
            if self._output_file_path is None:
                raise ValueError(
                    "events_file_path is required when output_file_path is None."
                )
            self._events_file_path = os.path.join(
                os.path.dirname(self._output_file_path), "gpu_events.csv"
            )
            
        if self._enable_metrics and self._write_metrics_csv:
            self._initialize_log_file()
        self._initialize_events_log_file()

    def _initialize_log_file(self):
        os.makedirs(os.path.dirname(self._output_file_path), exist_ok=True)
        with open(self._output_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(LOG_CSV_HEADERS)

    def _initialize_events_log_file(self):
        assert self._events_file_path is not None
        os.makedirs(os.path.dirname(self._events_file_path), exist_ok=True)
        if os.path.exists(self._events_file_path):
            return
        with open(self._events_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(EVENT_CSV_HEADERS)

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
                relative_step = int((timestamp - self._tb_time_anchor) * 100)

                rows = self._collector.read_metrics()
                
                # --- TensorBoard Metric Logging ---
                if self._tb_writer:
                    for row in rows:
                        gid = row["gpu_id"]
                        if self._is_main_for_tb:
                            self._tb_writer.add_scalar(f"System/GPU_{gid}/Memory_Util", row["memory_utilization"], relative_step)
                            self._tb_writer.add_scalar(f"System/GPU_{gid}/GPU_Util", row["gpu_utilization"], relative_step)
                        if (
                            gid in self._rollout_gpu_ids
                            and self._my_physical_gpu_id is not None
                            and gid == self._my_physical_gpu_id
                        ):
                            self._tb_writer.add_scalar(f"Rollout/GPU_{gid}_Phase", self._rollout_state[gid], relative_step)
                        write_trainer = (
                            (self._my_physical_gpu_id is None) or (gid == self._my_physical_gpu_id)
                        )
                        if write_trainer and gid not in self._rollout_gpu_ids:
                            self._tb_writer.add_scalar(f"Trainer/GPU_{gid}_Phase", self._trainer_state[gid], relative_step)

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
                        if self._write_metrics_csv:
                            self._pending_rows.extend(rows)
                        if timestamp - self._last_flush_time >= self._write_interval:
                            if self._write_metrics_csv:
                                self._flush_pending_rows()
                            self._flush_pending_events()

                time.sleep(self._interval)
        except Exception as error:
            import traceback
            traceback.print_exc()
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
                writer.writerow(
                    [
                        row["timestamp"],
                        row["gpu_id"],
                        row["step"],
                        row["event_type"],
                        row.get("mode", ""),
                        row.get("role", ""),
                    ]
                )
        self._pending_events.clear()

    def _update_gpu_state(self, gpu_id, event_str, role):
        """按 role 更新 Rollout 或 Trainer 状态。role 为 'rollout' 或 'trainer'。"""
        event_name = str(event_str).upper()
        new_state = 0
        state_map = ROLLOUT_STATE_MAP if role == "rollout" else TRAINER_STATE_MAP
        if "START" in event_name:
            for key, val in state_map.items():
                if key in event_name:
                    new_state = val
                    break
        elif "END" in event_name:
            new_state = 0
        if role == "rollout":
            self._rollout_state[gpu_id] = new_state
        else:
            self._trainer_state[gpu_id] = new_state

    def add_event(self, event_type, step=None, gpu_id=None, mode=None, role=None):
        ts = time.time()
        if gpu_id is not None and role is not None:
            self._update_gpu_state(gpu_id, event_type, role)

        with self._lock:
            event_data = {
                "timestamp": ts,
                "gpu_id": gpu_id,
                "step": step,
                "event_type": str(event_type),
                "mode": mode,
                "role": role,
            }
            self._events_buffer.append(event_data)
            self._pending_events.append(event_data)
        self._flush_pending_events()

    def start(self):
        if not self._is_running and self._enable_metrics:
            self._is_running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def stop(self):
        if self._is_running:
            print("Stopping GPU monitor...")
            self._is_running = False
            if self._thread:
                self._thread.join()
            with self._lock:
                if self._write_metrics_csv:
                    self._flush_pending_rows()
                self._flush_pending_events()
            
            # Close TensorBoard writer
            if self._tb_writer:
                self._tb_writer.close()
                
            print("GPU monitor stopped.")
            
    # ... (Keep update_interval, get_recent_rows, etc. unchanged) ...
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

    def get_recent_rows(self, window_seconds=None):
        if window_seconds is None:
            window_seconds = self._buffer_seconds
        cutoff = time.time() - window_seconds
        with self._lock:
            return [
                row.copy() for row in self._metrics_buffer if row["timestamp"] >= cutoff
            ]

    def get_gpu_choices(self):
        with self._lock:
            return self._collector.get_gpu_choices()

    def get_latest_snapshot(self):
        with self._lock:
            return [row.copy() for _, row in sorted(self._latest_by_gpu.items())]

    def get_recent_events(self, window_seconds=None, limit=None):
        if window_seconds is None:
            window_seconds = self._buffer_seconds
        cutoff = time.time() - window_seconds
        with self._lock:
            rows = [
                row.copy() for row in self._events_buffer if row["timestamp"] >= cutoff
            ]
        if limit is not None:
            return rows[-limit:]
        return rows
