import csv
import os
import time

import numpy as np

SAMPLES_CSV_HEADERS = [
    "step_id",
    "prompt_idx",
    "completion_idx",
    "completion_length",
    "mode",
    "gpu_id",
    "timestamp",
]

GROUPS_CSV_HEADERS = [
    "step_id",
    "prompt_idx",
    "group_size",
    "len_mean",
    "len_max",
    "len_min",
    "len_std",
    "len_p90",
    "tail_ratio",
    "tail_share",
    "mode",
    "gpu_id",
    "timestamp",
]


class RolloutStatsRecorder:
    """记录每次 rollout 的 completion 长度明细和 group 聚合统计。"""

    def __init__(self, output_dir: str, gpu_id: int | None = None):
        os.makedirs(output_dir, exist_ok=True)
        self.gpu_id = gpu_id

        suffix = f"_gpu{gpu_id}" if gpu_id is not None else ""
        self.samples_path = os.path.join(output_dir, f"rollout_samples{suffix}.csv")
        self.groups_path = os.path.join(output_dir, f"rollout_groups{suffix}.csv")

        self._init_csv(self.samples_path, SAMPLES_CSV_HEADERS)
        self._init_csv(self.groups_path, GROUPS_CSV_HEADERS)
        print(
            f"[RolloutStats] Recorder initialized -> {self.samples_path}, {self.groups_path}"
        )

    @staticmethod
    def _init_csv(path: str, headers: list[str]):
        with open(path, "w", newline="") as file:
            csv.writer(file).writerow(headers)

    def record(
        self,
        step_id: int,
        lengths: list[int],
        num_generations: int,
        mode: str = "external",
    ):
        if num_generations <= 0:
            num_generations = 1

        ts = time.time()
        gpu_id = self.gpu_id

        samples_rows = []
        for index, length in enumerate(lengths):
            prompt_idx = index // num_generations
            completion_idx = index % num_generations
            samples_rows.append(
                [step_id, prompt_idx, completion_idx, length, mode, gpu_id, ts]
            )
        with open(self.samples_path, "a", newline="") as file:
            csv.writer(file).writerows(samples_rows)

        num_prompts = len(lengths) // num_generations if num_generations > 0 else 0
        groups_rows = []
        for prompt_idx in range(num_prompts):
            group = lengths[
                prompt_idx * num_generations : (prompt_idx + 1) * num_generations
            ]
            if not group:
                continue
            groups_rows.append(
                self._build_group_row(
                    step_id=step_id,
                    prompt_idx=prompt_idx,
                    group=group,
                    mode=mode,
                    gpu_id=gpu_id,
                    timestamp=ts,
                )
            )

        remainder = (
            len(lengths) % num_generations if num_generations > 0 else len(lengths)
        )
        if remainder > 0:
            group = lengths[num_prompts * num_generations :]
            groups_rows.append(
                self._build_group_row(
                    step_id=step_id,
                    prompt_idx=num_prompts,
                    group=group,
                    mode=mode,
                    gpu_id=gpu_id,
                    timestamp=ts,
                )
            )

        with open(self.groups_path, "a", newline="") as file:
            csv.writer(file).writerows(groups_rows)

    @staticmethod
    def _build_group_row(
        *,
        step_id: int,
        prompt_idx: int,
        group: list[int],
        mode: str,
        gpu_id: int | None,
        timestamp: float,
    ):
        arr = np.array(group, dtype=float)
        mean_val = float(np.mean(arr))
        max_val = float(np.max(arr))
        min_val = float(np.min(arr))
        std_val = float(np.std(arr))
        p90_val = float(np.percentile(arr, 90))
        total = float(np.sum(arr))
        tail_ratio = max_val / mean_val if mean_val > 0 else 0.0
        tail_share = max_val / total if total > 0 else 0.0

        return [
            step_id,
            prompt_idx,
            len(group),
            f"{mean_val:.1f}",
            f"{max_val:.0f}",
            f"{min_val:.0f}",
            f"{std_val:.2f}",
            f"{p90_val:.0f}",
            f"{tail_ratio:.3f}",
            f"{tail_share:.4f}",
            mode,
            gpu_id,
            timestamp,
        ]
