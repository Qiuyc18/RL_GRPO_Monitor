"""
轻量 Rollout 长度统计模块 —— 独立于 PhaseEvent 体系。

负责把每次 rollout 生成的 completion 长度写到两个 CSV：
  - rollout_samples.csv  : 每条 completion 一行
  - rollout_groups.csv   : 每个 prompt group 一行（聚合统计）

与现有 Monitor / PhaseEvent 解耦，通过 step_id 关联。
"""

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
        print(f"[RolloutStats] Recorder initialized -> {self.samples_path}, {self.groups_path}")

    @staticmethod
    def _init_csv(path: str, headers: list[str]):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)

    def record(
        self,
        step_id: int,
        lengths: list[int],
        num_generations: int,
        mode: str = "external",
    ):
        """
        记录一次 rollout 的全部 completion 长度。

        Args:
            step_id: 当前训练 step（与 PhaseEvent 中的 step 对齐）
            lengths: 平铺的 completion token 长度列表，长度 = batch_size_local
                     连续 num_generations 条属于同一个 prompt group
            num_generations: group size（每个 prompt 生成多少条 completion）
            mode: "external" | "colocate"
        """
        ts = time.time()
        gpu_id = self.gpu_id

        # ---- per-completion 明细 ----
        samples_rows = []
        for i, length in enumerate(lengths):
            prompt_idx = i // num_generations
            completion_idx = i % num_generations
            samples_rows.append(
                [step_id, prompt_idx, completion_idx, length, mode, gpu_id, ts]
            )
        with open(self.samples_path, "a", newline="") as f:
            csv.writer(f).writerows(samples_rows)

        # ---- per-group 聚合 ----
        num_prompts = len(lengths) // num_generations if num_generations > 0 else 0
        groups_rows = []
        for p in range(num_prompts):
            group = lengths[p * num_generations : (p + 1) * num_generations]
            if not group:
                continue
            arr = np.array(group, dtype=float)
            mean_val = float(np.mean(arr))
            max_val = float(np.max(arr))
            min_val = float(np.min(arr))
            std_val = float(np.std(arr))
            p90_val = float(np.percentile(arr, 90))
            total = float(np.sum(arr))
            tail_ratio = max_val / mean_val if mean_val > 0 else 0.0
            tail_share = max_val / total if total > 0 else 0.0

            groups_rows.append([
                step_id,
                p,
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
                ts,
            ])

        # 处理尾部不足 num_generations 的余量
        remainder = len(lengths) % num_generations if num_generations > 0 else len(lengths)
        if remainder > 0:
            group = lengths[num_prompts * num_generations :]
            arr = np.array(group, dtype=float)
            mean_val = float(np.mean(arr))
            max_val = float(np.max(arr))
            min_val = float(np.min(arr))
            std_val = float(np.std(arr))
            p90_val = float(np.percentile(arr, 90))
            total = float(np.sum(arr))
            tail_ratio = max_val / mean_val if mean_val > 0 else 0.0
            tail_share = max_val / total if total > 0 else 0.0
            groups_rows.append([
                step_id,
                num_prompts,
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
                ts,
            ])

        with open(self.groups_path, "a", newline="") as f:
            csv.writer(f).writerows(groups_rows)
