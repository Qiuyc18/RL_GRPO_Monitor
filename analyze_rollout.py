"""
最小可行分析脚本 —— 验证 GRPO 训练中是否存在 APRIL 论文所述的 rollout response length 长尾问题。

用法:
    python analyze_rollout.py <log_dir>

其中 <log_dir> 为训练日志目录，例如 logs/grpo_experiment_20250101_120000
脚本会自动查找其中的 rollout_samples_*.csv / rollout_groups_*.csv / gpu_events_*.csv

输出 4 张图：
  1. completion length 直方图
  2. 每个 step 的 len_max / len_mean（tail_ratio）
  3. 每个 step 的 rollout_phase_duration vs len_max
  4. 每个 group 的 tail_share 分布
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_csvs(log_dir: str, pattern: str) -> pd.DataFrame:
    """加载 log_dir 下所有匹配 pattern 的 CSV，合并成一个 DataFrame。"""
    files = sorted(glob.glob(os.path.join(log_dir, pattern)))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_rollout_durations(events_df: pd.DataFrame) -> pd.DataFrame:
    """从 phase event CSV 中计算每个 step 的 rollout phase 耗时。

    由于当前 events CSV 可能不含 step 列（或全为空），
    这里用 ROLLOUT_PHASE_START/END 的时间戳配对来估算。
    """
    if events_df.empty:
        return pd.DataFrame(columns=["step_id", "rollout_duration"])

    # 标准化 event_type 列
    events_df = events_df.copy()
    events_df["event_type"] = events_df["event_type"].astype(str)

    starts = events_df[
        events_df["event_type"].str.contains("ROLLOUT_PHASE_START", case=False)
    ].copy()
    ends = events_df[
        events_df["event_type"].str.contains("ROLLOUT_PHASE_END", case=False)
    ].copy()

    if starts.empty or ends.empty:
        return pd.DataFrame(columns=["step_id", "rollout_duration"])

    starts = starts.sort_values("timestamp").reset_index(drop=True)
    ends = ends.sort_values("timestamp").reset_index(drop=True)

    # 配对：按顺序一一对应
    n = min(len(starts), len(ends))
    durations = []
    for i in range(n):
        dur = ends.iloc[i]["timestamp"] - starts.iloc[i]["timestamp"]
        if dur >= 0:
            durations.append({"step_id": i, "rollout_duration": dur})

    return pd.DataFrame(durations)


def plot_analysis(log_dir: str, output_dir: str | None = None):
    if output_dir is None:
        output_dir = os.path.join(log_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # ---- 加载数据 ----
    samples = load_csvs(log_dir, "rollout_samples*.csv")
    groups = load_csvs(log_dir, "rollout_groups*.csv")
    events = load_csvs(log_dir, "gpu_events*.csv")

    if samples.empty:
        print("[ERROR] No rollout_samples*.csv found. Run training first.")
        sys.exit(1)

    # 确保类型
    for col in ["completion_length", "step_id"]:
        if col in samples.columns:
            samples[col] = pd.to_numeric(samples[col], errors="coerce")
    for col in ["len_mean", "len_max", "len_min", "len_std", "len_p90",
                 "tail_ratio", "tail_share", "step_id"]:
        if col in groups.columns:
            groups[col] = pd.to_numeric(groups[col], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Rollout Completion Length Analysis (APRIL Long-Tail Check)", fontsize=14)

    # ---- 1. Completion length 直方图 ----
    ax = axes[0, 0]
    lengths = samples["completion_length"].dropna()
    ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(lengths.mean(), color="red", linestyle="--", label=f"mean={lengths.mean():.0f}")
    ax.axvline(lengths.quantile(0.9), color="orange", linestyle="--", label=f"p90={lengths.quantile(0.9):.0f}")
    ax.axvline(lengths.max(), color="darkred", linestyle="--", label=f"max={lengths.max():.0f}")
    ax.set_xlabel("Completion Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("1. Completion Length Distribution")
    ax.legend(fontsize=8)

    # ---- 2. 每个 step 的 tail_ratio (len_max / len_mean) ----
    ax = axes[0, 1]
    if not groups.empty and "tail_ratio" in groups.columns:
        step_agg = groups.groupby("step_id").agg(
            tail_ratio_mean=("tail_ratio", "mean"),
            len_max=("len_max", "max"),
            len_mean=("len_mean", "mean"),
        ).reset_index()
        ax.bar(step_agg["step_id"], step_agg["tail_ratio_mean"], alpha=0.7, color="steelblue")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("tail_ratio (len_max / len_mean)")
        ax.set_title("2. Per-Step Tail Ratio")
        # 标注极端值
        if len(step_agg) > 0:
            max_tr = step_agg["tail_ratio_mean"].max()
            ax.axhline(max_tr, color="red", linestyle="--", alpha=0.5, label=f"max={max_tr:.2f}")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No group data", ha="center", va="center", transform=ax.transAxes)

    # ---- 3. rollout_phase_duration vs len_max ----
    ax = axes[1, 0]
    durations = compute_rollout_durations(events)
    if not durations.empty and not groups.empty:
        step_max = groups.groupby("step_id")["len_max"].max().reset_index()
        merged = pd.merge(durations, step_max, on="step_id", how="inner")
        if not merged.empty:
            ax.scatter(merged["len_max"], merged["rollout_duration"], alpha=0.6, edgecolors="black", s=40)
            ax.set_xlabel("len_max (tokens)")
            ax.set_ylabel("Rollout Phase Duration (s)")
            ax.set_title("3. Rollout Duration vs Max Completion Length")
            # 线性拟合
            if len(merged) > 2:
                z = np.polyfit(merged["len_max"], merged["rollout_duration"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(merged["len_max"].min(), merged["len_max"].max(), 50)
                ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"slope={z[0]:.4f}")
                corr = merged["len_max"].corr(merged["rollout_duration"])
                ax.legend(title=f"r={corr:.3f}", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Cannot merge durations & groups", ha="center", va="center", transform=ax.transAxes)
    else:
        missing = []
        if durations.empty:
            missing.append("rollout durations")
        if groups.empty:
            missing.append("group stats")
        ax.text(0.5, 0.5, f"Missing: {', '.join(missing)}", ha="center", va="center", transform=ax.transAxes)

    # ---- 4. tail_share 分布 ----
    ax = axes[1, 1]
    if not groups.empty and "tail_share" in groups.columns:
        ts = groups["tail_share"].dropna()
        ax.hist(ts, bins=30, edgecolor="black", alpha=0.7, color="coral")
        ax.axvline(ts.mean(), color="red", linestyle="--", label=f"mean={ts.mean():.3f}")
        # 理想均匀情况下 tail_share = 1/group_size
        if "group_size" in groups.columns:
            ideal = 1.0 / groups["group_size"].mean()
            ax.axvline(ideal, color="green", linestyle="--", label=f"ideal (1/G)={ideal:.3f}")
        ax.set_xlabel("tail_share (len_max / sum(lengths))")
        ax.set_ylabel("Count")
        ax.set_title("4. Tail Share Distribution")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No group data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "rollout_longtail_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Analysis] Saved -> {out_path}")
    plt.close(fig)

    # ---- 汇总统计输出 ----
    print("\n========== Rollout Length Summary ==========")
    print(f"Total completions: {len(samples)}")
    print(f"Length — mean: {lengths.mean():.1f}, std: {lengths.std():.1f}, "
          f"min: {lengths.min():.0f}, max: {lengths.max():.0f}, "
          f"p90: {lengths.quantile(0.9):.0f}, p99: {lengths.quantile(0.99):.0f}")

    if not groups.empty and "tail_ratio" in groups.columns:
        tr = groups["tail_ratio"]
        print(f"\nTail Ratio (len_max/len_mean) — "
              f"mean: {tr.mean():.3f}, max: {tr.max():.3f}, "
              f"p90: {tr.quantile(0.9):.3f}")
        ts_col = groups["tail_share"]
        print(f"Tail Share (len_max/sum) — "
              f"mean: {ts_col.mean():.4f}, max: {ts_col.max():.4f}")

    if not durations.empty:
        print(f"\nRollout Duration — "
              f"mean: {durations['rollout_duration'].mean():.2f}s, "
              f"max: {durations['rollout_duration'].max():.2f}s")

    # 长尾判定（启发式）
    if not groups.empty and "tail_ratio" in groups.columns:
        high_tail = (groups["tail_ratio"] > 2.0).sum()
        total_groups_count = len(groups)
        pct = high_tail / total_groups_count * 100 if total_groups_count > 0 else 0
        print(f"\n>>> Long-tail check: {high_tail}/{total_groups_count} groups ({pct:.1f}%) "
              f"have tail_ratio > 2.0")
        if pct > 10:
            print(">>> CONCLUSION: Significant long-tail phenomenon detected!")
        else:
            print(">>> CONCLUSION: No significant long-tail detected (threshold: 10% groups with ratio>2.0)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rollout completion lengths for APRIL long-tail verification"
    )
    parser.add_argument(
        "log_dir",
        help="Training log directory containing rollout_samples*.csv and rollout_groups*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save analysis results (default: <log_dir>/analysis)",
    )
    args = parser.parse_args()
    plot_analysis(args.log_dir, args.output_dir)


if __name__ == "__main__":
    main()
