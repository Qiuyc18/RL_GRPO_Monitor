# RL/GRPO Monitor

## 项目定位
这个仓库现在的目标是做一个可复用的训练监控子模块，而不是绑定某个训练框架的整包工程。

它提供两部分能力：

- GPU 指标采集：按时间序列记录利用率、显存占用、温度
- 阶段事件对齐：把 rollout、reward、forward、backward、optim step 等事件写到同一条时间轴

当前默认集成方向是 veRL 风格训练循环，`app.py` 保留为独立 Gradio UI。`ms-swift` 不再作为仓库内置子模块，只保留一个可选兼容适配层。

## 当前结构

- `monitor/`
  - 监控核心实现
  - `monitor/events.py`：统一阶段事件定义
  - `monitor/integrations/verl.py`：推荐的通用/veRL 事件桥接层
  - `monitor/integrations/ms_swift.py`：旧的 ms-swift 兼容适配层，需外部自行安装 `swift`
- `app.py`
  - Gradio 实时监控面板
- `run.py`
  - 通用 smoke/demo 入口，用于验证事件与 GPU 指标是否正常落盘
- `plugin.py`、`plugin_rollout.py`
  - 保留旧文件名，内部改为兼容壳层

## 安装

先安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后安装当前机器所需的最小依赖：

```bash
chmod +x install.sh
bash install.sh
```

如果你只想把它作为其他项目的子模块使用，也可以直接在目标项目环境里安装：

```bash
uv pip install -e .
```

按平台补充依赖：

- NVIDIA：安装 `nvidia-ml-py`
- AMD：安装 `amdsmi`

如果需要使用 `download.py` 或 `prepare_data.py`，再额外安装 `huggingface_hub` 和 `datasets`。

## 快速验证

1. 生成一份模拟训练日志：

```bash
python run.py --steps 5
```

2. 打开监控 UI：

```bash
python app.py
```

默认会在 `logs/` 下生成：

- `gpu_metrics.csv`
- `gpu_events.csv`
- `rollout_samples*.csv`
- `rollout_groups*.csv`

## 作为子模块接入其他项目

推荐接法是直接在训练循环里接 `VerlMonitorBridge`，而不是复用旧的框架启动脚本。

```python
from monitor import Monitor
from monitor.integrations.verl import VerlMonitorBridge

monitor = Monitor(
    platform="nvidia",
    output_file_path="logs/gpu_metrics.csv",
    events_file_path="logs/gpu_events.csv",
    interval=0.1,
)
monitor.start()

bridge = VerlMonitorBridge.from_local_rank(monitor, local_rank=0, mode="verl")
bridge.step_start(step=0)
bridge.rollout_start(step=0)
bridge.rollout_end(step=0)
bridge.forward_start(step=0)
bridge.forward_end(step=0)
bridge.backward_start(step=0)
bridge.backward_end(step=0)
bridge.step_end(step=0)
```

如果你后面切到 veRL，只需要把这些事件打点插到你自己的 trainer / worker 循环里即可。

## 兼容说明

- 仓库已去掉对 `ms-swift` 子模块的默认依赖
- `plugin.py` 和 `plugin_rollout.py` 仍然保留
- 只有在你显式调用 `monitor.integrations.ms_swift` 里的补丁函数时，才会要求外部环境已经装好 `swift`

## 后续建议

- 把 veRL 中 actor / rollout / critic 的具体事件边界映射到 `PhaseEvent`
- 如果需要多进程聚合，可在事件里补充 `worker_id` 或 `rank`
- 如果以后要完全作为库分发，可以继续把根目录脚本收缩到 `examples/`
