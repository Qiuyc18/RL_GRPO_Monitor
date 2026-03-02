# GRPO 训练流可视分析系统

## 项目说明
用于在 GRPO/RLHF 训练或推理过程中对 GPU 负载进行可视化监控，并尝试在时间轴上对齐训练事件与硬件指标。

## 部署

- 首先确保安装了 uv，如果没装，使用下面这条命令安装
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    - 随后重启终端或者刷新一下 `~/.bashrc`

- 安装环境
    ```bash
    chmod +x install.sh && bash install.sh
    ```

- 新建文件 `.env` ，在 `.env` 中更新 HuggingFace Token

- 下载子模块
    ```bash
    git submodule update --init --recursive
    ```

## 下载模型

```bash
# 搜索模型
python download.py search Qwen --limit 20

# 下载模型（默认保存到 ./models/）
python download.py download Qwen/Qwen2.5-1.5B-Instruct
```

**服务器方式部署：**
```bash
# 1. 下载到临时目录
python download.py download Qwen/Qwen2.5-1.5B-Instruct --local-dir tmp

# 2. 移动到统一存放的目录
sudo mv tmp /etc/moreh/checkpoint/Qwen/Qwen2.5-1.5B-Instruct

# 3. 创建软链接
ln -s /etc/moreh/checkpoint/Qwen ./models/
```

## 训练测试
验证 grpo 能否正常跑起来
- 测试脚本：`run.py`
```bash
# 在 GPU0 上部署 vllm rollout
python run.py --rollout
# 在另一个窗口：使用 torchrun 运行训练，会调用剩下的几个 GPU 进行训练（指定数量）
uv run torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    run.py --grpo
```


## 当前进度
- **自动化训练与监控脚本**:
    - 提供 `run.py` 脚本，可通过 `--rollout` 和 `--grpo` 参数一键启动 vLLM 服务和 GRPO 训练。
    - **智能 GPU 分配**: 自动将 vLLM 推理服务部署在 GPU 0，GRPO 训练则使用所有其他剩余 GPU，实现资源隔离与高效利用。
    - **集成的监控系统**: 训练启动时，自动运行监控模块，捕获所有物理 GPU 的关键指标（利用率、显存、功率、温度等）并存入 CSV。
    - **关键事件捕获**: 通过对 `ms-swift` 中 `GRPOTrainer` 的动态修改 (Monkey Patching)，已能准确捕获 `推理开始/结束` 和 `训练开始/结束` 等核心事件，并与 GPU 指标在时间轴上对齐。
- **数据准备与模型管理**:
    - 提供通用的 `prepare_data.py` 脚本，支持从 Hugging Face Hub 搜索和下载数据集，并将其处理为 GRPO 训练所需的格式。
    - 提供 `download.py` 脚本，方便地从 Hugging Face Hub 下载和管理模型。
- **监控 UI**:
    - 提供 Gradio 实时监控 UI（可调窗口、采样间隔、GPU 选择、暂停刷新）。

## 下一步计划
- **更精细的事件捕获**: 在训练循环内部（如数据加载、前向/反向传播、梯度更新等）添加更详细的事件探针，以实现对训练阶段的微观分析。
- **事件与 GPU 的精确绑定**: 在多卡训练环境下，确保每个事件都能明确关联到触发它的具体 GPU 设备。
- **实时监控与训练联动**: 实现训练脚本启动时，能自动在后台打开并运行 `app.py` 监控仪表盘，方便实时观察。
- **AMD ROCm 平台全面支持**: 将监控和训练脚本扩展至完全支持 AMD GPU，包括使用 `rocm_smi_lib` 进行指标收集和 `HIP_VISIBLE_DEVICES` 的等效设置。
- **多节点训练支持**: 适配监控与启动脚本，以支持跨多个节点的大规模分布式训练。
