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

**服务器方式部署（推荐）：**
```bash
# 1. 下载到临时目录
python download.py download Qwen/Qwen2.5-1.5B-Instruct --local-dir tmp

# 2. 移动到统一存放的目录
sudo mv tmp /etc/moreh/checkpoint/Qwen/Qwen2.5-1.5B-Instruct

# 3. 创建软链接
ln -s /etc/moreh/checkpoint/Qwen ./models/
```

## 当前进度
- 支持 NVIDIA / AMD SMI 采样、内存缓冲与 CSV 日志
- 提供 Gradio 实时监控 UI（可调窗口、采样间隔、GPU 选择、暂停刷新）
- 提供模型下载工具与推理 demo 脚本（用于事件标记验证）

## 下一步计划
- 更复杂的插桩与回调，监控更细粒度的训练阶段
- 优化与完善图表标记、数据保存与导出
- 多卡设备测试与稳定性改进
- 完善 AMD 显卡数据字段适配
- 与训练脚本联动
