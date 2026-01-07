# **1. 执行摘要与系统愿景**

在当前大语言模型（LLM）对齐技术的演进浪潮中，群组相对策略优化（Group Relative Policy Optimization, GRPO）已成为替代传统 PPO 算法的关键范式。GRPO 通过消除独立的价值模型（Critic Model）并利用群组内的相对优势进行策略更新，显著降低了显存开销，使得在有限算力下对 Qwen2.5-1.5B 等模型进行强化学习微调（RLHF）成为可能 。然而，GRPO 引入了独特的“生成-训练”交替工作负载，其资源消耗模式在时间维度上呈现出剧烈的锯齿状波动。

目前的训练基础设施主要依赖 `ms-swift` 框架提供的命令行接口（CLI），如 `swift rlhf`。这种“黑盒”式的启动方式虽然简化了流程，却在工程实践中造成了显著的可观测性盲区。运维与算法工程师无法实时关联 GPU 显存峰值与具体的训练阶段（是采样生成的 KV Cache 爆炸，还是反向传播的梯度累积溢出？），也难以在多卡环境中精确定位通信瓶颈（NCCL Timeout）。

本报告旨在详尽阐述一套从 CLI 迁移至 Python原生控制流（Script-based Workflow）的完整技术方案。该方案的核心在于构建一个**GPU 监控与 GRPO 训练流可视分析系统**。通过在主进程（Rank 0）中集成高频 `GpuMonitor` 遥测模块，并利用 Python 动态特性（Monkey Patching）对 `GRPOTrainer` 进行运行时插桩，系统将实现对训练全生命周期的毫秒级观测。最终，借助 Gradio 与 Altair 的可视化能力，系统将在时间轴上精准叠加 GPU 硬件指标与算法逻辑事件，为从单卡验证（MVP）到多机多卡大规模训练提供坚实的调试与优化依据。

# **2. GRPO 算法原理与硬件负载动力学分析**

要设计有效的监控系统，首先必须深入剖析被监控对象——GRPO 算法的计算特性及其对 GPU 资源的动态需求。与监督微调（SFT）相对平稳的负载不同，GRPO 是一种混合型负载，交替执行推理（Inference）与训练（Training）。

## **2.1 GRPO 的双模态计算周期**

GRPO 的核心循环可以被抽象为两个截然不同的计算模态，监控系统必须能够清晰地识别并标记这两个阶段的边界。

### **2.1.1 采样生成阶段**

在此阶段，策略模型（Policy Model）根据输入的 Prompts 生成一系列回复（Completions）。假设群组大小（Group Size）为 *G*，模型需要进行自回归解码。

- **计算特征**：这是典型的推理任务，受限于显存带宽（Memory Bandwidth Bound）。
- **显存动力学**：显存占用随生成序列长度线性增长。主要消耗在于 KV Cache（Key-Value Cache）。每生成一个 Token，模型必须缓存所有层的 Key 和 Value 矩阵以避免重算。
- **OOM 风险点**：当 `max_new_tokens` 设置过大或 batch size 较高时，KV Cache 可能瞬间填满显存，尤其是在长上下文（Long Context）场景下 。此时的显存曲线呈现“爬坡”状。

### **2.1.2 策略优化阶段**

生成结束后，系统根据奖励模型（Reward Model）或规则函数对回复进行打分，并计算优势（Advantage）。随后，进入标准的梯度下降过程。

- **计算特征**：这是计算密集型任务（Compute Bound），涉及前向传播（计算 Logits 和 Loss）与反向传播（计算梯度）。
- **显存动力学**：显存占用呈现阶跃式突增。系统不仅需要存储模型参数，还需要存储优化器状态（Optimizer States，如 AdamW 的动量与方差，通常是参数量的 2 倍）、梯度（Gradients）以及用于反向传播的激活值（Activations）。
- **OOM 风险点**：当 `per_device_train_batch_size` 过大时，激活值显存可能溢出。此时的显存曲线呈现“平台”状的高位运行 。

## **2.2 锯齿状显存模式与监控盲区**

文献指出，GRPO 的显存特征在时间轴上表现为“锯齿状”（Sawtooth Pattern）。在 `swift rlhf` 的黑盒模式下，用户只能看到显存占用的最终结果（即 OOM 报错），而无法得知 OOM 发生的精确时刻。

- **盲区一：阶段归属不明**。如果 OOM 发生在第 N 步，CLI 日志通常只显示 `CUDA out of memory`。如果缺乏时间轴对齐，工程师无法判断是应该减少生成长度（针对采样阶段）还是减少训练 Batch Size（针对优化阶段）。
- **盲区二：利用率波谷归因**。在多卡训练中，若 GPU 计算利用率（SM Utilization）突然跌至 0%，可能是因为 Rank 0 正在进行 CPU 密集的 Tokenization，也可能是因为 NCCL 通信死锁。传统的 `nvidia-smi` 无法提供上下文信息来区分这两者。

因此，构建可视分析系统的首要理论依据，就是必须将**算法逻辑时间**（Step, Phase）与**物理时间**（Timestamp）进行强耦合。

# **3. 核心技术路线：从 CLI 到 Python 控制流的范式转移**

为了获得对训练过程的完全控制权，必须放弃 `swift rlhf` 这种 shell 脚本启动方式，转而采用 Python 脚本 (`run_train.py`) 直接调用 `ms-swift` 的底层 API。这不仅是实现监控的前提，也是通向高级定制化（如自定义 Trainer、复杂回调）的必经之路。

## **3.1 现有架构剖析：`ms-swift` 的启动机制**

通过查阅 `ms-swift` 源代码 ，我们发现 `swift rlhf` 命令本质上是调用了 `swift.llm.train.rlhf.rlhf_main` 函数。该函数的标准执行流程如下：

1. **参数解析**：解析命令行参数为 `RLHFArguments` 对象。
2. **实例构建**：使用参数初始化 `SwiftRLHF` 类。
3. **执行训练**：调用 `SwiftRLHF.main()`，进而调用内部的 `GRPOTrainer.train()`。

这种封装虽然方便，但它是一个阻塞过程。一旦调用 `rlhf_main()`，控制权即移交给库函数，直到训练结束或崩溃。要实现**并发监控**，我们需要在调用 `main()` 之前启动独立的监控线程，并在 `Trainer` 内部注入探针。

## **3.2 目标架构设计**

本报告提出的架构方案包含三个在同一进程空间内并行协作的子系统：

| 子系统 | 职责 | 技术栈 | 运行载体 |
| --- | --- | --- | --- |
| **控制平面 (Control Plane)** | 负责模型加载、训练循环驱动、钩子注入 | `ms-swift`, `transformers`, `pynvml` | 主线程 (Main Thread) |
| **遥测引擎 (Telemetry Engine)** | 高频采集 GPU 状态，写入缓冲区与 CSV | `pynvml`, `threading` | 守护线程 (Daemon Thread) |
| **可视分析层 (Visualizer)** | 读取数据并渲染实时图表，提供 Web UI | `gradio`, `altair`, `pandas` | 独立服务器/线程 |

## **3.3 进程模型与 Rank 0 独占策略**

在多卡训练环境（DDP/DeepSpeed）中，`torchrun` 会启动多个进程（例如 8 卡启动 8 个进程）。如果每个进程都启动一个监控 UI，会导致端口冲突（Port Conflict）和资源浪费。

**关键设计决策**：只有全局主进程（Rank 0）负责启动监控服务。

根据 `ms-swift` 的工具库文档 ，我们可以使用 `get_dist_setting` 函数来获取当前进程的 Rank 信息：

```python
from swift.utils import get_dist_setting

rank, local_rank, world_size, _ = get_dist_setting()
is_master = (rank == 0)
```

系统逻辑将严格分支：

- **Rank 0**：初始化 `GpuMonitor`，启动 Gradio Server，挂载 Monkey Patch 钩子，执行训练。
- **Rank > 0**：仅挂载必要的同步钩子（如果需要），执行训练。

# **4. 实施阶段一：控制平面迁移 (run_train.py)**

第一步是将 Shell 命令转化为等效的 Python 脚本。这不仅是为了监控，更是为了让训练配置代码化（Infrastructure as Code）。

## **4.1 参数对象的构建**

在 CLI 模式下，参数通过 `--model_type` 等标志传递。在 Python 模式下，我们需要构建 `RLHFArguments` 对象。根据 Snippet ，该类位于 `swift.llm` 模块中。

**代码实现范式**：

```python
import os
import sys
from swift.llm import RLHFArguments, rlhf_main
from swift.llm.train.rlhf import SwiftRLHF

def build_args():
    # 模拟 CLI 参数，或者直接通过 kwargs 初始化
    # 推荐使用环境变量或配置文件来管理这些参数，以保持灵活性
    args = RLHFArguments(
        model_type='qwen2.5-1.5b-instruct',
        rlhf_type='grpo',
        output_dir='./output/qwen-grpo-monitor',
        dataset='/path/to/dataset',
        # 关键的 GRPO 超参数
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        # 必须显式启用 logging 以便 TrainerCallback 工作
        logging_steps=1, 
        report_to=['tensorboard']
    )
    return args
```

## **4.2 训练入口的封装**

为了插入监控逻辑，我们不能直接调用 `rlhf_main()`，因为它会立即阻塞。我们需要将训练逻辑封装在一个函数中，以便在监控线程启动后调用。

```python
def start_training_loop(args):
    # 初始化 SwiftRLHF 引擎
    # 注意：此时模型并未加载，直到调用.main()
    rlhf_engine = SwiftRLHF(args)
    
    # 在此处可以进行更深度的定制，例如修改 rlhf_engine 内部的 trainer_args
    
    # 启动训练
    rlhf_engine.main()
```

此结构的优势在于，我们可以在 `SwiftRLHF` 实例化之后、`main()` 执行之前，对环境进行修改，例如应用 Monkey Patch。

# **5. 实施阶段二：核心监控模块 (GpuMonitor)**

`GpuMonitor` 是系统的感知神经，负责将底层的硬件状态转化为上层可理解的数据流。

## **5.1 基于 pynvml 的高频遥测**

选择 `pynvml` 而非 `torch.cuda.memory_allocated()` 至关重要。PyTorch 的内存 API 只能看到 PyTorch 自身管理的显存池，而无法看到驱动层面的开销（Context Overhead）、碎片化损耗以及其他进程（如系统显示输出）占用的显存。对于 OOM 排查，`pynvml` 提供的 `used` 总量才是真理 。

**Monitor 类设计规范**：

1. **线程安全性**：由于 Monitor 线程与 Gradio 线程可能同时访问数据缓冲区，必须使用 `threading.Lock` 保护读写操作。
2. **数据持久化**：内存缓冲区（Deque）用于实时展示，仅保留最近 N 分钟的数据；CSV 文件用于全量归档，必须支持 `flush` 操作，防止程序崩溃时数据丢失。
3. **资源清理**：利用 `atexit` 模块注册 `nvmlShutdown`，防止 Python 进程退出后僵尸句柄残留。

## **5.2 核心代码实现逻辑**

```python
import threading
import time
import csv
import atexit
import pynvml
from collections import deque

class GpuMonitor:
    def __init__(self, log_dir, interval=1.0):
        self.log_dir = log_dir
        self.interval = interval
        self.running = False
        self.lock = threading.Lock()
        
        # 实时数据缓冲 (UI读取)
        self.metrics_buffer = deque(maxlen=600) # 保留最近600个采样点
        self.events_buffer = deque(maxlen=100)  # 保留最近100个事件
        
        # 初始化 NVML
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        
        # 初始化 CSV
        self.csv_f = open(f"{log_dir}/gpu_metrics.csv", "w", newline='')
        self.csv_writer = csv.writer(self.csv_f)
        self.csv_writer.writerow(["timestamp", "gpu_id", "memory_used", "utilization"])
        
        atexit.register(self.shutdown)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def add_event(self, event_type, step):
        """供外部钩子调用的事件标记接口"""
        ts = time.time()
        with self.lock:
            self.events_buffer.append({
                "timestamp": ts,
                "event": event_type,
                "step": step
            })

    def _loop(self):
        while self.running:
            ts = time.time()
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # 写入 CSV (无锁)
                self.csv_writer.writerow([ts, i, mem.used // 1024**2, util.gpu])
                self.csv_f.flush()
                
                # 写入 Buffer (有锁)
                with self.lock:
                    self.metrics_buffer.append({
                        "timestamp": ts,
                        "gpu_id": i,
                        "memory_used": mem.used // 1024**2,
                        "utilization": util.gpu
                    })
            
            time.sleep(self.interval)

    def shutdown(self):
        self.running = False
        self.csv_f.close()
        try:
            pynvml.nvmlShutdown()
        except:
            pass
```

# **6. 实施阶段三：高级插桩策略 (Hook Strategy)**

这是本项目的核心难点。如何精确地在 `ms-swift` 封装的内部感知到 "Generating" 和 "Backwards" 的边界？

## **6.1 Level 1: 基础回调 (TrainerCallback) 的局限性**

HuggingFace Transformers 提供了 `TrainerCallback` 机制 。

- **优点**：官方支持，稳定性高。
- **能力**：可以捕获 `on_step_begin` 和 `on_step_end`。
- **缺陷**：在 GRPO 中，一个 "Step" 包含了【生成 + 评分 + 训练】的全过程。`on_step_begin` 触发时，生成尚未开始；`on_step_end` 触发时，训练已经结束。Callback 无法深入到 Step **内部** 去区分生成和训练。它只能告诉我们整个 Step 花了多长时间，显存峰值是多少，但无法区分峰值来自哪一阶段。

## **6.2 Level 2: 运行时热补丁 (Monkey Patching)**

为了获取粒度更细的信号，我们必须采用 Monkey Patching 技术。这是一种在运行时动态修改类方法的高级 Python 技巧 。

### **6.2.1 寻找切入点 (The Injection Point)**

通过深入研究 `trl` 库的源代码（`ms-swift` 依赖 `trl` 进行 GRPO 训练），我们发现 `GRPOTrainer` 重写了 `training_step` 方法，或者在 `_prepare_inputs` 中触发生成逻辑。

根据 Snippet  的最新代码线索，生成逻辑通常被封装在 `_generate_and_score_completions` 方法中，或者在 `_prepare_inputs` 中被调用。

- **目标方法 A**: `GRPOTrainer._generate_and_score_completions`
    - 这是最理想的切入点。如果能 wrap 这个方法，我们就能精确包围“生成阶段”。
- **目标方法 B**: `GRPOTrainer.training_step`
    - 这是外层包裹。

### **6.2.2 实现补丁逻辑**

我们需要编写一个函数，该函数接收我们的 `monitor` 实例，并替换 `trl.trainer.grpo_trainer.GRPOTrainer` 的相应方法。

```python
import functools
from trl import GRPOTrainer

def apply_monkey_patches(monitor):
    """
    对 GRPOTrainer 进行动态插桩，注入监控逻辑。
    """
    
    # 1. 捕获原始方法
    # 注意：必须在 SwiftRLHF 初始化之前执行此操作，或者确保类被加载后立即执行
    if not hasattr(GRPOTrainer, '_original_generate'):
        GRPOTrainer._original_generate = GRPOTrainer._generate_and_score_completions

    # 2. 定义 Wrapper
    @functools.wraps(GRPOTrainer._original_generate)
    def patched_generate(self, *args, **kwargs):
        # [进入生成阶段]
        # 获取当前 Step
        current_step = self.state.global_step
        if monitor:
            monitor.add_event("Sampling_Start", current_step)
            
        # 执行原始逻辑 (这是最耗显存的 KV Cache 阶段)
        try:
            result = GRPOTrainer._original_generate(self, *args, **kwargs)
        finally:
            # [退出生成阶段 -> 进入训练阶段]
            if monitor:
                monitor.add_event("Sampling_End", current_step)
                # 紧接着通常就是 Loss 计算和 Backward，所以这也标志着 Training Start
                monitor.add_event("Backward_Start", current_step)
        
        return result

    # 3. 应用补丁
    GRPOTrainer._generate_and_score_completions = patched_generate
    print(" GRPOTrainer monkey patch applied successfully.")
    
    # 4. 辅助补丁：Step 结束
    # 我们可以继续 Patch training_step 来获取 Step End，或者直接用 TrainerCallback
    # 这里为了统一，也可以 Patch training_step
    if not hasattr(GRPOTrainer, '_original_training_step'):
        GRPOTrainer._original_training_step = GRPOTrainer.training_step

    @functools.wraps(GRPOTrainer._original_training_step)
    def patched_training_step(self, *args, **kwargs):
        if monitor:
             monitor.add_event("Step_Start", self.state.global_step)
        
        res = GRPOTrainer._original_training_step(self, *args, **kwargs)
        
        if monitor:
             monitor.add_event("Step_End", self.state.global_step)
        return res
        
    GRPOTrainer.training_step = patched_training_step
```

通过这种双重 Patch 策略，我们在时间轴上获得了四个关键锚点：

1. `Step_Start`
2. `Sampling_Start` (开始爬坡)
3. `Sampling_End` / `Backward_Start` (爬坡结束，开始计算梯度，显存可能阶跃)
4. `Step_End`

# **7. 实施阶段四：UI 联动与可视化 (Gradio + Altair)**

目前的 Gradio 原型仅使用了 `gr.LinePlot`，该组件功能较为基础，难以实现复杂的“垂直标记线”（Vertical Rule）动态叠加。为了满足“精准标记训练阶段”的需求，我们需要下沉到 `gr.Plot` 并直接使用 `Altair` 绘图库。

## **7.1 为什么选择 Altair？**

Gradio 的 `gr.Plot` 组件支持直接渲染 Altair 对象。Altair 基于 Vega-Lite 语法，非常适合处理分层图表（Layered Charts）。我们需要两层图表：

1. **底层 (Line Chart)**：展示 GPU 指标随时间的变化。
2. **顶层 (Rule Chart)**：根据 Event 数据，在特定时间点绘制垂直线，并用颜色区分事件类型（如蓝色表示采样开始，红色表示反向传播开始）。

## **7.2 可视化代码实现**

```python
import gradio as gr
import altair as alt
import pandas as pd

def render_plot(monitor):
    # 1. 获取数据快照 (Snapshot)
    with monitor.lock:
        metrics_data = list(monitor.metrics_buffer)
        events_data = list(monitor.events_buffer)
    
    if not metrics_data:
        return None

    df_metrics = pd.DataFrame(metrics_data)
    df_events = pd.DataFrame(events_data)
    
    # 2. 构建基础显存曲线
    base = alt.Chart(df_metrics).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(format='%H:%M:%S', title='Time'))
    )
    
    line = base.mark_line().encode(
        y=alt.Y('memory_used:Q', title='VRAM (MB)'),
        color=alt.Color('gpu_id:N', title='GPU ID'),
        tooltip=['gpu_id', 'memory_used', 'utilization']
    )
    
    # 3. 构建事件标记线 (Vertical Rules)
    # 只有当有事件时才绘制
    if not df_events.empty:
        # 定义颜色映射：Sampling=Blue, Backward=Red
        domain =
        range_ = ['gray', 'blue', 'orange', 'red', 'green']
        
        rules = alt.Chart(df_events).mark_rule(strokeWidth=2).encode(
            x='timestamp:T',
            color=alt.Color('event:N', scale=alt.Scale(domain=domain, range=range_), title='Event'),
            tooltip=['event', 'step']
        )
        
        # 叠加图表
        chart = (line + rules).interactive()
    else:
        chart = line.interactive()
        
    return chart

# Gradio 界面定义
def launch_ui(monitor):
    with gr.Blocks(title="GRPO Training Monitor") as demo:
        gr.Markdown("## Qwen2.5 GRPO 实时训练监控")
        
        with gr.Row():
            plot = gr.Plot(label="显存与训练阶段关联分析")
        
        # 定时刷新器 (每1秒刷新一次)
        timer = gr.Timer(1)
        timer.tick(lambda: render_plot(monitor), outputs=plot)
        
    # 启动服务，允许并发访问，避免阻塞训练线程
    demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
```

## **7.3 多卡数据聚合展示 (Future-Proofing)**

在多卡环境中（如阶段二），`monitor.metrics_buffer` 将包含所有 GPU 的数据（通过 `gpu_id` 区分）。Altair 的 `color='gpu_id:N'` 参数会自动为每张卡生成一条曲线，无需额外代码即可支持 8 卡环境的并行展示。

# **8. 下一步实施计划与验证策略**

## **8.1 阶段一：WSL 单卡 MVP 验证**

**目标**：跑通 `run_train.py`，验证 Monkey Patch 是否生效，确认 UI 上能看到垂直线与显存波动的对齐。

**操作步骤**：

1. **环境准备**：在 WSL 中配置 `ms-swift` 和 `pynvml`。
2. **脚本编写**：整合上述代码为 `run_train.py` 和 `monitor.py`。
3. **小规模测试**：
    - 使用极小的数据集（如 100 条）和极小的模型（如 Qwen2.5-0.5B）进行快速迭代。
    - 检查 `training_events.csv`，确认 `Sampling_Start` 和 `Backward_Start` 是否成对出现。
    - **关键验证点**：观察 Gradio 图表。理论上，在 `Sampling_Start` 和 `Sampling_End` 之间的显存曲线应呈现爬坡状；而在 `Backward_Start` 之后应出现显存阶跃。如果观测到此现象，证明系统成功解构了 GRPO 的黑盒。

## **8.2 阶段二：多卡环境适配 (Future)**

**挑战**：在 DDP 模式下，多个进程同时运行。 **解决方案**：

- **Rank 0 独占**：代码中严格检查 `rank == 0` 才启动 `GpuMonitor` 和 `Gradio`。
- **数据一致性**：`pynvml` 在 Rank 0 上可以读取本机所有 GPU 的状态（假设所有卡在同一物理机）。如果跨机训练（Multi-Node），目前的方案只能监控主节点。未来需引入 Redis 或分布式队列来汇聚所有节点的指标。
- **压力测试**：在 8 卡 H800 环境下，高频（0.1s）监控可能会引入 GIL 竞争，影响训练效率。需进行 A/B 测试（开启/关闭监控）对比 Throughput (tokens/sec)，根据结果调整采样频率至 1.0s 或更低。

# **9. 结论**

本报告提出的**GPU 监控与 GRPO 训练流可视分析系统**，通过从 CLI 到 Python 脚本的架构迁移，结合 `pynvml` 硬件遥测与 `GRPOTrainer` 运行时热补丁，彻底解决了传统训练模式下的可观测性缺失问题。利用 Altair 的分层可视化能力，系统能够直观地揭示 GRPO 算法独特的“锯齿状”资源消耗特征，为排查 Qwen2.5-1.5B 训练中的 OOM 和通信瓶颈提供了强有力的工具。该方案不仅满足了当前的单卡验证需求，也为未来的大规模分布式训练监控奠定了坚实的技术基础。

# 参考资料

[**swift.readthedocs.io**GRPO — swift 3.6.4 documentation在新窗口中打开](https://swift.readthedocs.io/en/v3.6/Instruction/GRPO/GetStarted/GRPO.html)

[**huggingface.co**GRPO Trainer - Hugging Face在新窗口中打开](https://huggingface.co/docs/trl/main/en/grpo_trainer)

[**github.com**ms-swift/docs/source_en/Instruction/Frequently-asked-questions.md at main - GitHub在新窗口中打开](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md)

[**huggingface.co**Liger GRPO meets TRL - Hugging Face在新窗口中打开](https://huggingface.co/blog/liger-grpo)

[**github.com**TypeError in rollout_server.sh: EngineArgs.__init__() got unexpected keyword 'worker_extension_cls' · Issue #4202 · modelscope/ms-swift - GitHub在新窗口中打开](https://github.com/modelscope/ms-swift/issues/4202)

[**github.com**qwen3-embedding infonce的loss没完整实现论文中的loss设计· Issue #6273 - GitHub在新窗口中打开](https://github.com/modelscope/ms-swift/issues/6273)

[**github.com**ms-swift/swift/llm/model/model/qwen.py at main - GitHub在新窗口中打开](https://github.com/modelscope/ms-swift/blob/main/swift/llm/model/model/qwen.py)

[**github.com**ms-swift/swift/plugin/callback.py at main · modelscope/ms-swift ...在新窗口中打开](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/callback.py)

[**training-docs.cerebras.ai**Customizing the Trainer with Callbacks - Cerebras AI在新窗口中打开](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/components/trainer-components/customizing-the-trainer-with-callbacks)

[**swift.readthedocs.io**Pluginization — swift 3.13.0.dev0 documentation在新窗口中打开](https://swift.readthedocs.io/en/latest/Customization/Pluginization.html)

[**huggingface.co**Decorators in Machine Learning - Hugging Face在新窗口中打开](https://huggingface.co/blog/NormalUhr/decorators)

[**github.com**trl/trl/trainer/grpo_trainer.py at main · huggingface/trl - GitHub在新窗口中打开](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)

[**github.com**RuntimeError: CUDA error: · Issue #3150 · unslothai/unsloth - GitHub在新窗口中打开](https://github.com/unslothai/unsloth/issues/3150)

[**docs.streamlit.io**Annotate an Altair chart - Streamlit Docs在新窗口中打开](https://docs.streamlit.io/develop/tutorials/elements/annotate-an-altair-chart)

[**gradio.app**LinePlot - Gradio Docs](https://www.gradio.app/docs/gradio/lineplot)