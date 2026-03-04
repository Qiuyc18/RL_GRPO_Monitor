# GRPO 监控事件语义与精度边界

统一事件层使用 `PhaseEvent` 枚举，与 Colocate / External 模式解耦，便于后续双模式支持。

## 事件枚举 (PhaseEvent)

| 事件 | 语义 | 当前注入位置 | 精度边界 |
|------|------|--------------|----------|
| `STEP_START` | 单个 training step 开始 | `training_step` 入口 | 步级；与 backward 结束之间可能含 optimizer step |
| `STEP_END` | 单个 training step 结束 | `training_step` 返回前 | 步级 |
| `ROLLOUT_PHASE_START` | 进入“rollout 阶段”（等待/执行生成） | External: `_generate_and_score_completions` 入口；Rollout 进程: `SwiftRolloutDeploy.infer` 入口 | External: trainer 侧为“等待 vLLM”；rollout 侧为 vLLM 推理开始 |
| `ROLLOUT_PHASE_END` | rollout 阶段结束 | External: `_generate_and_score_completions` 返回前；Rollout: `infer` 返回前 | 同上 |
| `REWARD_CALC_START` | 开始计算 reward | `_score_completions` 入口 | 与模型 forward 无关的 reward 计算开始 |
| `REWARD_CALC_END` | reward 计算结束 | `_score_completions` 返回前 | 同上 |
| `BATCH_PREP_START` | 开始准备 batch（如 tokenize、拼 batch） | External: `_generate_and_score_completions` 返回后（generate 完成后） | 与 vLLM 返回后到送入 compute_loss 前的预处理对应 |
| `BATCH_PREP_END` | batch 准备结束，即将 forward | `compute_loss` 入口 | 进入 loss 计算前一刻 |
| `FORWARD_START` | 模型 forward 开始 | `compute_loss` 入口（BATCH_PREP_END 之后） | 含 loss 计算，不含 backward |
| `FORWARD_END` | 模型 forward 结束 | `compute_loss` 返回前 | 同上 |
| `BACKWARD_START` | backward 开始 | `compute_loss` 返回前（FORWARD_END 之后） | 与框架 backward 调用边界一致 |
| `BACKWARD_END` | backward 结束 | `training_step` 返回前 | 不含 optimizer.step；若使用 gradient accumulation 则为当前 micro step 的 backward 结束 |
| `OPTIM_STEP_START` | 预留：optimizer step 开始 | 未注入 | TODO |
| `OPTIM_STEP_END` | 预留：optimizer step 结束 | 未注入 | TODO |
| `WEIGHT_SYNC_START` | 预留：权重同步开始（如 ZeRO） | 未注入 | TODO |
| `WEIGHT_SYNC_END` | 预留：权重同步结束 | 未注入 | TODO |

## 元信息

每个事件可携带最小元信息，用于区分来源与模式：

- **mode**: `"external"`（当前）或 后续 `"colocate"`
- **role**: `"trainer"` 或 `"rollout"`
- **gpu_id**: 物理 GPU 编号（与 NVML 一致）

CSV 列为：`timestamp,gpu_id,step,event_type,mode,role`。

## 命名迁移

- `VLLM_WAIT_*` → `ROLLOUT_PHASE_*`（trainer 侧“等待 vLLM”视为 rollout 阶段）
- `TOKENIZE_*` → `BATCH_PREP_*`（tokenize 归属为 batch 准备）

## Monkey-patch 分层

- **common**: 所有模式共用的 trainer 事件（STEP、REWARD_CALC、BATCH_PREP_END、FORWARD、BACKWARD）
- **external rollout**: 仅 External 模式，trainer 侧 ROLLOUT_PHASE_*、BATCH_PREP_START；rollout 进程内 ROLLOUT_PHASE_*（role=rollout）
- **colocate**: 预留，Colocate 实现时在此注入特有事件（TODO）

## 精度边界说明

- **步级事件**：STEP_* 与 training_step 调用边界一致；中间可能包含 gradient accumulation、optimizer step 等，当前不细分。
- **Backward 边界**：BACKWARD_START/END 与 `compute_loss` 返回和 `training_step` 返回对应，不区分 autograd 内部与 optimizer。
- **External 双进程**：rollout 进程与 trainer 进程各自打事件，通过 `role` 与 `gpu_id` 区分；时间戳为各进程本地时间，跨进程对齐需考虑时钟与延迟。
