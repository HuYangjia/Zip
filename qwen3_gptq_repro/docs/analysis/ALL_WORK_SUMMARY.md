# Qwen3 GPTQ 全量工作总结（重写版）

## 1. 目标与范围

本阶段围绕三条主线推进，并保持“代码隔离、可复现、先做 sanity 不做端到端评测”的原则：

1. 原始权重直接 GPTQ（基线）。
2. Smooth(alpha=1) 后再 GPTQ（平滑链路）。
3. Percentile + Tail 分层残差吸收原型（两种模式 A/B）。

当前范围不包含 perplexity、生成质量、下游任务等端到端评测，仅聚焦方法实现、产物落盘与层级统计。

## 2. 代码与文档结构调整

### 2.1 文档拆分与导航

- 将主文档改为导航页：`LEARN_AND_RUN.md`
- 拆分出独立流程文档：
  - `LEARN_AND_RUN_GPTQ_ONLY.md`
  - `LEARN_AND_RUN_SMOOTH_GPTQ.md`
- 新增实现复核文档：
  - `SMOOTH_IMPL_REVIEW.md`

### 2.2 Smooth 独立实现

新增 `qwen3_smooth.py`，核心能力：

- 采集 RMSNorm 前向激活绝对值统计；
- 执行 smooth（alpha 可配，实验使用 alpha=1）；
- 保存三类产物：
  - `smoothed_model_state_dict.pt`
  - `smooth_scales.pt`
  - `smooth_metadata.json`

### 2.3 GPTQ 脚本增强（兼容 smooth 后链路）

在 `qwen3_gptq.py` 中增加：

- `--init-state-dict`：先载入外部 state_dict 再量化；
- `--smooth-scales-path` / `--smooth-metadata-path`：在 metadata 记录 smooth 溯源信息。

### 2.4 推理脚本与便捷命令

- 复用 `infer_quantized_qwen3.py` 做三类权重推理；
- 新增 `run_three_infer.sh`，统一执行：
  - smooth 后未 GPTQ 权重；
  - smooth 后 GPTQ 权重；
  - 原始 GPTQ 权重。

## 3. 新方法原型：Percentile + Tail

## 3.1 代码隔离

新增独立脚本：`qwen3_gptq_percentile_tail.py`，不污染原始 `qwen3_gptq.py`。

## 3.2 方法机制

按列分块：`W = [W_main, W_tail]`，其中：

- `main`：低比特（INT4）主体；
- `tail`：高精度缓冲（本阶段用 INT8）。

两种模式：

- **Mode A**：main 使用 percentile scale + 约束吸收 + 溢出投影到 tail；
- **Mode B**：main 先走 GPTQ，再做约束吸收与 tail 补偿。

## 3.3 稳定性与可观测性

已实现并记录：

- `H_tt` 相关稳定化与 `lambda_reg`；
- 主吸收预算约束检查；
- 三段 residual 统计；
- 每层统计 + 全局 metadata 输出。

## 4. 关键问题与修复

### 4.1 mode B 高违规问题

问题：`main_constraint_violations` 初期偏高。

修复：

1. 将 mode B 的约束边界从 percentile 边界改为 `gptq_q0_rowmax`（与 GPTQ 主量化结果一致）；
2. 新增 `--constraint-eps`（默认 `1e-5`）过滤数值噪声；
3. 重跑后违规归零（在 epsilon 判定口径下）。

## 5. 产物与记录目录

实验记录已集中在：

- `EXPERIMENTS/PERCENTILE_TAIL_METHOD.md`
- `EXPERIMENTS/PERCENTILE_TAIL_RUNLOG.md`
- `EXPERIMENTS/PERCENTILE_TAIL_SANITY_SUMMARY.md`
- `EXPERIMENTS/ALL_WORK_SUMMARY.md`（本文件）

模型与量化输出位于 `output/`（包含 smooth、中间态、不同模式权重与 metadata）。

## 6. 当前状态（截至本次重写）

- 主线 1（原始 GPTQ）：已完成并可推理。
- 主线 2（Smooth + GPTQ）：已完成并可推理。
- 主线 3（Percentile + Tail）：
  - 原型实现完成；
  - mode A 已有多组 sanity 结果；
  - mode B 经过边界与 epsilon 修复后可稳定运行。

## 7. 后续建议（仍限 sanity）

若继续迭代，建议保留“仅层级趋势分析”：

- 小网格比对 `k`、`tail_rank` 对 residual/budget/tail 饱和度影响；
- 固定预算（`nsamples=32, seqlen=1024`）保持横向可比；
- 暂不进入端到端评测，待机制收敛后再统一评估。
