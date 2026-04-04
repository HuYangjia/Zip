# 工作总结（重写版）

## 本次整体目标

围绕 Qwen3 量化实验，完成了三条链路的可复现实现与记录：

1. 原始权重直接 GPTQ；
2. Smooth(alpha=1) 后再 GPTQ；
3. Percentile + Tail（main 低比特 + tail 高精度缓冲）原型。

强调先做方法与 sanity 验证，不做端到端评测。

## 代码改动总览

### 1) Smooth 独立脚本

- 新增 `qwen3_gptq_repro/qwen3_smooth.py`
- 能力：
  - 采集激活统计；
  - 执行 smooth；
  - 保存 `smoothed_model_state_dict.pt`、`smooth_scales.pt`、`smooth_metadata.json`。

### 2) GPTQ 主脚本增强

- 修改 `qwen3_gptq_repro/qwen3_gptq.py`
- 新增能力：
  - `--init-state-dict` 支持先加载外部权重（用于 smooth 后再 GPTQ）；
  - metadata 记录 smooth 相关路径信息，方便追溯实验来源。

### 3) Tail 混合与 Percentile+Tail 原型

- 新增 `qwen3_gptq_repro/qwen3_gptq_tail_mixed.py`
- 新增 `qwen3_gptq_repro/qwen3_gptq_percentile_tail.py`
- 支持：
  - mode A（percentile main + constrained absorb + tail 补偿）；
  - mode B（GPTQ main + constrained absorb + tail 补偿）；
  - tail INT8 选项、预算与稳定性参数、每层统计输出。

### 4) 推理与便捷脚本

- 复用 `qwen3_gptq_repro/infer_quantized_qwen3.py`
- 新增 `qwen3_gptq_repro/run_three_infer.sh`，统一跑三类权重推理。

## 文档改造总览

### 1) 运行文档拆分

- `qwen3_gptq_repro/LEARN_AND_RUN.md` 改为导航页；
- 新增：
  - `qwen3_gptq_repro/LEARN_AND_RUN_GPTQ_ONLY.md`
  - `qwen3_gptq_repro/LEARN_AND_RUN_SMOOTH_GPTQ.md`

### 2) Smooth 实现复核

- 新增 `qwen3_gptq_repro/SMOOTH_IMPL_REVIEW.md`

### 3) 实验记录集中化

- 重点记录在：
  - `qwen3_gptq_repro/EXPERIMENTS/PERCENTILE_TAIL_METHOD.md`
  - `qwen3_gptq_repro/EXPERIMENTS/PERCENTILE_TAIL_RUNLOG.md`
  - `qwen3_gptq_repro/EXPERIMENTS/PERCENTILE_TAIL_SANITY_SUMMARY.md`
  - `qwen3_gptq_repro/EXPERIMENTS/ALL_WORK_SUMMARY.md`

## 关键问题与修复

1. mode B 约束违规偏高：将边界口径修正为 `gptq_q0_rowmax`；
2. 剩余极小违规：引入 `--constraint-eps`（默认 `1e-5`）过滤数值噪声；
3. 修复后在既定口径下违规清零。

## 产物状态

- smooth 权重与元数据已产出；
- 原始 GPTQ 与 smooth 后 GPTQ 权重已产出；
- Percentile+Tail 多组实验产物与 metadata 已产出并归档；
- 文档已分层组织，支持复现与回溯。

## 本次新增（Git 管理）

- 已在 `zip` 根目录初始化 Git 仓库；
- 已新增 `zip/.gitignore`，默认不跟踪权重与量化产物（`models/`、`**/output/`、`*.pt` 等）。
