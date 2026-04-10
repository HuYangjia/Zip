# Qwen3 GPTQ 全量工作总结（重写版）

## 1. 目标与范围

本项目围绕 Qwen3-4B-Instruct-2507 模型的 4-bit 权重量化展开，从 V1 到 V7 共完成 7 个版本的实验迭代，涵盖以下主线：

1. 原始权重直接 GPTQ（基线）。
2. Smooth(alpha=1) 后再 GPTQ（平滑链路）。
3. Percentile + Tail 分层残差吸收原型（两种模式 A/B）。
4. Tail Absorb（误差正常传播版）与 Tail Spill（误差不传播版）对比。
5. SmoothQuant + Tail Absorb 组合 + act-order 消融实验。

已完成 WikiText-2 PPL 端到端评测（多种激活量化配置：Anone / A8 / A4g128 / A4g128+down:int8），评测脚本为 `benchmark/eval_ppl.py`。

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

## 6. 当前状态（V1-V3 阶段，TAG1 里程碑）

> ✅ V1-V3 阶段已全部完成。后续 V4-V7 的工作详见第 8-11 节。

- 主线 1（原始 GPTQ）：✅ 已完成并可推理。
- 主线 2（Smooth + GPTQ）：✅ 已完成并可推理。
- 主线 3（Percentile + Tail）：✅ 原型实现完成。
  - mode A 已有多组 sanity 结果；
  - mode B 经过边界与 epsilon 修复后可稳定运行。

> 📌 TAG1 里程碑详细总结见 `docs/analysis/TAG1_MILESTONE_SUMMARY.md`。

## 7. V1-V3 阶段后续方向（已在 V4-V7 中推进）

以下方向已在后续版本中逐步推进：

- ✅ V4：将 Tail Spill 改为 Tail Absorb（误差正常传播），使用 Percentile Quantizer（详见第 8 节）。
- ✅ V5：切换为标准 min/max Quantizer + Tail Spill，从原始权重出发（详见第 9 节）。
- ✅ V6：在 V5 基础上加入 SmoothQuant 预处理，发现 rank-PPL 反转 bug（详见第 10 节）。
- ✅ V7：使用 Tail Absorb 修复误差传播问题，完成 act-order 消融实验（详见第 11 节）。
- ✅ 已完成 WikiText-2 PPL 端到端评测，全局汇总见第 12 节。

---

## 8. V4 — Tail Absorb（Percentile Quantizer）

> 📋 AI 指令文档：`docs/ai_instructions/W4A_BENCHMARK_V4_TAIL_ABSORB.md`

### 8.1 核心思路

V4 引入了全新的 `GPTQTailAbsorb` 类（位于 `gptq_tail_absorb.py`），与 V3 的 Tail Spill 机制的关键区别在于：

- **Tail 列做 INT8 fake-quant 并正常传播误差**：tail 列不再跳过量化，而是执行 INT8 对称量化后立即反量化（fake-quant），量化误差通过 GPTQ 的 Hessian 补偿机制正常传播到后续列。
- **Main 列使用 `PercentileQuantizer`**：以第 k 百分位数确定量化范围（而非标准 min/max），截断 outlier 以降低量化误差。

### 8.2 实验脚本

- 量化入口：`qwen3_gptq_tail_absorb.py`
- GPTQ 核心：`gptq_tail_absorb.py`（`GPTQTailAbsorb` 类）

### 8.3 实验变体

3 种 tail_rank 配置，均从原始 FP16 权重出发：

| 变体 | tail_rank | percentile_k | 权重来源 |
|------|-----------|-------------|---------|
| Tail Absorb (raw, r=16) | 16 | 75 | FP16 原始 |
| Tail Absorb (raw, r=64) | 64 | 75 | FP16 原始 |
| Tail Absorb (raw, r=128) | 128 | 75 | FP16 原始 |

> ⚠️ V4 实验未进行 PPL 评测（仅做 sanity check），PPL 数据不可用。

---

## 9. V5 — Standard Tail Spill（from raw）

> 📋 AI 指令文档：`docs/ai_instructions/W4A_BENCHMARK_V5_STANDARD_TAIL_SPILL.md`

### 9.1 核心思路

V5 使用 `GPTQTailSpill` 类（位于 `gptq_tail_spill.py`），其核心机制为：

- **Tail 列跳过量化，保留浮点值**：在 GPTQ 的 `fasterquant` 逐列迭代中，当列索引 >= `tail_start` 时，跳过量化步骤，直接保留原始浮点权重。
- **误差不传播**：由于 tail 列未被量化，不产生量化误差，因此不会通过 Hessian 补偿传播到后续列。
- **使用标准 min/max Quantizer**：通过 `--use-standard-quantizer` 参数切换为标准量化器（而非 Percentile），以隔离 quantizer 类型对结果的影响。

### 9.2 实验脚本

- 量化入口：`qwen3_gptq_percentile_tail_spill.py`
- GPTQ 核心：`gptq_tail_spill.py`（`GPTQTailSpill` 类）

### 9.3 实验变体

3 种 tail_rank 配置，均从原始 FP16 权重出发：

| 变体 | tail_rank | Quantizer | 权重来源 |
|------|-----------|-----------|---------|
| Std Tail Spill (raw, r=16) | 16 | 标准 min/max | FP16 原始 |
| Std Tail Spill (raw, r=64) | 64 | 标准 min/max | FP16 原始 |
| Std Tail Spill (raw, r=128) | 128 | 标准 min/max | FP16 原始 |

> ⚠️ V5 实验未进行 PPL 评测，PPL 数据不可用。

---

## 10. V6 — Smooth + Standard Tail Spill

> 📋 AI 指令文档：`docs/ai_instructions/W4A_BENCHMARK_V6_SMOOTH_STANDARD_TAIL_SPILL.md`

### 10.1 核心思路

V6 在 V5 的基础上加入 SmoothQuant 预处理：先对权重做 Smooth（alpha=1），再执行标准 Tail Spill 量化。目的是验证 SmoothQuant + Tail Spill 的组合效果。

### 10.2 实验变体与 PPL 结果

3 种 tail_rank 配置，从 smooth 后的权重出发：

| 变体 | tail_rank | ActQ=none | ActQ=int8 | ActQ=int4-g128 | ActQ=int4-g128+down:int8 |
|------|-----------|-----------|-----------|----------------|--------------------------|
| smooth_stdts_r16 | 16 | 10.4404 | 10.7206 | 13.3974 | 12.7426 |
| smooth_stdts_r64 | 64 | 10.6128 | 10.9005 | 13.5311 | 12.8757 |
| smooth_stdts_r128 | 128 | 11.1749 | 11.5178 | 14.7505 | 14.0515 |

> 数据来源：`output/benchmark/results_smooth_stdts.txt`

### 10.3 ⚠️ 发现 rank-PPL 反转 bug

**现象**：rank 越大，PPL 越差（与预期完全相反）：

- r16 = 10.4404 → r64 = 10.6128 → r128 = 11.1749（Anone 配置下）

**根因分析**：

`GPTQTailSpill` 的 tail 列**不传播误差**。当 rank 增大时，更多列被划入 tail 区域，这些列脱离了 GPTQ 的 Hessian 误差补偿链。rank 越大 → 越多列无法参与误差补偿 → 整体量化质量下降。

**结论**：Tail Spill 的"不传播误差"设计存在根本缺陷，需要改用 Tail Absorb（误差正常传播）来修复。

---

## 11. V7 — Smooth + Tail Absorb（修复误差传播 + act-order 消融）

> 📋 AI 指令文档：`docs/ai_instructions/W4A_BENCHMARK_V7_SMOOTH_TAIL_ABSORB.md`
> 📊 完整分析报告：`V7_tail_absorb_analysis.md`

### 11.1 修复方案

V7 使用 `GPTQTailAbsorb`（来自 `gptq_tail_absorb.py`）替代 `GPTQTailSpill`，核心改进：

- Tail 列执行 **INT8 fake-quant**（量化后立即反量化），量化误差**正常传播**到后续列
- Main 列使用**标准 min/max Quantizer**（与 V5/V6 保持一致，隔离 quantizer 变量）
- 权重来源：SmoothQuant 预处理后的权重

### 11.2 act-order 消融实验设计

为每种 rank 配置分别运行 act-order ON（默认）和 act-order OFF（`--no-act-order`），共 6 个变体：

| 变体 | tail_rank | act-order | 标签 |
|------|-----------|-----------|------|
| smooth_ta_r16 | 16 | ON | V7-1 |
| smooth_ta_r64 | 64 | ON | V7-2 |
| smooth_ta_r128 | 128 | ON | V7-3 |
| smooth_ta_r16_noact | 16 | OFF | V7-4 |
| smooth_ta_r64_noact | 64 | OFF | V7-5 |
| smooth_ta_r128_noact | 128 | OFF | V7-6 |

### 11.3 PPL 结果

**act-order ON：**

| 变体 | ActQ=none | ActQ=int8 | ActQ=int4-g128 | ActQ=int4-g128+down:int8 |
|------|-----------|-----------|----------------|--------------------------|
| smooth_ta_r16 | 10.3428 | 10.6156 | 13.2699 | 12.6241 |
| smooth_ta_r64 | 10.4042 | 10.6748 | 13.3250 | 12.6773 |
| smooth_ta_r128 | 10.4184 | 10.6897 | 13.3127 | 12.6651 |

**act-order OFF：**

| 变体 | ActQ=none | ActQ=int8 | ActQ=int4-g128 | ActQ=int4-g128+down:int8 |
|------|-----------|-----------|----------------|--------------------------|
| smooth_ta_r16_noact | 10.3846 | 10.6643 | 13.1251 | 12.5639 |
| smooth_ta_r64_noact | 10.3909 | 10.6664 | 13.1782 | 12.6008 |
| smooth_ta_r128_noact | 10.3949 | 10.6680 | 13.2016 | 12.6254 |

> 数据来源：`output/benchmark/results_smooth_tail_absorb.txt`

### 11.4 核心结论

1. **✅ 误差传播修复成功**：V6 的 rank-PPL 反转问题已消除。V7 中 r128 (10.42) 与 r16 (10.34) 差距极小，而 V6 中 r128 (11.17) 远差于 r16 (10.44)。

2. **⚠️ rank 参数影响极小**：r16 / r64 / r128 之间的 PPL 差异仅 0.01-0.08，r16 在所有配置下均为最优。rank 增大并未带来预期的改善。

3. **⚠️ act-order OFF 在多数场景略优**：12 组对比中有 9 组 act-order OFF 更好，但差异很小（最大 0.15），说明 act-order 对 Tail Absorb 的影响有限。

4. **⭐ 最佳配置**：`smooth_ta_r16` + act-order ON，PPL = 10.3428，距 FP16 基线 (10.0449) 仅 +0.2979。

---

## 12. 全局 PPL 汇总表

以下为 9 种权重配置 × 4 种激活量化的完整 PPL 数据（WikiText-2）：

| 权重变体 | ActQ=none | ActQ=int8 | ActQ=int4-g128 | ActQ=int4-g128+down:int8 |
|----------|-----------|-----------|----------------|--------------------------|
| **FP16 baseline** | 10.0449 | 10.3204 | 14.1556 | 13.4046 |
| **GPTQ 4bit (from raw)** | 10.3845 | 10.7033 | 15.3095 | 14.3323 |
| **Smooth + GPTQ 4bit** | 10.8361 | 11.1372 | 14.1750 | 13.4645 |
| **smooth_ta_r16 (act ON)** | **10.3428** | 10.6156 | 13.2699 | 12.6241 |
| **smooth_ta_r64 (act ON)** | 10.4042 | 10.6748 | 13.3250 | 12.6773 |
| **smooth_ta_r128 (act ON)** | 10.4184 | 10.6897 | 13.3127 | 12.6651 |
| **smooth_ta_r16_noact (act OFF)** | 10.3846 | 10.6643 | **13.1251** | **12.5639** |
| **smooth_ta_r64_noact (act OFF)** | 10.3909 | 10.6664 | 13.1782 | 12.6008 |
| **smooth_ta_r128_noact (act OFF)** | 10.3949 | 10.6680 | 13.2016 | 12.6254 |

> 数据来源：`V7_summary_table.txt`、`output/benchmark/results_smooth_tail_absorb.txt`

**各列最优值（不含 FP16）：**

| 激活量化配置 | 最优变体 | PPL | act-order |
|-------------|---------|-----|-----------|
| ActQ=none | smooth_ta_r16 | 10.3428 | ON |
| ActQ=int8 | smooth_ta_r16 | 10.6156 | ON |
| ActQ=int4-g128 | smooth_ta_r16_noact | 13.1251 | OFF |
| ActQ=int4-g128+down:int8 | smooth_ta_r16_noact | 12.5639 | OFF |

**全局最优量化配置**：`smooth_ta_r16` (act-order ON), ActQ=none → PPL = 10.3428（距 FP16 仅 +0.2979）

---

## 13. 更新后的状态与后续建议

### 13.1 当前状态

- **V1-V3**（TAG1 里程碑）：✅ 已完成。基线 GPTQ、SmoothQuant + GPTQ、Percentile + Tail 原型均已实现并验证。
- **V4**（Tail Absorb 原型）：✅ 已完成。引入 `GPTQTailAbsorb` 类，验证误差正常传播机制。
- **V5**（Standard Tail Spill from raw）：✅ 已完成。切换为标准 Quantizer，建立对照实验基线。
- **V6**（Smooth + Tail Spill）：✅ 已完成。发现并定位 rank-PPL 反转 bug。
- **V7**（Smooth + Tail Absorb + act-order 消融）：✅ 已完成。修复误差传播问题，完成全面评测。

### 13.2 后续可能方向

1. **混合精度部署方案**：将 Tail Absorb 的 main (INT4) + tail (INT8) 混合精度方案适配到推理框架（如 vLLM、TensorRT-LLM）。
2. **更大模型验证**：在 Qwen3-7B / 14B 等更大模型上验证 Tail Absorb 的泛化性。
3. **Tail Rank 自适应**：探索按层自适应选择 tail_rank（而非全局固定值）。
4. **与其他量化方法对比**：与 AWQ、AQLM、QuIP# 等方法进行横向对比。
5. **下游任务评测**：在 MMLU、GSM8K 等下游任务上评估量化模型的实际能力。
