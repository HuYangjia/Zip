# Percentile Tail 方法说明（原型阶段）

## 1. 目标

实现并验证以下链路是否可运行且日志可解释：

`main 激进量化 -> main 内部受约束吸收 -> tail overflow 吸收`

当前阶段只做方法原型，不做端到端评测。

## 2. 代码隔离

- 原脚本保持不改：`qwen3_gptq.py`
- 新增实验脚本：`qwen3_gptq_percentile_tail.py`

## 3. 模式定义

- `mode A`（`--use-gptq-main` 不开启）  
  `main` 使用 percentile uniform 量化，再做 main absorb，再 spill 到 tail。

- `mode B`（`--use-gptq-main` 开启）  
  `main` 先走 GPTQ，再做 main absorb，然后 spill 到 tail。

## 4. 当前实现要点

1. 支持按输入列切分 main/tail，优先 `tail_rank`，否则用 `tail_ratio` 计算。  
2. `q_max` 不写死，来自 `main_wbits`。  
3. main absorb 采用 `budget_clamped`：  
   - 预算 `b = s_main * q_max - |w_main_q0|`
   - 吸收量 `delta = clamp(residual, [-b, b])`
4. tail compensation 使用校准统计近似实现：
   - 使用输入摘要构造 `H_tt` 与 `E_spill * H_mt`
   - 求解 `(H_tt + lambda I)^-1` 的稳定版本
5. tail 高精量化当前支持 `int8`；`int4_fp4` 预留接口（未实现会明确报错）。

## 5. Sanity 相关输出

脚本会输出：

- 全局配置（`metadata.json`）
- 每层统计（`metadata.layer_stats`）
- 三段 residual 相关字段（before/after/spill + final residual）
- main 边界约束违规计数
- `H_tt` 条件数估计与实际 `lambda_used`

补充：在 mode B（GPTQ main）中，main 约束边界当前采用 `gptq_q0_rowmax` 口径（按行最大绝对值推导边界），用于避免 GPTQ 主量化与 percentile 边界直接混用造成的伪违规。

另外，已增加违规明细字段：

- `main_constraint_violation_positions_preview`
- `main_constraint_violation_overflow_max`

用于快速定位剩余越界点。

同时支持容差参数：

- `--constraint-eps`（默认 `1e-5`）

用于把浮点数抖动造成的近零超界从硬违规统计中排除。

## 6. 阶段边界

本阶段明确不包含：

- perplexity
- 文本生成质量比较
- 统一 prompt 评测
- benchmark 总表
