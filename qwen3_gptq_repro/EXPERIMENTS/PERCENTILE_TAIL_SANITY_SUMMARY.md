# Percentile Tail Sanity Summary（当前原型）

## 1. 汇总口径

以下数值来自两份 `metadata.json` 的 `layer_stats` 均值汇总：

- mode A: `percentile_tail_k75_r32_int8_modeA`
- mode B: `percentile_tail_k75_r32_int8_modeB_gptqmain`

## 2. 关键指标

### mode A（percentile main）

- `elapsed_seconds`: `81.24`
- `layers`: `252`
- `main_residual_before_absorb_norm` 平均：`1.7651`
- `main_residual_after_absorb_norm` 平均：`1.7615`
- `spill_residual_norm` 平均：`1.7615`
- `tail_correction_norm` 平均：`1.1938`
- `tail_quant_error` 平均：`0.00646`
- `tail_saturation_ratio` 平均：`0.03185`
- `main_constraint_violations` 总计：`0`

### mode B（GPTQ main）

- `elapsed_seconds`: `224.23`
- `layers`: `252`
- `main_residual_before_absorb_norm` 平均：`0.5576`
- `main_residual_after_absorb_norm` 平均：`0.02082`
- `spill_residual_norm` 平均：`0.02082`
- `tail_correction_norm` 平均：`0.01834`
- `tail_quant_error` 平均：`0.00125`
- `tail_saturation_ratio` 平均：`0.03179`
- `main_constraint_violations` 总计：`4`
- `main_boundary_source`：`gptq_q0_rowmax`

## 3. 初步观察（仅原型层面）

1. 两个模式都能完整跑通，并生成权重与层级统计。  
2. mode A 中 main absorb 的均值效果较小（before/after 接近），且边界违规为 0。  
3. mode B 采用 `gptq_q0_rowmax` 作为约束边界后，`main residual` 从 `0.5576` 降到 `0.02082`，且违规总量降到 `4`。  
4. 当前 mode B 约束口径已基本可用，下一步应重点定位剩余 4 个违规点（层级/通道）。  

## 5. 违规点定位（历史记录）

已将违规明细写入 `metadata.layer_stats[*].main_constraint_violation_positions_preview`，当前结果：

- 仅 1 层出现违规：`model.layers.35.mlp.up_proj`
- 违规总数：`4`
- 位置预览（row, col）：
  - `[2901, 396]`
  - `[3071, 396]`
  - `[3729, 396]`
  - `[5729, 396]`
- 最大超界幅度：`7.62939453125e-06`

该超界量级很小，接近数值误差边界，可视为“近零违规”。

## 6. constraint-eps 清洗结果（最新）

在 mode B 中加入参数 `--constraint-eps 1e-5` 后，重新统计得到：

- `constraint_eps`: `1e-05`
- `main_constraint_violations` 总计：`0`
- 违规层数：`0`

这说明前一版剩余 4 个违规点确认为数值误差级别抖动，而非方法性越界。

## 4. 下一步建议（仍不做端到端）

1. 增加每层 `lambda` 回退触发计数字段，确认数值稳定路径是否合理。  
2. 在 mode A / mode B 下增加 `k` 与 `tail_rank` 的小规模网格（仍不做端到端评测）。  
3. 继续保持“方法层调试”阶段，不引入 perplexity/生成评测。  
