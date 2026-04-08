# TAG1：阶段性实验设计与进展总结

## 1. 本阶段总目标

本阶段围绕 Qwen3-4B 的量化复现与改进，主要做了三条技术线：

1. 原始 GPTQ 4-bit 基线复现  
2. SmoothQuant（alpha=1）预处理后再 GPTQ  
3. 尾部列混合精度（tail mixed）方案设计与首次实现（main 4-bit GPTQ + tail INT8）

目标是逐步建立可复现的对比框架，最终用于判断哪种方案在误差控制和成本之间更优。

---

## 2. 文档与工程组织设计（已完成）

为避免流程混杂，已做文档拆分与目录约定：

- 总导航：`LEARN_AND_RUN.md`
- 单独 GPTQ：`LEARN_AND_RUN_GPTQ_ONLY.md`
- Smooth + GPTQ：`LEARN_AND_RUN_SMOOTH_GPTQ.md`
- Smooth 实现审查：`SMOOTH_IMPL_REVIEW.md`
- 三模型统一推理脚本：`run_three_infer.sh`
- Tail mixed 专项文档目录：`MD/`
  - `TAIL_MIXED_EXPERIMENT_PLAN.md`
  - `TAIL_MIXED_CHANGELOG.md`
  - `TAIL_MIXED_RESULTS.md`

输出目录按实验路径隔离：

- `output/gptq_from_raw/`
- `output/smooth/`
- `output/gptq_from_smooth/`
- `output/exp_tail_mixed/baseline_gptq4/`
- `output/exp_tail_mixed/tail5p_int8_gptq4/`

---

## 3. 实验一：原始 GPTQ 4-bit（基线）

### 3.1 设计目的

建立可复现 baseline，作为后续 Smooth 和 Tail-mixed 的对照。

### 3.2 关键实现

- 脚本：`qwen3_gptq.py`
- 核心参数：`wbits=4`、`act-order`、`true-sequential`
- 校准集策略：优先 WikiText2，失败时自动 fallback

### 3.3 额外工程增强

在 `qwen3_gptq.py` 增加了“可从外部 state_dict 初始化”的入口：

- `--init-state-dict`
- metadata 中记录 `init_state_dict` 来源信息

这为 “Smooth 后继续 GPTQ” 提供了统一接口，而不破坏原有 baseline 路径。

---

## 4. 实验二：Smooth(alpha=1) + GPTQ

### 4.1 设计目的

先通过 SmoothQuant 把激活离群值迁移到权重/scale 空间，再执行 GPTQ，观察是否更稳。

### 4.2 核心设计

- 新增独立脚本：`qwen3_smooth.py`（不与 GPTQ 主流程耦合）
- 固定 `alpha=1.0`
- 针对每层：
  - 收集 RMSNorm 输出通道统计（amax）
  - 对应做 `norm.weight /= s`
  - 对下游线性层做 `W *= s`

映射关系：

- `input_layernorm -> (q_proj, k_proj, v_proj)`
- `post_attention_layernorm -> (gate_proj, up_proj)`

### 4.3 产物设计

- `smoothed_model_state_dict.pt`
- `smooth_scales.pt`
- `smooth_metadata.json`

### 4.4 与 GPTQ 衔接

通过 `qwen3_gptq.py --init-state-dict <smoothed_model_state_dict.pt>` 直接进入 GPTQ 量化。

---

## 5. 实验三：尾部列混合精度（Tail Mixed）

### 5.1 背景想法

思路是把“尾部列”当作 residual buffer：  
主体列高压缩（4-bit GPTQ），尾部少量列用更高精度，减少误差持续放大。

本轮选择：

- tail 比例：5%
- tail 精度：INT8（可选项）
- 预算：`nsamples=32, seqlen=1024`

### 5.2 代码隔离策略（已按要求完成）

- 保持原脚本不动：`qwen3_gptq.py`
- 单独复制并实验：`qwen3_gptq_tail_mixed.py`

### 5.3 实验脚本设计（已实现）

新增参数：

- `--enable-tail-mixed`
- `--tail-ratio`
- `--tail-quant int8`

对每个线性层输入维做列切分：

- `W_main`：前 95% 列，继续 4-bit GPTQ
- `W_tail`：后 5% 列，执行 INT8 量化再反量化写回（保留高于 4-bit 的表达能力）

metadata 中新增：

- `tail_mixed.enabled / tail_ratio / tail_quant`
- 每层尾部区间（tail_start, tail_cols）
- 每层 tail 量化 scale 与平均绝对误差统计

---

## 6. 当前运行状态（根据现有产物核对）

### 6.1 已完成

1. Baseline（中等预算）已完成  
   - 目录：`output/exp_tail_mixed/baseline_gptq4/`
   - metadata 显示：`nsamples=32, seqlen=1024, elapsed_seconds≈161.50`

2. Tail5%+INT8（中等预算）已有完整产物与 metadata  
   - 目录：`output/exp_tail_mixed/tail5p_int8_gptq4/`
   - metadata 显示：`tail_mixed.enabled=true`，`num_layers_with_tail_mixed=252`，`elapsed_seconds≈160.98`

3. 三权重对比入口已具备  
   - 脚本：`run_three_infer.sh`
   - 可统一跑 smooth / gptq_from_smooth / gptq_from_raw 三份权重推理

### 6.2 仍需补齐（建议下一步）

1. 把 Tail mixed 的实际运行命令与结果回填到 `MD/TAIL_MIXED_RESULTS.md`（目前仍有 Pending 标记）  
2. 统一 prompt 集做文本对比（建议固定 10~20 条）  
3. 增加统一损失/困惑度对比，形成可量化结论  

---

## 7. 你可以据此做的下一步决策

### 选项 A（快速收敛）

先不继续扩展算法，直接完善评测：

- 固定 baseline vs tail5%+INT8
- 做文本输出 + perplexity 对比
- 得到是否“值得继续”的一版结论

### 选项 B（继续扩展设计空间）

若 tail5%+INT8 有正向趋势，再扩展小网格：

- tail_ratio: 1% / 2% / 5%
- tail_quant: INT8（必要时对比 FP16）

---

## 8. 一句话阶段结论

本阶段已经从“单一路径 GPTQ 复现”升级为“多路径可对比实验框架”，并完成了 tail mixed 的独立代码实现与首次产物落地；下一步重点应从“继续写代码”转到“统一评测与结果归纳”。
