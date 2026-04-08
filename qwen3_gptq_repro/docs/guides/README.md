# Qwen3-4B 量化运行文档导航

本文件是项目文档的总导航页。所有文档已按作用分类存放在 `docs/` 目录下。

---

## 快速入口

- 单独 GPTQ（原始权重直量化）：[LEARN_AND_RUN_GPTQ_ONLY.md](./LEARN_AND_RUN_GPTQ_ONLY.md)
- Smooth + GPTQ（先 smooth 再量化）：[LEARN_AND_RUN_SMOOTH_GPTQ.md](./LEARN_AND_RUN_SMOOTH_GPTQ.md)
- PPL 评测指南：[BENCHMARK_GUIDE.md](./BENCHMARK_GUIDE.md)

推荐目录约定（避免覆盖）：

- `output/gptq_from_raw/`：原始权重直做 GPTQ
- `output/smooth/`：Smooth 中间产物
- `output/gptq_from_smooth/`：Smooth 后再 GPTQ

如果你只想跑之前的原始 GPTQ，请直接看 [LEARN_AND_RUN_GPTQ_ONLY.md](./LEARN_AND_RUN_GPTQ_ONLY.md)。
如果你要跑当前实验（Smooth + GPTQ），请看 [LEARN_AND_RUN_SMOOTH_GPTQ.md](./LEARN_AND_RUN_SMOOTH_GPTQ.md)，其中已单列"调用原始 GPTQ 的注意事项"。

---

## 📂 完整文档分类索引

### 📖 操作手册 (`docs/guides/`)

给人看的教程和操作步骤。

| 文件 | 说明 |
|------|------|
| [README.md](./README.md) | 本导航页 |
| [LEARN_AND_RUN_GPTQ_ONLY.md](./LEARN_AND_RUN_GPTQ_ONLY.md) | GPTQ 量化的完整操作步骤（环境、命令、验证） |
| [LEARN_AND_RUN_SMOOTH_GPTQ.md](./LEARN_AND_RUN_SMOOTH_GPTQ.md) | Smooth+GPTQ 的完整操作步骤 |
| [BENCHMARK_GUIDE.md](./BENCHMARK_GUIDE.md) | PPL 评测方法论 + 各变体运行命令 + 结果解读指南 |

### 📝 分析/总结 (`docs/analysis/`)

给读者看的技术分析和总结文档。

| 文件 | 说明 |
|------|------|
| [WEIGHT_VARIANTS_OVERVIEW.md](../analysis/WEIGHT_VARIANTS_OVERVIEW.md) | 10 个权重变体的完整技术总览（架构、分类、关系图） |
| [SMOOTH_IMPL_REVIEW.md](../analysis/SMOOTH_IMPL_REVIEW.md) | Smooth 实现的代码审查结论 |
| [TAG1_MILESTONE_SUMMARY.md](../analysis/TAG1_MILESTONE_SUMMARY.md) | 阶段性实验设计与进展总结（TAG1 里程碑） |
| [ALL_WORK_SUMMARY.md](../analysis/ALL_WORK_SUMMARY.md) | 全量工作总结（三条主线） |
| [PERCENTILE_TAIL_METHOD.md](../analysis/PERCENTILE_TAIL_METHOD.md) | Percentile Tail 方法说明（原型阶段） |

### 📋 实验记录 (`docs/experiment_logs/`)

运行命令、结果和变更日志。

| 文件 | 说明 |
|------|------|
| [TAIL_MIXED_RECORD.md](../experiment_logs/TAIL_MIXED_RECORD.md) | Tail Mixed 实验完整记录（计划+变更+结果） |
| [PERCENTILE_TAIL_RUNLOG.md](../experiment_logs/PERCENTILE_TAIL_RUNLOG.md) | Percentile Tail 运行记录（命令+产物） |
| [PERCENTILE_TAIL_SANITY_SUMMARY.md](../experiment_logs/PERCENTILE_TAIL_SANITY_SUMMARY.md) | Percentile Tail Sanity 检查结果汇总 |

### 🤖 AI 指令 (`docs/ai_instructions/`)

给 AI 智能体执行的指令文件。

| 文件 | 状态 | 说明 |
|------|------|------|
| [W4A_BENCHMARK_V2.md](../ai_instructions/W4A_BENCHMARK_V2.md) | ✅ 当前有效 | V2 版 20 个实验的完整执行指令 |
| [W4A4_BENCHMARK_UPGRADE.md](../ai_instructions/_deprecated/W4A4_BENCHMARK_UPGRADE.md) | ⚠️ 已过时 | V1 版，已被 V2 取代 |
