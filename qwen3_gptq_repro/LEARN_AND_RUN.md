# Qwen3-4B 量化运行文档导航

本文件只做导航，具体流程已拆分为两份独立文档：

- 单独 GPTQ（原始权重直量化）：`LEARN_AND_RUN_GPTQ_ONLY.md`
- Smooth + GPTQ（先 smooth 再量化）：`LEARN_AND_RUN_SMOOTH_GPTQ.md`

推荐目录约定（避免覆盖）：

- `output/gptq_from_raw/`：原始权重直做 GPTQ
- `output/smooth/`：Smooth 中间产物
- `output/gptq_from_smooth/`：Smooth 后再 GPTQ

如果你只想跑之前的原始 GPTQ，请直接看 `LEARN_AND_RUN_GPTQ_ONLY.md`。  
如果你要跑当前实验（Smooth + GPTQ），请看 `LEARN_AND_RUN_SMOOTH_GPTQ.md`，其中已单列“调用原始 GPTQ 的注意事项”。
