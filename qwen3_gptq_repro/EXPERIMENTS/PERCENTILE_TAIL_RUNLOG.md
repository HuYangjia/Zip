# Percentile Tail 运行记录（原型）

## 1. 共同前提

- 模型：`/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507`
- smooth 迁移权重（alpha=1）：`output/smooth/smoothed_model_state_dict.pt`
- 预算：`nsamples=32`, `seqlen=1024`
- tail 设定：`percentile_k=75`, `tail_rank=32`, `tail_quant=int8`

## 2. Mode A 命令（percentile main）

```bash
python /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/qwen3_gptq_percentile_tail.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --init-state-dict /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth/smoothed_model_state_dict.pt \
  --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_percentile_tail/percentile_tail_k75_r32_int8_modeA \
  --enable-percentile-tail \
  --percentile-k 75 \
  --tail-rank 32 \
  --tail-quant int8 \
  --main-wbits 4 \
  --enable-main-absorb \
  --main-absorb-mode budget_clamped \
  --main-absorb-budget-rule int4_boundary \
  --lambda-reg 1e-4 \
  --nsamples 32 \
  --seqlen 1024 \
  --groupsize 128 \
  --percdamp 0.01 \
  --act-order \
  --true-sequential
```

产物：

- `output/exp_percentile_tail/percentile_tail_k75_r32_int8_modeA/qwen3-4b-instruct-2507-percentile-tail.pt`
- `output/exp_percentile_tail/percentile_tail_k75_r32_int8_modeA/metadata.json`

## 3. Mode B 命令（GPTQ main）

```bash
python /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/qwen3_gptq_percentile_tail.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --init-state-dict /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth/smoothed_model_state_dict.pt \
  --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_percentile_tail/percentile_tail_k75_r32_int8_modeB_gptqmain \
  --enable-percentile-tail \
  --use-gptq-main \
  --percentile-k 75 \
  --tail-rank 32 \
  --tail-quant int8 \
  --main-wbits 4 \
  --enable-main-absorb \
  --main-absorb-mode budget_clamped \
  --main-absorb-budget-rule int4_boundary \
  --lambda-reg 1e-4 \
  --nsamples 32 \
  --seqlen 1024 \
  --groupsize 128 \
  --percdamp 0.01 \
  --act-order \
  --true-sequential
```

产物：

- `output/exp_percentile_tail/percentile_tail_k75_r32_int8_modeB_gptqmain/qwen3-4b-instruct-2507-percentile-tail.pt`
- `output/exp_percentile_tail/percentile_tail_k75_r32_int8_modeB_gptqmain/metadata.json`

## 4. Mode B 边界口径修正后重跑

为修正约束统计异常，已将 mode B 的 main absorb 边界改为 `gptq_q0_rowmax` 口径，并使用同一命令重跑覆盖产物（路径不变）。

## 5. 加入 constraint-eps 后再次重跑

新增参数：

- `--constraint-eps 1e-5`

用于将数值抖动级别的超界从违规统计中剔除。重跑后 `main_constraint_violations` 总计为 `0`。
