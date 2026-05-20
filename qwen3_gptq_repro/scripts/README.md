# SmoothBlock Run Scripts

这个目录里的两个脚本用于批量运行 `qwen3_smooth_block_mixed.py`。它们都会先检查模型目录, 尝试激活 conda 环境 `zip`, 然后在 repo 根目录下依次跑 3 组 block-row 配置。

被调用的主程序:

```bash
qwen3_smooth_block_mixed.py
```

每个输出目录通常会包含:

- `qwen3-smooth-block-mixed-state_dict.pt`: SmoothBlock/GPTQ 后的模型权重。
- `smooth_groups.pt`: 每个模块/组搜索到的 smooth scale 和相关信息, 后续 Hessian 诊断脚本会读取它。
- `qwen3_smooth_block_mixed_metadata.json`: 搜索配置和结果摘要, 可用 `scripts/analyze_output_error_losses.py` 继续整理 loss 报告。
- `layer_output_errors.json`: 因为两个脚本都启用了 `--save-layer-output-errors`, 会保存 layer/module 级 output error 信息。

## `qwen3-8b.sh`

用途: 对 `/root/autodl-tmp/model/Qwen3-8B` 跑 Smooth Block Mixed V1-V3。

运行命令:

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro
bash scripts/qwen3-8b.sh
```

实际运行的 3 组配置:

| block rows | block cols | groupsize | 输出目录 |
|---:|---:|---:|---|
| 16 | 128 | 32 | `output/qwen3_8b_v1` |
| 32 | 128 | 32 | `output/qwen3_8b_v2` |
| 64 | 128 | 32 | `output/qwen3_8b_v3` |

主要固定参数:

- `--model-dir /root/autodl-tmp/model/Qwen3-8B`
- `--second-path residual_int4`
- `--act-order`
- `--percdamp 0.01`
- `--budget-ratio 0.2`
- `--save-layer-output-errors`

可用环境变量覆盖的参数:

```bash
NSAMPLES=64 \
SEQLEN=1024 \
MAX_SEARCH_BATCHES=64 \
SEARCH_EVAL_BATCH_SIZE=1 \
GPTQ_BATCH_SIZE=1 \
OUTPUT_ERROR_BATCH_SIZE=1 \
bash scripts/qwen3-8b.sh
```

日志输出:

- 主日志: `scripts/logs/qwen3_8b_v1_v3/main_<timestamp>.log`
- 每组子日志:
  - `scripts/logs/qwen3_8b_v1_v3/qwen3_8b_v1_block_rows_16.log`
  - `scripts/logs/qwen3_8b_v1_v3/qwen3_8b_v2_block_rows_32.log`
  - `scripts/logs/qwen3_8b_v1_v3/qwen3_8b_v3_block_rows_64.log`

## `run_smooth_v15_v17.sh`

用途: 对 `/root/autodl-tmp/model/Qwen3-4B-Instruct-2507` 跑 Smooth Block Mixed V15-V17。

运行命令:

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro
bash scripts/run_smooth_v15_v17.sh
```

实际运行的 3 组配置:

| block rows | block cols | groupsize | 输出目录 |
|---:|---:|---:|---|
| 16 | 128 | 64 | `output/smooth_v15_b32` |
| 32 | 128 | 64 | `output/smooth_v16_b32` |
| 64 | 128 | 64 | `output/smooth_v17_b32` |

主要固定参数:

- `--model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507`
- `--nsamples 128`
- `--seqlen 1024`
- `--second-path residual_int4`
- `--act-order`
- `--percdamp 0.01`
- `--budget-ratio 0.2`
- `--max-search-batches 128`
- `--save-layer-output-errors`

日志输出:

- 主日志: `scripts/logs/smooth_v15_v17/main_<timestamp>.log`
- 每组子日志:
  - `scripts/logs/smooth_v15_v17/smooth_v15_b32_block_rows_16.log`
  - `scripts/logs/smooth_v15_v17/smooth_v16_b32_block_rows_32.log`
  - `scripts/logs/smooth_v15_v17/smooth_v17_b32_block_rows_64.log`

## 后续常用分析

查看某个输出目录的 loss 汇总:

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro
python scripts/analyze_output_error_losses.py \
  --metadata output/smooth_v16_b32/qwen3_smooth_block_mixed_metadata.json \
  --output output/smooth_v16_b32/output_error_loss_analysis.xlsx \
  --csv-dir output/smooth_v16_b32/output_error_loss_csv
```

用 `smooth_groups.pt` 做 Hessian-sensitive block 诊断:

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro
bash scripts/run_prove_hessian_raw_heatmaps.sh
```
