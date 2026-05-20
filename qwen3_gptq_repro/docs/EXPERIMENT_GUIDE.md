# Qwen3 Smooth Block Mixed 实验指南

本文档按一次完整实验的顺序说明：

1. 下载 Qwen3 模型。
2. 运行 `qwen3_smooth_block_mixed.py` 做 Smooth + Block Mixed Fake 量化。
3. 使用 `benchmark/eval_ppl.py` 测 WikiText-2 PPL。
4. 使用 `benchmark/eval_lm_eval.py` 测 LM-Eval zero-shot 任务。

下面所有命令默认在项目目录执行：

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro
```

---

## 1. 环境准备

建议使用 Python 3.10 环境：

```bash
conda create -n qwen3-gptq python=3.10 -y
conda activate qwen3-gptq
```

安装基础依赖：

```bash
pip install torch transformers datasets sentencepiece accelerate huggingface_hub openpyxl
```

如果要运行 `benchmark/eval_lm_eval.py`，还需要安装 LM-Eval Harness：

```bash
pip install "lm_eval[hf]"
```

如果使用 ModelScope 下载模型，可以额外安装：

```bash
pip install modelscope
```

---

## 2. 下载模型

### 2.1 下载 Qwen3-4B-Instruct-2507

仓库里已有 HuggingFace 下载脚本：

```bash
cd /root/autodl-tmp/Zip
python download/dl_qwen3_4b.py
```

该脚本会下载：

```text
Qwen/Qwen3-4B-Instruct-2507
```

默认保存到：

```text
/root/autodl-tmp/Zip/models/Qwen3-4B-Instruct-2507
```

脚本默认使用 `HF_ENDPOINT=https://hf-mirror.com`。如果你想使用官方 HuggingFace，可以这样运行：

```bash
HF_ENDPOINT=https://huggingface.co python download/dl_qwen3_4b.py
```

后续命令中把模型路径设置为：

```bash
export MODEL_DIR=/root/autodl-tmp/Zip/models/Qwen3-4B-Instruct-2507
```

如果你的模型已经放在别处，例如：

```text
/root/autodl-tmp/model/Qwen3-4B-Instruct-2507
```

则直接使用：

```bash
export MODEL_DIR=/root/autodl-tmp/model/Qwen3-4B-Instruct-2507
```

### 2.2 可选：下载 Qwen3-8B

另一个下载脚本是：

```bash
cd /root/autodl-tmp/Zip
python download/dl_modelscope_qwen3.py
```

它通过 ModelScope 下载：

```text
Qwen/Qwen3-8B
```

默认保存到：

```text
/root/autodl-tmp/models/Qwen3-8B
```

本文下面的例子使用 Qwen3-4B-Instruct-2507。

---

## 3. Smooth Block Mixed Fake 量化

主脚本：

```text
/root/autodl-tmp/Zip/qwen3_gptq_repro/qwen3_smooth_block_mixed.py
```

这个脚本会完成一条完整 pipeline：

1. 加载 Qwen3 原始模型。
2. 用 WikiText-2 或本地数据做校准。
3. 为不同模块组搜索 SmoothQuant alpha。
4. 对选中的模块组 fuse smooth scale。
5. 做 block mask 搜索。
6. 执行 GPTQ 风格的 W4 Fake 量化。
7. 可选保存每个 Linear 的 output error 诊断。

这里的“Fake 量化”指：最终保存的是普通 PyTorch `state_dict`，权重仍是浮点 tensor，但数值已经经过 quantize-dequantize 处理，因此可以直接用原始 HuggingFace 模型结构加载评测。

### 3.1 运行示例

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro

export MODEL_DIR=/root/autodl-tmp/Zip/models/Qwen3-4B-Instruct-2507
export OUT_DIR=/root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_block_mixed_example

python qwen3_smooth_block_mixed.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUT_DIR" \
  --nsamples 128 \
  --seqlen 1024 \
  --block-rows 16 \
  --block-cols 128 \
  --groupsize 64 \
  --budget-ratio 0.2 \
  --second-path residual_int4 \
  --act-order \
  --percdamp 0.01 \
  --max-search-batches 128 \
  --save-layer-output-errors
```

这是一组适合跑通流程的例子参数。更大的 `--block-rows` 会让 block 粒度更粗；`--groupsize` 可以取 `32`、`64`、`128`；`--budget-ratio` 控制 second path / mixed block 的预算比例。

### 3.2 常用参数说明

| 参数 | 作用 |
|---|---|
| `--model-dir` | 原始 HuggingFace Qwen3 模型目录。 |
| `--output-dir` | 量化结果、metadata、诊断文件保存目录。 |
| `--nsamples` | 校准样本数量。样本越多，Hessian / 搜索统计越稳定，但耗时更长。 |
| `--seqlen` | 校准序列长度。 |
| `--block-rows` | block mask 的行方向大小。 |
| `--block-cols` | block mask 的列方向大小。当前脚本固定要求为 `128`。 |
| `--groupsize` | 权重量化 group size，可选 `32`、`64`、`128`。 |
| `--budget-ratio` | 选中高精度/second path block 的预算比例，范围 `[0, 1]`。 |
| `--second-path` | 第二路径类型，可选 `residual_int4` 或 `int8`。 |
| `--act-order` | 启用 GPTQ activation-order。这个脚本默认关闭，需要显式传入。 |
| `--percdamp` | GPTQ Hessian damp 比例，常用 `0.01`。 |
| `--max-search-batches` | Smooth alpha 搜索、mask 搜索和 A4 recalibration 使用的最大 batch 数。 |
| `--save-layer-output-errors` | 保存每个 Linear 的 output error 诊断。 |
| `--local-wikitext2-dir` | 可选，本地 WikiText-2 数据目录，离线环境建议指定。 |
| `--dtype` | 模型加载精度，可选 `float16`、`bfloat16`、`float32`。 |

### 3.3 输出文件

运行完成后，`$OUT_DIR` 下主要有：

```text
qwen3-smooth-block-mixed-state_dict.pt
smooth_groups.pt
qwen3_smooth_block_mixed_metadata.json
layer_output_errors.json
```

含义如下：

| 文件 | 作用 |
|---|---|
| `qwen3-smooth-block-mixed-state_dict.pt` | Smooth + Block Mixed Fake 量化后的模型权重。后续 PPL 和 LM-Eval 都加载这个文件。 |
| `smooth_groups.pt` | 每个 smooth group 的 alpha、scale、mask 等信息。观察和诊断脚本会用到。 |
| `qwen3_smooth_block_mixed_metadata.json` | 本次运行的参数、校准数据来源、搜索摘要、最终量化摘要。 |
| `layer_output_errors.json` | 只有传入 `--save-layer-output-errors` 时生成，保存每个 Linear 的 output error / loss 诊断。 |
| `qwen3-smooth-block-mixed-results.pt` | 只有传入 `--save-full-results` 时生成，保存完整中间结果。 |

可以用下面脚本把 metadata 里的 layer loss 摘成表格：

```bash
python scripts/analyze_output_error_losses.py \
  --metadata "$OUT_DIR/qwen3_smooth_block_mixed_metadata.json" \
  --csv-dir "$OUT_DIR/loss_csv"
```

输出包括：

```text
output_error_losses.xlsx
loss_csv/loss.csv
loss_csv/loss_before_residual.csv
```

---

## 4. PPL 测评：benchmark/eval_ppl.py

脚本：

```text
/root/autodl-tmp/Zip/qwen3_gptq_repro/benchmark/eval_ppl.py
```

它在 WikiText-2 test set 上计算 Perplexity。可以评估：

1. 原始 FP16 模型。
2. 加载 `qwen3-smooth-block-mixed-state_dict.pt` 后的 fake 量化模型。
3. 额外叠加 activation fake quantization 的模型。

### 4.1 测原始 FP16 PPL

```bash
python benchmark/eval_ppl.py \
  --model-dir "$MODEL_DIR" \
  --label fp16_baseline \
  --act-quant none \
  --output-dir output/benchmark
```

### 4.2 测 Smooth Block Mixed Fake 量化权重 PPL

```bash
python benchmark/eval_ppl.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --label smooth_block_mixed_w4 \
  --act-quant none \
  --output-dir output/benchmark
```

### 4.3 测权重 Fake 量化 + 激活 Fake 量化

例如叠加 per-group INT4 activation fake quantization：

```bash
python benchmark/eval_ppl.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --label smooth_block_mixed_w4_a4g64 \
  --act-quant int4-g64 \
  --output-dir output/benchmark
```

也可以全局使用 INT4 activation，但对某些层改用 INT8：

```bash
python benchmark/eval_ppl.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --label smooth_block_mixed_w4_a4g64_down_int8 \
  --act-quant int4-g64 \
  --act-quant-override down_proj:int8 \
  --output-dir output/benchmark
```

### 4.4 PPL 输出

默认保存到：

```text
output/benchmark/
```

主要输出：

```text
ppl_<label>.json
results.txt
```

其中：

| 文件 | 作用 |
|---|---|
| `ppl_<label>.json` | 单次 PPL 结果，包含 PPL、tokens、耗时、模型路径、权重路径、activation quant 设置。 |
| `results.txt` | 追加式汇总表，每次运行会追加一行，便于比较多个实验。 |

如果是离线环境，可以指定本地 WikiText-2：

```bash
python benchmark/eval_ppl.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --label smooth_block_mixed_w4_offline \
  --act-quant none \
  --local-wikitext2-dir /root/autodl-tmp/datasets/wikitext2 \
  --output-dir output/benchmark
```

---

## 5. Zero-Shot 测评：benchmark/eval_lm_eval.py

脚本：

```text
/root/autodl-tmp/Zip/qwen3_gptq_repro/benchmark/eval_lm_eval.py
```

它使用 LM-Eval Harness 测 zero-shot / few-shot 任务。常见任务包括：

```text
arc_challenge,arc_easy,piqa,hellaswag,lambada_openai,winogrande
```

### 5.1 测原始 FP16 模型

```bash
python benchmark/eval_lm_eval.py \
  --model-dir "$MODEL_DIR" \
  --tasks arc_challenge,arc_easy,piqa,hellaswag,lambada_openai,winogrande \
  --batch-size auto \
  --apply-chat-template \
  --label fp16_zero_shot \
  --output-dir output/lm_eval
```

### 5.2 测 Smooth Block Mixed Fake 量化模型

```bash
python benchmark/eval_lm_eval.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --tasks arc_challenge,arc_easy,piqa,hellaswag,lambada_openai,winogrande \
  --batch-size auto \
  --apply-chat-template \
  --label smooth_block_mixed_w4_zero_shot \
  --act-quant none \
  --output-dir output/lm_eval
```

### 5.3 测权重 Fake 量化 + 激活 Fake 量化

```bash
python benchmark/eval_lm_eval.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --tasks arc_challenge,arc_easy,piqa,hellaswag,lambada_openai,winogrande \
  --batch-size auto \
  --apply-chat-template \
  --label smooth_block_mixed_w4_a4g64_zero_shot \
  --act-quant int4-g64 \
  --output-dir output/lm_eval
```

### 5.4 离线数据集缓存

如果 zero-shot 数据集已经缓存到本地，可以使用：

```bash
python benchmark/eval_lm_eval.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --tasks arc_challenge,arc_easy,piqa,hellaswag,lambada_openai,winogrande \
  --batch-size auto \
  --apply-chat-template \
  --label smooth_block_mixed_w4_zero_shot_offline \
  --act-quant none \
  --dataset-cache-dir /root/autodl-tmp/datasets \
  --datasets-offline \
  --output-dir output/lm_eval
```

`winogrande` 的 HuggingFace hub id 是 `allenai/winogrande`。脚本会自动为它创建一个常用 cache alias；如果你的本地目录名不同，可以显式指定：

```bash
--dataset-cache-alias allenai/winogrande=/root/autodl-tmp/datasets/winogrande
```

### 5.5 LM-Eval 输出

默认保存到：

```text
output/lm_eval/
```

主要输出：

```text
lm_eval_<label>.json
results_lm_eval.txt
```

其中：

| 文件 | 作用 |
|---|---|
| `lm_eval_<label>.json` | LM-Eval 原始结果和本次运行 metadata。 |
| `results_lm_eval.txt` | 追加式指标汇总表，每个 task/metric 一行。 |

---

## 6. 一次完整实验的最小命令

下面是从模型路径到量化、PPL、zero-shot 的完整最小流程：

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro

export MODEL_DIR=/root/autodl-tmp/Zip/models/Qwen3-4B-Instruct-2507
export OUT_DIR=/root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_block_mixed_example

python qwen3_smooth_block_mixed.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUT_DIR" \
  --nsamples 128 \
  --seqlen 1024 \
  --block-rows 16 \
  --block-cols 128 \
  --groupsize 64 \
  --budget-ratio 0.2 \
  --second-path residual_int4 \
  --act-order \
  --percdamp 0.01 \
  --max-search-batches 128 \
  --save-layer-output-errors

python benchmark/eval_ppl.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --label smooth_block_mixed_w4 \
  --act-quant none \
  --output-dir output/benchmark

python benchmark/eval_lm_eval.py \
  --model-dir "$MODEL_DIR" \
  --quant-weights "$OUT_DIR/qwen3-smooth-block-mixed-state_dict.pt" \
  --tasks arc_challenge,arc_easy,piqa,hellaswag,lambada_openai,winogrande \
  --batch-size auto \
  --apply-chat-template \
  --label smooth_block_mixed_w4_zero_shot \
  --act-quant none \
  --output-dir output/lm_eval
```

---


