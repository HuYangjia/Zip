# WikiText-2 Perplexity Benchmark 评测指南

## 1. 评测方法说明

### 什么是 Perplexity (PPL)

Perplexity 是衡量语言模型质量的标准指标，定义为：

```
PPL = exp( -1/N * Σ log P(token_i | context) )
```

其中 N 是评测 token 总数。**PPL 越低，模型越好**。直觉上，PPL 表示模型在预测下一个 token 时的"困惑程度"——PPL=10 意味着模型平均在 10 个候选中犹豫。

### 滑动窗口计算方式

由于 Transformer 模型有最大上下文长度限制（本项目中 seqlen=2048），我们将整个 test set 切分为不重叠的固定长度窗口：

```
|--- window 1 (2048 tokens) ---|--- window 2 (2048 tokens) ---|--- ... ---|
```

对每个窗口独立做前向传播，计算每个 token 的 cross-entropy loss，最后汇总：

```
PPL = exp( sum_of_all_losses / total_token_count )
```

**为什么选择非重叠 stride**：
- 重叠 stride（如 stride=512）会让每个 token 被评测多次，结果更平滑但耗时成倍增加
- 非重叠 stride 保证每个 token 只评测一次，耗时适中（3-10 分钟），且结果已足够稳定
- 量化论文（GPTQ、AWQ 等）通常也使用非重叠方式，便于对比

### 评测数据集

使用 **WikiText-2 test set**（约 24.5 万 token），这是量化领域最常用的评测基准。

---

## 2. 环境依赖

```
torch >= 2.0
transformers >= 4.40
datasets >= 2.0       # 用于加载 WikiText-2（在线或本地）
```

如果在离线环境中运行，需要提前下载 WikiText-2 数据集到本地（见第 6 节）。

---

## 3. 目录结构

```
qwen3_gptq_repro/
├── benchmark/                  ← benchmark 相关文件集中在此
│   ├── eval_ppl.py             ← 评测脚本
│   └── BENCHMARK_GUIDE.md      ← 本文档
├── output/
│   └── benchmark/              ← 评测结果输出目录（自动创建）
│       ├── ppl_fp16_baseline.json
│       ├── ppl_gptq_4bit_raw.json
│       ├── ...
│       └── results.txt         ← 所有评测结果汇总文件
├── qwen3_gptq.py
├── ...
```

---

## 4. 路径配置

运行前需要根据你的实际环境修改以下路径：

```bash
# ⚠️ 请修改为你的实际路径
MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
OUTPUT_DIR="${PROJECT_DIR}/output"
```

---

## 5. 各变体运行命令

以下命令均从项目根目录 `qwen3_gptq_repro/` 运行。请确保已按第 4 节配置好路径变量。

### A. FP16 原始模型（基线上界）

```bash
cd ${PROJECT_DIR}

python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --label fp16_baseline \
  --output-dir "${OUTPUT_DIR}/benchmark"
```

### B. GPTQ 4-bit (from raw)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label gptq_4bit_raw \
  --output-dir "${OUTPUT_DIR}/benchmark"
```

### C. Smooth(α=1) + GPTQ 4-bit

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label smooth_gptq_4bit \
  --output-dir "${OUTPUT_DIR}/benchmark"
```

### D. Tail Mixed (5% INT8)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_tail_mixed/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label tail_mixed_5pct \
  --output-dir "${OUTPUT_DIR}/benchmark"
```

### E. Percentile+Tail Mode A

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_percentile_tail/modeA/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label percentile_tail_modeA \
  --output-dir "${OUTPUT_DIR}/benchmark"
```

### F. Percentile+Tail Mode B

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_percentile_tail/modeB/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label percentile_tail_modeB \
  --output-dir "${OUTPUT_DIR}/benchmark"
```

---

## 6. 为新实验添加评测

只需指定新的 `--quant-weights` 和 `--label` 即可：

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "output/my_new_experiment/weights.pt" \
  --label my_new_experiment \
  --output-dir "${OUTPUT_DIR}/benchmark"
```

脚本会自动：
1. 加载原始模型结构
2. 用你的量化权重替换 state_dict
3. 在 WikiText-2 test set 上计算 PPL
4. 将结果保存为 `output/benchmark/ppl_my_new_experiment.json`
5. 将结果追加到 `output/benchmark/results.txt` 汇总文件

---

## 7. 离线环境：使用本地 WikiText-2

如果无法访问 HuggingFace，可以提前下载数据集：

```python
# 在有网络的机器上运行
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
ds.save_to_disk("/path/to/wikitext2_local")
```

然后评测时指定本地路径：

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --local-wikitext2-dir "/path/to/wikitext2_local" \
  --label fp16_baseline
```

---

## 8. 结果输出说明

每次评测会产生两种输出：

### 8.1 JSON 文件（单次结果）

每次评测在 `--output-dir` 下生成一个独立的 JSON 文件 `ppl_<label>.json`：

```json
{
  "label": "gptq_4bit_raw",
  "ppl": 8.2345,
  "total_tokens": 245678,
  "elapsed_seconds": 312.45,
  "model_dir": "/home/zhou/.../Qwen3-4B-Instruct-2507",
  "quant_weights": "/home/zhou/.../gptq-4bit.pt",
  "seqlen": 2048,
  "seed": 0,
  "dtype": "float16",
  "data_source": "online: huggingface wikitext-2-raw-v1 test",
  "timestamp": "2026-04-04T20:00:00.000000"
}
```

### 8.2 results.txt（汇总文件）

所有评测结果会自动追加到同一个 `results.txt` 文件中，方便一目了然地对比各变体：

```
WikiText-2 Perplexity Benchmark Results
========================================================================================================
Label                           PPL      Tokens     Time(s)  Dtype       Timestamp                   Quant Weights
--------------------------------------------------------------------------------------------------------
fp16_baseline                    7.2100   245678      180.32  float16     2026-04-04T20:00:00.000000  (FP16 原始模型)
gptq_4bit_raw                    8.2345   245678      312.45  float16     2026-04-04T20:10:00.000000  /home/.../gptq-4bit.pt
smooth_gptq_4bit                 8.1200   245678      305.12  float16     2026-04-04T20:20:00.000000  /home/.../gptq-4bit.pt
...
```

> **提示**：如果需要重新开始汇总，删除 `output/benchmark/results.txt` 即可，下次评测会自动重建。

---

## 9. 结果解读指南

### PPL 差异的意义

| PPL 差异 (相对基线) | 含义 |
|---------------------|------|
| < 0.1 | 可忽略，统计噪声范围内 |
| 0.1 ~ 0.5 | 有意义的差异，值得关注 |
| 0.5 ~ 1.0 | 明显退化，需要分析原因 |
| > 1.0 | 显著退化，量化方法可能存在问题 |

### 各变体预期排序（PPL 从低到高）

```
FP16 基线 (最好)
  ↓
GPTQ 4-bit / Smooth+GPTQ (接近，差异通常 < 0.5)
  ↓
Tail Mixed / Percentile+Tail (实验性方法，PPL 可能略高)
```

> **注意**：以上排序为一般预期。如果某个变体的 PPL 显著偏离预期，可能说明量化过程存在问题（参见项目的错误分析报告）。

---

## 10. 完整参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-dir` | str | (必填) | HuggingFace 模型目录 |
| `--quant-weights` | str | (空) | 量化权重 .pt 路径，不填则评测 FP16 |
| `--label` | str | unnamed | 评测标签，用于标识结果 |
| `--seqlen` | int | 2048 | 滑动窗口大小 |
| `--output-dir` | str | ./output/benchmark | 结果保存目录 |
| `--local-wikitext2-dir` | str | (空) | 本地 WikiText-2 目录 |
| `--seed` | int | 0 | 随机种子 |
| `--dtype` | str | float16 | 模型加载精度 |
