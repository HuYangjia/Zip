> ⚠️ DEPRECATED: This instruction document has been superseded by W4A_BENCHMARK_V2.md

# W4A4 Benchmark 升级 — 变更说明与运行命令

> **本次升级核心变更**：所有 benchmark 评测默认启用 **A4 激活 INT4 量化模拟**，PPL 结果从 W4 升级为 **W4A4**。

---

## 0. 路径配置

运行前请根据你的实际环境修改以下路径变量：

```bash
# ⚠️ 请修改为你的实际路径
MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
OUTPUT_DIR="${PROJECT_DIR}/output"
WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"  # 本地 WikiText-2 数据集目录（可选）
```

---

## 1. 变更说明

### 1.1 修改的文件

| 文件 | 变更内容 |
|------|----------|
| `benchmark/eval_ppl.py` | 新增 A4 激活量化模拟功能（3 个核心函数 + 命令行参数 + 结果记录） |

### 1.2 代码修改摘要

**`benchmark/eval_ppl.py`** 新增内容：

1. **`fake_quantize_activation_int4(x)`** — INT4 per-token 对称量化→反量化函数
   - `scale = max(|x|, dim=-1) / 7`
   - `x_q = clamp(round(x / scale), -7, 7)`
   - `x_dq = x_q * scale`

2. **`inject_activation_quant(model)`** — 遍历模型所有 `nn.Linear` 层，注册 `forward_pre_hook`，在每次前向传播前对输入激活执行 fake quantization
   - 包含 NaN/Inf 检测：异常时打印警告并跳过该层

3. **`remove_activation_quant(handles)`** — 评测结束后移除所有 hook，确保不影响模型后续使用

4. **`--no-act-quant`** 命令行参数 — 禁用激活量化，回退到纯 FP16 激活模式（用于对比实验或复现旧结果）

5. **结果记录增强** — JSON 和 results.txt 中新增 `act_quant` / `ActQ` 字段，标注激活量化状态（`int4` 或 `none`）

### 1.3 行为变化

| 项目 | 旧行为 | 新行为 |
|------|--------|--------|
| 激活精度 | FP16（不量化） | **INT4 per-token 对称量化→反量化**（默认） |
| 命令行 | 无激活量化选项 | 默认启用 A4，`--no-act-quant` 可关闭 |
| results.txt | 无 ActQ 列 | 新增 `ActQ` 列（`int4` / `none`） |
| JSON 结果 | 无 act_quant 字段 | 新增 `act_quant` 字段 |

---

## 2. 清理旧结果

在重新评测前，**必须**先删除旧的 results.txt，确保结果文件从零开始：

```bash
cd ${PROJECT_DIR}

# 删除旧的汇总文件
rm -f ${OUTPUT_DIR}/benchmark/results.txt

# 可选：删除所有旧的 JSON 结果文件
rm -f ${OUTPUT_DIR}/benchmark/ppl_*.json
```

---

## 3. 新增实验：Percentile Tail Spill (smooth, k=90, r=16)

### 3.1 量化（生成权重）

```bash
cd ${PROJECT_DIR}

python qwen3_gptq_percentile_tail_spill.py \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}/exp_percentile_tail_spill/from_smooth_k90_r16" \
  --init-state-dict "${OUTPUT_DIR}/smooth/smoothed_model_state_dict.pt" \
  --nsamples 32 --seqlen 1024 \
  --groupsize 128 --percdamp 0.01 \
  --true-sequential \
  --tail-rank 16 --percentile-k 90.0 \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### 3.2 评测（见第 4 节 J 组）

---

## 4. 全量 Benchmark 评测命令

以下命令均从项目根目录 `${PROJECT_DIR}` 运行。所有命令**默认启用 A4 激活量化**。

> **Label 命名规范**：`<方法>_<来源>_<参数>`，例如 `percentile_tail_spill_smooth_k90_r16`

### A. FP16 基线（无权重量化，有 A4 激活量化）

```bash
cd ${PROJECT_DIR}

python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --label fp16_baseline \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### B. GPTQ 4-bit (from raw)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label gptq_4bit_raw \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### C. Smooth(α=1) + GPTQ 4-bit

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label smooth_gptq_4bit \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### D. Tail Mixed (5% INT8)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_tail_mixed/tail5p_int8_gptq4/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label tail_mixed_5pct \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### E. Mode A — Percentile+Tail (k=75, r=16)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_percentile_tail/modeA_k75_r16_int8/qwen3-4b-instruct-2507-percentile-tail.pt" \
  --label percentile_tail_modeA_k75_r16 \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### F. Mode B — GPTQ main + tail 补偿 (k=75, r=16)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_percentile_tail/modeB_k75_r16_int8/qwen3-4b-instruct-2507-percentile-tail.pt" \
  --label percentile_tail_modeB_k75_r16 \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### G. Tail Spill Standard (r=16, 标准 min/max Quantizer)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_percentile_tail_spill/from_raw_standard_r16/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label tail_spill_standard_r16 \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### H. Percentile Tail Spill (raw, k=75, r=16)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_percentile_tail_spill/from_raw_k75_r16/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label percentile_tail_spill_raw_k75_r16 \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### I. Percentile Tail Spill (smooth, k=75, r=16)

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_percentile_tail_spill/from_smooth_k75_r16/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label percentile_tail_spill_smooth_k75_r16 \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

### J. 【新增】Percentile Tail Spill (smooth, k=90, r=16)

> ⚠️ 需要先运行第 3 节的量化命令生成权重

```bash
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --quant-weights "${OUTPUT_DIR}/exp_percentile_tail_spill/from_smooth_k90_r16/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --label percentile_tail_spill_smooth_k90_r16 \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

---

## 5. 可选参数说明

### `--no-act-quant`

禁用 A4 激活量化模拟，回退到纯 FP16 激活模式。适用于：

- **对比实验**：对比 W4 vs W4A4 的 PPL 差异
- **复现旧结果**：复现升级前的 W4-only benchmark 结果

```bash
# 示例：FP16 基线，不启用激活量化（复现旧结果）
python benchmark/eval_ppl.py \
  --model-dir "${MODEL_DIR}" \
  --label fp16_baseline_no_act_quant \
  --no-act-quant \
  --output-dir "${OUTPUT_DIR}/benchmark" \
  --local-wikitext2-dir "${WIKITEXT2_DIR}"
```

在 results.txt 中，`ActQ` 列会显示 `none`（而非默认的 `int4`），方便区分。

---

## 6. 一键运行脚本（可选）

如果希望一次性运行所有评测，可以将以下命令保存为 shell 脚本：

```bash
#!/bin/bash
set -e

MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
OUTPUT_DIR="${PROJECT_DIR}/output"
WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

cd ${PROJECT_DIR}

# 清理旧结果
rm -f ${OUTPUT_DIR}/benchmark/results.txt
rm -f ${OUTPUT_DIR}/benchmark/ppl_*.json

echo "=== A. FP16 基线 ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --label fp16_baseline --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== B. GPTQ 4-bit (raw) ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt" --label gptq_4bit_raw --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== C. Smooth + GPTQ 4-bit ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt" --label smooth_gptq_4bit --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== D. Tail Mixed 5% ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/exp_tail_mixed/tail5p_int8_gptq4/qwen3-4b-instruct-2507-gptq-4bit.pt" --label tail_mixed_5pct --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== E. Mode A (k=75, r=16) ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/exp_percentile_tail/modeA_k75_r16_int8/qwen3-4b-instruct-2507-percentile-tail.pt" --label percentile_tail_modeA_k75_r16 --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== F. Mode B (k=75, r=16) ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/exp_percentile_tail/modeB_k75_r16_int8/qwen3-4b-instruct-2507-percentile-tail.pt" --label percentile_tail_modeB_k75_r16 --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== G. Tail Spill Standard (r=16) ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/exp_percentile_tail_spill/from_raw_standard_r16/qwen3-4b-instruct-2507-gptq-4bit.pt" --label tail_spill_standard_r16 --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== H. Percentile Tail Spill (raw, k=75, r=16) ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/exp_percentile_tail_spill/from_raw_k75_r16/qwen3-4b-instruct-2507-gptq-4bit.pt" --label percentile_tail_spill_raw_k75_r16 --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== I. Percentile Tail Spill (smooth, k=75, r=16) ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/exp_percentile_tail_spill/from_smooth_k75_r16/qwen3-4b-instruct-2507-gptq-4bit.pt" --label percentile_tail_spill_smooth_k75_r16 --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo "=== J. Percentile Tail Spill (smooth, k=90, r=16) [新增] ==="
python benchmark/eval_ppl.py --model-dir "${MODEL_DIR}" --quant-weights "${OUTPUT_DIR}/exp_percentile_tail_spill/from_smooth_k90_r16/qwen3-4b-instruct-2507-gptq-4bit.pt" --label percentile_tail_spill_smooth_k90_r16 --output-dir "${OUTPUT_DIR}/benchmark" --local-wikitext2-dir "${WIKITEXT2_DIR}"

echo ""
echo "=== 全部评测完成！==="
echo "结果汇总: ${OUTPUT_DIR}/benchmark/results.txt"
cat ${OUTPUT_DIR}/benchmark/results.txt
```

---

## 7. 预期结果

加入 A4 激活量化后，预期所有方案的 PPL 都会上升（因为激活也引入了量化误差）。但 **smooth 系列方案的相对排名可能会改善**，因为 SmoothQuant 改善了激活分布，使其更适合低精度量化。

| 变体 | 旧 PPL (W4) | 新 PPL (W4A4) | 说明 |
|------|-------------|---------------|------|
| FP16 基线 | 10.0449 | ↑ | 仅有 A4 损失 |
| Mode B | 10.0454 | ↑ | |
| Tail Mixed | 10.2043 | ↑ | |
| Tail Spill Standard | 10.2282 | ↑ | |
| GPTQ 4-bit | 10.3845 | ↑ | |
| Smooth+GPTQ | 10.8361 | ↑ 但相对排名可能改善 | smooth 改善激活分布 |
| Percentile Tail Spill (smooth, k=90) | — | 新增 | 更温和的 percentile |
