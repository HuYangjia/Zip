# V9 Submatrix Mixed Precision — Server Execution Guide

> **Document Purpose**: This guide is for the server-side AI to execute weight variant generation and PPL benchmark evaluation.
> **Date**: 2026-04-17
> **Code Location**: `Zip/qwen3_gptq_repro/`

---

## 0. Overview

V9 implements **submatrix-level mixed precision quantization**. It splits the weight matrix into a grid of `(brow × bcol)` submatrix blocks, selects the top-k% blocks with highest INT4 quantization error, and quantizes them with INT8 instead of INT4. The GPTQ error propagation mechanism is completely unchanged.

### Key Files

| File | Role |
|------|------|
| `qwen3_gptq_submatrix_mixed.py` | **Entry script** — run this to generate quantized weights |
| `gptq_submatrix_mixed.py` | Core GPTQ class `GPTQSubmatrixMixed` |
| `benchmark/eval_ppl.py` | PPL evaluation script (reused from existing pipeline) |
| `gptq_tail_absorb.py` | Dependency — provides `PercentileQuantizer` and `_int8_fakequant_column` |
| `qwen3_gptq.py` | Dependency — provides `get_qwen3`, `get_wikitext2_or_fallback_loader`, etc. |

### Output Format

The entry script outputs:
- `qwen3-4b-instruct-2507-gptq-4bit.pt` — FakeQuant float16 weights (compatible with `eval_ppl.py`)
- `metadata.json` — All quantization parameters and per-layer statistics
- `submatrix_mixed.log` — Detailed log file

---

## 1. Environment Prerequisites

```bash
# Ensure you are in the correct working directory
cd /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro

# Required Python packages (should already be installed)
# torch, transformers, datasets
```

---

## 2. Step 1 — Generate Quantized Weight Variants

### 2.1 CLI Parameter Reference

```
python qwen3_gptq_submatrix_mixed.py \
    --model-dir <MODEL_PATH> \
    --output-dir <OUTPUT_PATH> \
    --block-rows <INT>           # Submatrix row size (default: 128)
    --block-cols <INT>           # Submatrix col size (default: 128, recommend = groupsize)
    --budget-ratio <FLOAT>       # INT8 budget ratio (default: 0.05 = 5%)
    --sensitivity-metric <STR>   # quant_error | weight_norm | hessian_weighted (default: quant_error)
    --groupsize <INT>            # GPTQ group size (default: 128)
    --percentile-k <FLOAT>       # Percentile for PercentileQuantizer (default: 75.0)
    --init-state-dict <PATH>     # Optional: SmoothQuant preprocessed weights
    --nsamples <INT>             # Calibration samples (default: 32)
    --seqlen <INT>               # Sequence length (default: 1024)
    --sym                        # Enable symmetric quantization
    --no-act-order               # Disable act-order (default: enabled)
    --use-standard-quantizer     # Use standard min/max Quantizer instead of PercentileQuantizer
```

### 2.2 Experiment Variants — Weight Generation

Below are the first batch of experiment variants. Each variant isolates one variable for controlled comparison.

| Variant | block_shape | budget | metric | Purpose |
|---------|-------------|--------|--------|---------|
| V1 | (128,128) | 5% | quant_error | Baseline config |
| V2 | (128,128) | 10% | quant_error | Budget ablation: does 10% help significantly? |
| V3 | (64,128) | 5% | quant_error | Finer row granularity: more precise outlier targeting |
| V4 | (128,128) | 5% | hessian_weighted | Metric ablation: Hessian-weighted vs pure quant_error |
| V5 | (128,128) | 5% | weight_norm | Metric ablation: simple weight norm baseline |
| V6 | (64,128) | 10% | quant_error | Combined: finer granularity + larger budget |

#### Variant 1: Baseline (block=128×128, budget=5%, metric=quant_error)

```bash
python qwen3_gptq_submatrix_mixed.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant1 \
    --block-rows 128 \
    --block-cols 128 \
    --budget-ratio 0.05 \
    --sensitivity-metric quant_error \
    --groupsize 128 \
    --percentile-k 75.0
```

#### Variant 2: Budget Ablation (block=128×128, budget=10%, metric=quant_error)

```bash
python qwen3_gptq_submatrix_mixed.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant2 \
    --block-rows 128 \
    --block-cols 128 \
    --budget-ratio 0.10 \
    --sensitivity-metric quant_error \
    --groupsize 128 \
    --percentile-k 75.0
```

#### Variant 3: Finer Row Granularity (block=64×128, budget=5%, metric=quant_error)

```bash
python qwen3_gptq_submatrix_mixed.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant3 \
    --block-rows 64 \
    --block-cols 128 \
    --budget-ratio 0.05 \
    --sensitivity-metric quant_error \
    --groupsize 128 \
    --percentile-k 75.0
```

#### Variant 4: Hessian-Weighted Metric (block=128×128, budget=5%, metric=hessian_weighted)

```bash
python qwen3_gptq_submatrix_mixed.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant4 \
    --block-rows 128 \
    --block-cols 128 \
    --budget-ratio 0.05 \
    --sensitivity-metric hessian_weighted \
    --groupsize 128 \
    --percentile-k 75.0
```

#### Variant 5: Weight Norm Metric (block=128×128, budget=5%, metric=weight_norm)

```bash
python qwen3_gptq_submatrix_mixed.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant5 \
    --block-rows 128 \
    --block-cols 128 \
    --budget-ratio 0.05 \
    --sensitivity-metric weight_norm \
    --groupsize 128 \
    --percentile-k 75.0
```

#### Variant 6: Finer Granularity + Larger Budget (block=64×128, budget=10%, metric=quant_error)

```bash
python qwen3_gptq_submatrix_mixed.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant6 \
    --block-rows 64 \
    --block-cols 128 \
    --budget-ratio 0.10 \
    --sensitivity-metric quant_error \
    --groupsize 128 \
    --percentile-k 75.0
```

### 2.3 Expected Output per Variant

After each variant completes, verify the output directory contains:

```
output/exp_v9_variantN/
├── qwen3-4b-instruct-2507-gptq-4bit.pt   # FakeQuant float16 weights (~7.5 GB)
├── metadata.json                           # Quantization parameters & per-layer stats
└── submatrix_mixed.log                     # Detailed log
```

**Sanity check**: The `metadata.json` should contain `"method": "submatrix_mixed_precision"` and non-empty `layer_stats`.

---

## 3. Step 2 — Run PPL Benchmark (eval_ppl.py)

For each generated weight variant, run the PPL evaluation:

### 3.1 CLI Parameter Reference for eval_ppl.py

```
python benchmark/eval_ppl.py \
    --model-dir <MODEL_PATH> \
    --quant-weights <WEIGHTS_PT_PATH> \
    --label <EXPERIMENT_LABEL> \
    --seqlen 2048 \
    --output-dir <BENCHMARK_OUTPUT_DIR> \
    --act-quant none \
    --results-file results_v9.txt
```

### 3.2 Benchmark Commands for Each Variant

#### Benchmark Variant 1 — Baseline (Anone)

```bash
python benchmark/eval_ppl.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --quant-weights /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant1/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label v9_b128x128_r5_qe_Anone \
    --seqlen 2048 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/benchmark \
    --act-quant none \
    --results-file results_v9.txt
```

#### Benchmark Variant 2 — Budget 10% (Anone)

```bash
python benchmark/eval_ppl.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --quant-weights /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant2/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label v9_b128x128_r10_qe_Anone \
    --seqlen 2048 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/benchmark \
    --act-quant none \
    --results-file results_v9.txt
```

#### Benchmark Variant 3 — Finer Row 64×128 (Anone)

```bash
python benchmark/eval_ppl.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --quant-weights /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant3/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label v9_b64x128_r5_qe_Anone \
    --seqlen 2048 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/benchmark \
    --act-quant none \
    --results-file results_v9.txt
```

#### Benchmark Variant 4 — Hessian-Weighted (Anone)

```bash
python benchmark/eval_ppl.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --quant-weights /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant4/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label v9_b128x128_r5_hw_Anone \
    --seqlen 2048 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/benchmark \
    --act-quant none \
    --results-file results_v9.txt
```

#### Benchmark Variant 5 — Weight Norm (Anone)

```bash
python benchmark/eval_ppl.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --quant-weights /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant5/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label v9_b128x128_r5_wn_Anone \
    --seqlen 2048 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/benchmark \
    --act-quant none \
    --results-file results_v9.txt
```

#### Benchmark Variant 6 — Finer + Larger Budget (Anone)

```bash
python benchmark/eval_ppl.py \
    --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
    --quant-weights /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/exp_v9_variant6/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label v9_b64x128_r10_qe_Anone \
    --seqlen 2048 \
    --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/benchmark \
    --act-quant none \
    --results-file results_v9.txt
```

### 3.3 Expected Benchmark Output

All results will be appended to a single summary file:

```
output/benchmark/results_v9.txt
```

Plus individual JSON files per variant:

```
output/benchmark/ppl_v9_variant1_Anone.json
output/benchmark/ppl_v9_variant2_Anone.json
...
```

### 3.4 Reference Baselines (for comparison)

| Baseline | PPL (Anone) | Description |
|----------|-------------|-------------|
| FP16 | 10.04 | No quantization |
| GPTQ 4-bit raw | 10.38 | Standard GPTQ |
| V7 smooth_ta_r16 | 10.34 | Current best (fixed tail columns) |

**Target**: V9 should achieve PPL < 10.34 (beat V7 best) with 5% INT8 budget.

---

## 4. Troubleshooting

### Common Issues

1. **`ModuleNotFoundError: No module named 'gptq'`**
   - Ensure you run from the `qwen3_gptq_repro/` directory, or the script's path resolution will handle it automatically.

2. **`RuntimeError: CUDA out of memory`**
   - The model is ~4B parameters. Ensure GPU has at least 16GB VRAM.
   - Try reducing `--nsamples` (e.g., `--nsamples 16`).

3. **`FileNotFoundError` for model or weights**
   - Verify `--model-dir` points to the correct Qwen3-4B-Instruct-2507 directory.
   - For `--init-state-dict`, verify the SmoothQuant weights file exists.

4. **`WARNING: block_cols != groupsize`**
   - This is expected if you intentionally set different values. For best results, keep `block_cols = groupsize = 128`.

---

## 5. Execution Order Summary

```
1. Run weight generation (Step 2) for ALL variants first
2. Then run PPL benchmark (Step 3) for ALL variants
3. Check output/benchmark/results_v9.txt for the comparison table
4. Report the results back
```

**Important**: Run weight generation variants sequentially (each takes ~10-30 minutes depending on GPU). Benchmark evaluation is faster (~5 minutes each).
