# W4A Benchmark V2 — Multi-Format Activation Quantization

> **Purpose**: Self-contained instruction document for running all 20 experiments
> (5 weight variants × 4 activation quantization formats) on Qwen3-4B-Instruct-2507.
>
> **Target script**: `benchmark/eval_ppl.py`
>
> **Date**: 2026-04-08

---

## 1. Path Configuration

Set these shell variables before running any experiment. Adjust paths to match your environment.

```bash
# === Path Configuration ===
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

# === Weight Variant Paths ===
# Variant 1: fp16_baseline — no quant-weights needed (FP16 original)
export W_GPTQ_RAW="${PROJECT_DIR}/output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_GPTQ="${PROJECT_DIR}/output/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_TAIL_SPILL_K75="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_TAIL_SPILL_K90="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"

# === Eval Script ===
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"
```

---

## 2. Cleanup — Archive Old Results

Run this **once** before starting the new benchmark to avoid mixing old and new data.

```bash
cd "${PROJECT_DIR}"

# Archive old results (preserve for reference)
if [ -f "${OUTPUT_DIR}/results.txt" ]; then
    mv "${OUTPUT_DIR}/results.txt" "${OUTPUT_DIR}/results_deprecated_v1.txt"
    echo "[cleanup] Archived old results.txt -> results_deprecated_v1.txt"
fi

# Remove all old per-experiment JSON files
rm -f "${OUTPUT_DIR}"/ppl_*.json
echo "[cleanup] Removed old ppl_*.json files"

# Remove old result_noA4.txt if exists
if [ -f "${OUTPUT_DIR}/result_noA4.txt" ]; then
    mv "${OUTPUT_DIR}/result_noA4.txt" "${OUTPUT_DIR}/result_noA4_deprecated.txt"
    echo "[cleanup] Archived result_noA4.txt"
fi

echo "[cleanup] Done. Ready for V2 benchmark."
```

---

## 3. Parameter Reference

### `--act-quant` (Global Activation Quantization Format)

| Value | Description | Quantization Details |
|-------|-------------|---------------------|
| `none` | FP16 pass-through (default) | No activation quantization applied |
| `int8` | Per-token INT8 symmetric | scale = max(\|x\|, dim=-1) / 127, clamp [-127, 127] |
| `int4-g128` | Per-group INT4 symmetric, group_size=128 | scale = max(\|x_group\|) / 7, clamp [-7, 7], groups of 128 elements |
| `int4-g64` | Per-group INT4 symmetric, group_size=64 | Same as above but with groups of 64 elements |

### `--act-quant-override` (Per-Layer Override)

| Syntax | Description |
|--------|-------------|
| `down_proj:int8` | Apply INT8 to all `down_proj` layers, override global setting |
| `down_proj:int8,o_proj:int8` | Apply INT8 to both `down_proj` and `o_proj` layers |

**Override matching**: If the override key (e.g., `down_proj`) is a substring of the full layer name (e.g., `model.layers.0.mlp.down_proj`), the override takes effect.

### Common CLI Patterns

```bash
# Weight-only evaluation (no activation quantization)
python ${EVAL_SCRIPT} --model-dir ${MODEL_DIR} --act-quant none --label xxx

# W4A8: per-token INT8 activation quantization
python ${EVAL_SCRIPT} --model-dir ${MODEL_DIR} --act-quant int8 --label xxx

# W4A4-g128: per-group INT4 activation quantization
python ${EVAL_SCRIPT} --model-dir ${MODEL_DIR} --act-quant int4-g128 --label xxx

# Mixed: A4-g128 globally, but A8 for down_proj
python ${EVAL_SCRIPT} --model-dir ${MODEL_DIR} --act-quant int4-g128 --act-quant-override down_proj:int8 --label xxx
```

---

## 4. Complete Experiment Commands (20 Experiments)

### Group A: fp16_baseline (Experiments 1–4)

```bash
# Exp 1: fp16_baseline | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label fp16_baseline_Anone

# Exp 2: fp16_baseline | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label fp16_baseline_A8

# Exp 3: fp16_baseline | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label fp16_baseline_A4g128

# Exp 4: fp16_baseline | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label fp16_baseline_A4g128_downA8
```

### Group B: gptq_4bit_raw (Experiments 5–8)

```bash
# Exp 5: gptq_4bit_raw | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_GPTQ_RAW} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label gptq_4bit_raw_Anone

# Exp 6: gptq_4bit_raw | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_GPTQ_RAW} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label gptq_4bit_raw_A8

# Exp 7: gptq_4bit_raw | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_GPTQ_RAW} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label gptq_4bit_raw_A4g128

# Exp 8: gptq_4bit_raw | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_GPTQ_RAW} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label gptq_4bit_raw_A4g128_downA8
```

### Group C: smooth_gptq_4bit (Experiments 9–12)

```bash
# Exp 9: smooth_gptq_4bit | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_gptq_4bit_Anone

# Exp 10: smooth_gptq_4bit | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_gptq_4bit_A8

# Exp 11: smooth_gptq_4bit | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_gptq_4bit_A4g128

# Exp 12: smooth_gptq_4bit | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_gptq_4bit_A4g128_downA8
```

### Group D: percentile_tail_spill_smooth_k75_r16 (Experiments 13–16)

```bash
# Exp 13: percentile_tail_spill_smooth_k75_r16 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_TAIL_SPILL_K75} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptss_k75_r16_Anone

# Exp 14: percentile_tail_spill_smooth_k75_r16 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_TAIL_SPILL_K75} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptss_k75_r16_A8

# Exp 15: percentile_tail_spill_smooth_k75_r16 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_TAIL_SPILL_K75} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptss_k75_r16_A4g128

# Exp 16: percentile_tail_spill_smooth_k75_r16 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_TAIL_SPILL_K75} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptss_k75_r16_A4g128_downA8
```

### Group E: percentile_tail_spill_smooth_k90_r16 (Experiments 17–20)

```bash
# Exp 17: percentile_tail_spill_smooth_k90_r16 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_TAIL_SPILL_K90} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptss_k90_r16_Anone

# Exp 18: percentile_tail_spill_smooth_k90_r16 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_TAIL_SPILL_K90} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptss_k90_r16_A8

# Exp 19: percentile_tail_spill_smooth_k90_r16 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_TAIL_SPILL_K90} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptss_k90_r16_A4g128

# Exp 20: percentile_tail_spill_smooth_k90_r16 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_TAIL_SPILL_K90} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptss_k90_r16_A4g128_downA8
```

---

## 5. Experiment Matrix Quick Reference

| # | Label | Weight Variant | ActQ Descriptor |
|---|-------|---------------|-----------------|
| 1 | fp16_baseline_Anone | fp16_baseline | none |
| 2 | fp16_baseline_A8 | fp16_baseline | int8 |
| 3 | fp16_baseline_A4g128 | fp16_baseline | int4-g128 |
| 4 | fp16_baseline_A4g128_downA8 | fp16_baseline | int4-g128+down_proj:int8 |
| 5 | gptq_4bit_raw_Anone | gptq_4bit_raw | none |
| 6 | gptq_4bit_raw_A8 | gptq_4bit_raw | int8 |
| 7 | gptq_4bit_raw_A4g128 | gptq_4bit_raw | int4-g128 |
| 8 | gptq_4bit_raw_A4g128_downA8 | gptq_4bit_raw | int4-g128+down_proj:int8 |
| 9 | smooth_gptq_4bit_Anone | smooth_gptq_4bit | none |
| 10 | smooth_gptq_4bit_A8 | smooth_gptq_4bit | int8 |
| 11 | smooth_gptq_4bit_A4g128 | smooth_gptq_4bit | int4-g128 |
| 12 | smooth_gptq_4bit_A4g128_downA8 | smooth_gptq_4bit | int4-g128+down_proj:int8 |
| 13 | ptss_k75_r16_Anone | percentile_tail_spill_smooth_k75_r16 | none |
| 14 | ptss_k75_r16_A8 | percentile_tail_spill_smooth_k75_r16 | int8 |
| 15 | ptss_k75_r16_A4g128 | percentile_tail_spill_smooth_k75_r16 | int4-g128 |
| 16 | ptss_k75_r16_A4g128_downA8 | percentile_tail_spill_smooth_k75_r16 | int4-g128+down_proj:int8 |
| 17 | ptss_k90_r16_Anone | percentile_tail_spill_smooth_k90_r16 | none |
| 18 | ptss_k90_r16_A8 | percentile_tail_spill_smooth_k90_r16 | int8 |
| 19 | ptss_k90_r16_A4g128 | percentile_tail_spill_smooth_k90_r16 | int4-g128 |
| 20 | ptss_k90_r16_A4g128_downA8 | percentile_tail_spill_smooth_k90_r16 | int4-g128+down_proj:int8 |

---

## 6. One-Click Shell Script

Save the following as `run_all_v2.sh` in the project root and execute with `bash run_all_v2.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# W4A Benchmark V2 — Run All 20 Experiments
# ============================================================

# --- Path Configuration ---
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

export W_GPTQ_RAW="${PROJECT_DIR}/output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_GPTQ="${PROJECT_DIR}/output/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_TAIL_SPILL_K75="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_TAIL_SPILL_K90="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"

export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"

TOTAL=20
CURRENT=0

run_exp() {
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "================================================================"
    echo "  Experiment ${CURRENT}/${TOTAL}: $1"
    echo "================================================================"
    shift
    python ${EVAL_SCRIPT} "$@"
    echo "[progress] Completed ${CURRENT}/${TOTAL}"
}

# --- Cleanup ---
echo "[cleanup] Archiving old results..."
if [ -f "${OUTPUT_DIR}/results.txt" ]; then
    mv "${OUTPUT_DIR}/results.txt" "${OUTPUT_DIR}/results_deprecated_v1.txt"
fi
rm -f "${OUTPUT_DIR}"/ppl_*.json
if [ -f "${OUTPUT_DIR}/result_noA4.txt" ]; then
    mv "${OUTPUT_DIR}/result_noA4.txt" "${OUTPUT_DIR}/result_noA4_deprecated.txt"
fi
echo "[cleanup] Done."

# === Group A: fp16_baseline ===
run_exp "fp16_baseline | none" \
    --model-dir ${MODEL_DIR} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label fp16_baseline_Anone

run_exp "fp16_baseline | int8" \
    --model-dir ${MODEL_DIR} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label fp16_baseline_A8

run_exp "fp16_baseline | int4-g128" \
    --model-dir ${MODEL_DIR} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label fp16_baseline_A4g128

run_exp "fp16_baseline | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label fp16_baseline_A4g128_downA8

# === Group B: gptq_4bit_raw ===
run_exp "gptq_4bit_raw | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_GPTQ_RAW} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label gptq_4bit_raw_Anone

run_exp "gptq_4bit_raw | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_GPTQ_RAW} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label gptq_4bit_raw_A8

run_exp "gptq_4bit_raw | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_GPTQ_RAW} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label gptq_4bit_raw_A4g128

run_exp "gptq_4bit_raw | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_GPTQ_RAW} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label gptq_4bit_raw_A4g128_downA8

# === Group C: smooth_gptq_4bit ===
run_exp "smooth_gptq_4bit | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_gptq_4bit_Anone

run_exp "smooth_gptq_4bit | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_gptq_4bit_A8

run_exp "smooth_gptq_4bit | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_gptq_4bit_A4g128

run_exp "smooth_gptq_4bit | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label smooth_gptq_4bit_A4g128_downA8

# === Group D: percentile_tail_spill_smooth_k75_r16 ===
run_exp "ptss_k75_r16 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_TAIL_SPILL_K75} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptss_k75_r16_Anone

run_exp "ptss_k75_r16 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_TAIL_SPILL_K75} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptss_k75_r16_A8

run_exp "ptss_k75_r16 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_TAIL_SPILL_K75} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptss_k75_r16_A4g128

run_exp "ptss_k75_r16 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_TAIL_SPILL_K75} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptss_k75_r16_A4g128_downA8

# === Group E: percentile_tail_spill_smooth_k90_r16 ===
run_exp "ptss_k90_r16 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_TAIL_SPILL_K90} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptss_k90_r16_Anone

run_exp "ptss_k90_r16 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_TAIL_SPILL_K90} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptss_k90_r16_A8

run_exp "ptss_k90_r16 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_TAIL_SPILL_K90} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptss_k90_r16_A4g128

run_exp "ptss_k90_r16 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_TAIL_SPILL_K90} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptss_k90_r16_A4g128_downA8

# === Done ===
echo ""
echo "================================================================"
echo "  ALL ${TOTAL} EXPERIMENTS COMPLETED"
echo "  Results: ${OUTPUT_DIR}/results.txt"
echo "================================================================"
```
