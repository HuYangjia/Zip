# W4A Benchmark V6 — Smooth + Standard Tail Spill (r16 / r64 / r128)

> **Purpose**: Self-contained instruction document for generating 3 Smooth-based Standard Tail Spill
> weight variants (rank=16, 64, 128) and running 24 PPL evaluation experiments
> on Qwen3-4B-Instruct-2507.
>
> **Key Difference from V5**: V5 started from **raw FP16 weights** (no SmoothQuant).
> This V6 experiment starts from **SmoothQuant(alpha=1) pre-processed weights** via
> `--init-state-dict`, combined with the **standard min/max Quantizer** (not PercentileQuantizer).
>
> **Quantization Pipeline**:
> ```
> FP16 原始权重 → SmoothQuant(alpha=1) → Standard GPTQ Tail Spill → 量化权重
> ```
>
> **Results**: All 24 experiment results will be written to a **new independent file**
> `output/benchmark/results_smooth_stdts.txt`. The existing `results.txt` and `results_stdts.txt`
> will NOT be modified.
>
> **Date**: 2026-04-10

---

## 1. Path Configuration

Set these shell variables before running any command. Adjust paths to match your environment.

```bash
# === Path Configuration ===
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

# === Scripts ===
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_percentile_tail_spill.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"

# === SmoothQuant Pre-processed Weights (alpha=1) ===
export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"

# === Results File (isolated from V5 and previous experiments) ===
export RESULTS_FILE="results_smooth_stdts.txt"

# === Existing Baseline Weight Paths ===
export W_RAW_GPTQ="${PROJECT_DIR}/output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_GPTQ="${PROJECT_DIR}/output/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt"

# === New Smooth + Standard Tail Spill Weight Paths ===
export W_SMOOTH_STDTS_R16="${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_STDTS_R64="${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_STDTS_R128="${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
```

---

## 2. Pre-flight Checks

Run these checks **before** starting any experiment to ensure all prerequisites are in place.

```bash
cd "${PROJECT_DIR}"

echo "=== Pre-flight Checks ==="

# Check quantization script
if [ -f "${QUANT_SCRIPT}" ]; then
    echo "[OK] Quantization script found: ${QUANT_SCRIPT}"
else
    echo "[FAIL] Quantization script NOT found: ${QUANT_SCRIPT}"
    exit 1
fi

# Check evaluation script
if [ -f "${EVAL_SCRIPT}" ]; then
    echo "[OK] Evaluation script found: ${EVAL_SCRIPT}"
else
    echo "[FAIL] Evaluation script NOT found: ${EVAL_SCRIPT}"
    exit 1
fi

# Check SmoothQuant pre-processed weights (CRITICAL for V6)
if [ -f "${SMOOTH_STATE_DICT}" ]; then
    echo "[OK] SmoothQuant state dict found: ${SMOOTH_STATE_DICT}"
else
    echo "[FAIL] SmoothQuant state dict NOT found: ${SMOOTH_STATE_DICT}"
    echo "       Please run qwen3_smooth.py first to generate smoothed weights."
    exit 1
fi

# Check WikiText-2 data
if [ -d "${WIKITEXT2_DIR}" ]; then
    echo "[OK] WikiText-2 data found: ${WIKITEXT2_DIR}"
else
    echo "[FAIL] WikiText-2 data NOT found: ${WIKITEXT2_DIR}"
    exit 1
fi

# Check model directory
if [ -d "${MODEL_DIR}" ]; then
    echo "[OK] Model directory found: ${MODEL_DIR}"
else
    echo "[FAIL] Model directory NOT found: ${MODEL_DIR}"
    exit 1
fi

# Check existing baseline weights
if [ -f "${W_RAW_GPTQ}" ]; then
    echo "[OK] Raw GPTQ weights found: ${W_RAW_GPTQ}"
else
    echo "[FAIL] Raw GPTQ weights NOT found: ${W_RAW_GPTQ}"
    exit 1
fi

if [ -f "${W_SMOOTH_GPTQ}" ]; then
    echo "[OK] Smooth GPTQ weights found: ${W_SMOOTH_GPTQ}"
else
    echo "[FAIL] Smooth GPTQ weights NOT found: ${W_SMOOTH_GPTQ}"
    exit 1
fi

echo "=== All checks passed. Ready to proceed. ==="
```

---

## 3. Phase 1 — Generate 3 Smooth + Standard Tail Spill Weight Variants

Generate 3 weight variants with rank=16, 64, 128, starting from **SmoothQuant(alpha=1) weights**.

**Key points:**
- Uses `--use-standard-quantizer` (standard min/max Quantizer, NOT PercentileQuantizer)
- Starts from **SmoothQuant(alpha=1) weights** via `--init-state-dict ${SMOOTH_STATE_DICT}`
- Does NOT use `--percentile-k` (irrelevant for standard quantizer)

### Common GPTQ Hyperparameters

| Parameter | Value |
|-----------|-------|
| `--wbits` | 4 |
| `--nsamples` | 128 |
| `--seqlen` | 2048 |
| `--groupsize` | 128 |
| `--percdamp` | 0.01 |
| `--act-order` | enabled |
| `--true-sequential` | enabled |
| `--use-standard-quantizer` | enabled |
| `--init-state-dict` | `${SMOOTH_STATE_DICT}` |

### Variant A: Smooth + Standard Tail Spill, rank=16

```bash
cd "${PROJECT_DIR}"

python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r16 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --act-order \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 16
```

### Variant B: Smooth + Standard Tail Spill, rank=64

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r64 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --act-order \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 64
```

### Variant C: Smooth + Standard Tail Spill, rank=128

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r128 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --act-order \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 128
```

### Expected Outputs

After Phase 1 completes, verify these files exist:

```bash
echo "=== Phase 1 Output Verification ==="
for rank in 16 64 128; do
    dir="${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r${rank}"
    pt="${dir}/qwen3-4b-instruct-2507-gptq-4bit.pt"
    meta="${dir}/metadata.json"
    if [ -f "${pt}" ] && [ -f "${meta}" ]; then
        echo "[OK] from_smooth_r${rank}: weights + metadata found"
    else
        echo "[FAIL] from_smooth_r${rank}: missing files in ${dir}"
    fi
done
```

---

## 4. Phase 2 — PPL Evaluation (24 Experiments)

Run PPL evaluation for 6 weight variants × 4 activation quantization formats = 24 experiments.

> **Note**: All results are written to `output/benchmark/results_smooth_stdts.txt` via
> the `--results-file results_smooth_stdts.txt` argument. The existing `results.txt` and
> `results_stdts.txt` are NOT modified.

### Group 1: fp16_baseline (Experiments 1–4)

```bash
cd "${PROJECT_DIR}"

# Exp 1: fp16_baseline | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label fp16_baseline_Anone \
    --results-file ${RESULTS_FILE}

# Exp 2: fp16_baseline | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label fp16_baseline_A8 \
    --results-file ${RESULTS_FILE}

# Exp 3: fp16_baseline | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label fp16_baseline_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 4: fp16_baseline | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label fp16_baseline_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 2: gptq_4bit_raw (Experiments 5–8)

```bash
# Exp 5: gptq_4bit_raw | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_RAW_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label gptq_4bit_raw_Anone \
    --results-file ${RESULTS_FILE}

# Exp 6: gptq_4bit_raw | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_RAW_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label gptq_4bit_raw_A8 \
    --results-file ${RESULTS_FILE}

# Exp 7: gptq_4bit_raw | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_RAW_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label gptq_4bit_raw_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 8: gptq_4bit_raw | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_RAW_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label gptq_4bit_raw_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 3: smooth_gptq_4bit (Experiments 9–12)

```bash
# Exp 9: smooth_gptq_4bit | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_gptq_4bit_Anone \
    --results-file ${RESULTS_FILE}

# Exp 10: smooth_gptq_4bit | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_gptq_4bit_A8 \
    --results-file ${RESULTS_FILE}

# Exp 11: smooth_gptq_4bit | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_gptq_4bit_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 12: smooth_gptq_4bit | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_GPTQ} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_gptq_4bit_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 4: smooth_stdts_r16 (Experiments 13–16)

```bash
# Exp 13: smooth_stdts_r16 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_stdts_r16_Anone \
    --results-file ${RESULTS_FILE}

# Exp 14: smooth_stdts_r16 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_stdts_r16_A8 \
    --results-file ${RESULTS_FILE}

# Exp 15: smooth_stdts_r16 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_stdts_r16_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 16: smooth_stdts_r16 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_stdts_r16_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 5: smooth_stdts_r64 (Experiments 17–20)

```bash
# Exp 17: smooth_stdts_r64 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_stdts_r64_Anone \
    --results-file ${RESULTS_FILE}

# Exp 18: smooth_stdts_r64 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_stdts_r64_A8 \
    --results-file ${RESULTS_FILE}

# Exp 19: smooth_stdts_r64 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_stdts_r64_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 20: smooth_stdts_r64 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_stdts_r64_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 6: smooth_stdts_r128 (Experiments 21–24)

```bash
# Exp 21: smooth_stdts_r128 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_stdts_r128_Anone \
    --results-file ${RESULTS_FILE}

# Exp 22: smooth_stdts_r128 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_stdts_r128_A8 \
    --results-file ${RESULTS_FILE}

# Exp 23: smooth_stdts_r128 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_stdts_r128_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 24: smooth_stdts_r128 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_STDTS_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_stdts_r128_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
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
| 13 | smooth_stdts_r16_Anone | smooth_stdts_r16 | none |
| 14 | smooth_stdts_r16_A8 | smooth_stdts_r16 | int8 |
| 15 | smooth_stdts_r16_A4g128 | smooth_stdts_r16 | int4-g128 |
| 16 | smooth_stdts_r16_A4g128_downA8 | smooth_stdts_r16 | int4-g128+down_proj:int8 |
| 17 | smooth_stdts_r64_Anone | smooth_stdts_r64 | none |
| 18 | smooth_stdts_r64_A8 | smooth_stdts_r64 | int8 |
| 19 | smooth_stdts_r64_A4g128 | smooth_stdts_r64 | int4-g128 |
| 20 | smooth_stdts_r64_A4g128_downA8 | smooth_stdts_r64 | int4-g128+down_proj:int8 |
| 21 | smooth_stdts_r128_Anone | smooth_stdts_r128 | none |
| 22 | smooth_stdts_r128_A8 | smooth_stdts_r128 | int8 |
| 23 | smooth_stdts_r128_A4g128 | smooth_stdts_r128 | int4-g128 |
| 24 | smooth_stdts_r128_A4g128_downA8 | smooth_stdts_r128 | int4-g128+down_proj:int8 |

---

## 6. One-Click Shell Script

Save the following as `run_v6_smooth_stdts.sh` in the project root and execute with `bash run_v6_smooth_stdts.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# W4A Benchmark V6 — Smooth + Standard Tail Spill (r16 / r64 / r128)
# Generates 3 weight variants (from Smooth) + runs 24 PPL evaluations
# ============================================================

# --- Path Configuration ---
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_percentile_tail_spill.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"
export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"
export RESULTS_FILE="results_smooth_stdts.txt"

export W_RAW_GPTQ="${PROJECT_DIR}/output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_GPTQ="${PROJECT_DIR}/output/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt"

export W_SMOOTH_STDTS_R16="${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_STDTS_R64="${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_STDTS_R128="${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"

cd "${PROJECT_DIR}"

# --- Pre-flight Checks ---
echo "=== Pre-flight Checks ==="
for f in "${QUANT_SCRIPT}" "${EVAL_SCRIPT}" "${W_RAW_GPTQ}" "${W_SMOOTH_GPTQ}" "${SMOOTH_STATE_DICT}"; do
    if [ ! -f "$f" ]; then echo "[FAIL] Missing: $f"; exit 1; fi
done
for d in "${MODEL_DIR}" "${WIKITEXT2_DIR}"; do
    if [ ! -d "$d" ]; then echo "[FAIL] Missing dir: $d"; exit 1; fi
done
echo "[OK] All pre-flight checks passed."

# ============================================================
# Phase 1: Generate 3 Weight Variants (from Smooth)
# ============================================================
PHASE1_TOTAL=3
PHASE1_CURRENT=0

run_quant() {
    PHASE1_CURRENT=$((PHASE1_CURRENT + 1))
    echo ""
    echo "================================================================"
    echo "  [Phase 1] Quantization ${PHASE1_CURRENT}/${PHASE1_TOTAL}: $1"
    echo "================================================================"
    local desc="$1"; shift
    python ${QUANT_SCRIPT} "$@"
    echo "[Phase 1] Completed ${PHASE1_CURRENT}/${PHASE1_TOTAL}: ${desc}"
}

# Variant A: Smooth + Standard Tail Spill, rank=16
run_quant "Smooth + Standard Tail Spill r16" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r16 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --use-standard-quantizer \
    --tail-rank 16

# Variant B: Smooth + Standard Tail Spill, rank=64
run_quant "Smooth + Standard Tail Spill r64" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r64 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --use-standard-quantizer \
    --tail-rank 64

# Variant C: Smooth + Standard Tail Spill, rank=128
run_quant "Smooth + Standard Tail Spill r128" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r128 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --use-standard-quantizer \
    --tail-rank 128

echo ""
echo "================================================================"
echo "  [Phase 1] ALL ${PHASE1_TOTAL} WEIGHT VARIANTS GENERATED"
echo "================================================================"

# Verify Phase 1 outputs
echo "=== Phase 1 Output Verification ==="
for rank in 16 64 128; do
    dir="${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r${rank}"
    pt="${dir}/qwen3-4b-instruct-2507-gptq-4bit.pt"
    meta="${dir}/metadata.json"
    if [ -f "${pt}" ] && [ -f "${meta}" ]; then
        echo "[OK] from_smooth_r${rank}: weights + metadata found"
    else
        echo "[FAIL] from_smooth_r${rank}: missing files in ${dir}"
        exit 1
    fi
done

# ============================================================
# Phase 2: PPL Evaluation (24 experiments)
# ============================================================
PHASE2_TOTAL=24
PHASE2_CURRENT=0

run_exp() {
    PHASE2_CURRENT=$((PHASE2_CURRENT + 1))
    echo ""
    echo "================================================================"
    echo "  [Phase 2] Experiment ${PHASE2_CURRENT}/${PHASE2_TOTAL}: $1"
    echo "================================================================"
    shift
    python ${EVAL_SCRIPT} "$@"
    echo "[Phase 2] Completed ${PHASE2_CURRENT}/${PHASE2_TOTAL}"
}

# === Group 1: fp16_baseline (no --quant-weights) ===
run_exp "fp16_baseline | none" \
    --model-dir ${MODEL_DIR} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label fp16_baseline_Anone --results-file ${RESULTS_FILE}

run_exp "fp16_baseline | int8" \
    --model-dir ${MODEL_DIR} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label fp16_baseline_A8 --results-file ${RESULTS_FILE}

run_exp "fp16_baseline | int4-g128" \
    --model-dir ${MODEL_DIR} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label fp16_baseline_A4g128 --results-file ${RESULTS_FILE}

run_exp "fp16_baseline | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label fp16_baseline_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 2: gptq_4bit_raw ===
run_exp "gptq_4bit_raw | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_RAW_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label gptq_4bit_raw_Anone --results-file ${RESULTS_FILE}

run_exp "gptq_4bit_raw | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_RAW_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label gptq_4bit_raw_A8 --results-file ${RESULTS_FILE}

run_exp "gptq_4bit_raw | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_RAW_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label gptq_4bit_raw_A4g128 --results-file ${RESULTS_FILE}

run_exp "gptq_4bit_raw | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_RAW_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label gptq_4bit_raw_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 3: smooth_gptq_4bit ===
run_exp "smooth_gptq_4bit | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_gptq_4bit_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_gptq_4bit | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_gptq_4bit_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_gptq_4bit | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_gptq_4bit_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_gptq_4bit | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_GPTQ} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_gptq_4bit_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 4: smooth_stdts_r16 ===
run_exp "smooth_stdts_r16 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_stdts_r16_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r16 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_stdts_r16_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r16 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_stdts_r16_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r16 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_stdts_r16_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 5: smooth_stdts_r64 ===
run_exp "smooth_stdts_r64 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_stdts_r64_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r64 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_stdts_r64_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r64 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_stdts_r64_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r64 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_stdts_r64_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 6: smooth_stdts_r128 ===
run_exp "smooth_stdts_r128 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_stdts_r128_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r128 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_stdts_r128_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r128 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_stdts_r128_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_stdts_r128 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_STDTS_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_stdts_r128_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Done ===
echo ""
echo "================================================================"
echo "  ALL TASKS COMPLETED"
echo "  Phase 1: ${PHASE1_TOTAL} weight variants generated (from Smooth)"
echo "  Phase 2: ${PHASE2_TOTAL} PPL evaluations completed"
echo "  Results: ${OUTPUT_DIR}/${RESULTS_FILE}"
echo "================================================================"
```
