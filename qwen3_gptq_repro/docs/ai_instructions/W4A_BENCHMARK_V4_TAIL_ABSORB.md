# W4A Benchmark V4 — Percentile Tail Absorb (r16 / r64 / r128)

> **Relationship to V2/V3**: This document is a **supplement** to V2 and V3, not a replacement.
> V2 covers the original 20 experiments (5 Tail Spill weight variants × 4 activation formats).
> V3 covers 16 experiments (4 high-rank Tail Spill weight variants × 4 activation formats).
> V4 adds 24 new experiments (6 Tail Absorb weight variants × 4 activation formats).
>
> **Results**: New experiment results will be **appended** to the existing
> `output/benchmark/results.txt`. Existing V2/V3 data will NOT be overwritten.
>
> **Purpose**: Self-contained instruction document for generating 6 new
> Percentile Tail Absorb weight variants (rank=16, rank=64, rank=128) and running
> 24 PPL evaluation experiments on Qwen3-4B-Instruct-2507.
>
> **Key Difference from V2/V3**: Tail Absorb quantizes tail columns with INT8
> per-column symmetric quantization (with error propagation), whereas Tail Spill
> skipped quantization for tail columns entirely.
>
> **Date**: 2026-04-08

---

## 1. Path Configuration

Set these shell variables before running any command. Adjust paths to match your environment.

```bash
# === Path Configuration ===
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

# === Smooth State Dict ===
export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"

# === Scripts ===
# NOTE: V4 uses qwen3_gptq_tail_absorb.py (NOT qwen3_gptq_percentile_tail_spill.py)
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_tail_absorb.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"

# === Weight Variant Paths (V4 Tail Absorb) ===
export W_PTAB_K75_R16="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K75_R64="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K75_R128="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K90_R16="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K90_R64="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K90_R128="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
```

---

## 2. Pre-flight Checks

Run these checks **before** starting any experiment to ensure all prerequisites are in place.

```bash
cd "${PROJECT_DIR}"

echo "=== Pre-flight Checks ==="

# Check smooth state dict
if [ -f "${SMOOTH_STATE_DICT}" ]; then
    echo "[OK] Smooth state dict found: ${SMOOTH_STATE_DICT}"
else
    echo "[FAIL] Smooth state dict NOT found: ${SMOOTH_STATE_DICT}"
    echo "       Run qwen3_smooth.py first to generate it."
    exit 1
fi

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

echo "=== All checks passed. Ready to proceed. ==="
```

---

## 3. Phase 1 — Generate Weight Variants

Generate 6 Percentile Tail Absorb weight variants with different percentile_k and tail_rank combinations.
All variants use the **smooth state dict** as the initialization base.

### Common GPTQ Hyperparameters

These are identical to V2/V3 except that `--act-order` is **not** passed (actorder is enabled by default in `qwen3_gptq_tail_absorb.py`):

| Parameter | Value |
|-----------|-------|
| `--wbits` | 4 |
| `--nsamples` | 128 |
| `--seqlen` | 2048 |
| `--groupsize` | 128 |
| `--percdamp` | 0.01 |
| actorder | enabled (default, do NOT pass `--act-order`) |
| `--true-sequential` | enabled |

### Variant J: k=75, rank=16

```bash
cd "${PROJECT_DIR}"

# Variant J: percentile_k=75, tail_rank=16
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r16 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --percentile-k 75 \
    --tail-rank 16
```

### Variant K: k=75, rank=64

```bash
# Variant K: percentile_k=75, tail_rank=64
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r64 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --percentile-k 75 \
    --tail-rank 64
```

### Variant L: k=75, rank=128

```bash
# Variant L: percentile_k=75, tail_rank=128
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r128 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --percentile-k 75 \
    --tail-rank 128
```

### Variant M: k=90, rank=16

```bash
# Variant M: percentile_k=90, tail_rank=16
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r16 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --percentile-k 90 \
    --tail-rank 16
```

### Variant N: k=90, rank=64

```bash
# Variant N: percentile_k=90, tail_rank=64
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r64 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --percentile-k 90 \
    --tail-rank 64
```

### Variant O: k=90, rank=128

```bash
# Variant O: percentile_k=90, tail_rank=128
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r128 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --percentile-k 90 \
    --tail-rank 128
```

### Expected Outputs

After Phase 1 completes, verify these files exist:

```bash
echo "=== Phase 1 Output Verification ==="
for variant in "from_smooth_k75_r16" "from_smooth_k75_r64" "from_smooth_k75_r128" "from_smooth_k90_r16" "from_smooth_k90_r64" "from_smooth_k90_r128"; do
    dir="${PROJECT_DIR}/output/exp_tail_absorb/${variant}"
    pt="${dir}/qwen3-4b-instruct-2507-gptq-4bit.pt"
    meta="${dir}/metadata.json"
    if [ -f "${pt}" ] && [ -f "${meta}" ]; then
        echo "[OK] ${variant}: weights + metadata found"
    else
        echo "[FAIL] ${variant}: missing files in ${dir}"
    fi
done
```

---

## 4. Phase 2 — PPL Evaluation (24 Experiments)

Run PPL evaluation for each of the 6 weight variants with 4 activation quantization formats.

> **Note**: Results will be **appended** to the existing `output/benchmark/results.txt`.
> V2/V3 results already present in that file will NOT be overwritten.

### Group J: ptab_k75_r16 (Experiments 1–4)

```bash
cd "${PROJECT_DIR}"

# Exp 1: ptab_k75_r16 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptab_k75_r16_Anone

# Exp 2: ptab_k75_r16 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptab_k75_r16_A8

# Exp 3: ptab_k75_r16 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptab_k75_r16_A4g128

# Exp 4: ptab_k75_r16 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptab_k75_r16_A4g128_downA8
```

### Group K: ptab_k75_r64 (Experiments 5–8)

```bash
# Exp 5: ptab_k75_r64 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptab_k75_r64_Anone

# Exp 6: ptab_k75_r64 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptab_k75_r64_A8

# Exp 7: ptab_k75_r64 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptab_k75_r64_A4g128

# Exp 8: ptab_k75_r64 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptab_k75_r64_A4g128_downA8
```

### Group L: ptab_k75_r128 (Experiments 9–12)

```bash
# Exp 9: ptab_k75_r128 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptab_k75_r128_Anone

# Exp 10: ptab_k75_r128 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptab_k75_r128_A8

# Exp 11: ptab_k75_r128 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptab_k75_r128_A4g128

# Exp 12: ptab_k75_r128 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K75_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptab_k75_r128_A4g128_downA8
```

### Group M: ptab_k90_r16 (Experiments 13–16)

```bash
# Exp 13: ptab_k90_r16 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptab_k90_r16_Anone

# Exp 14: ptab_k90_r16 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptab_k90_r16_A8

# Exp 15: ptab_k90_r16 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptab_k90_r16_A4g128

# Exp 16: ptab_k90_r16 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptab_k90_r16_A4g128_downA8
```

### Group N: ptab_k90_r64 (Experiments 17–20)

```bash
# Exp 17: ptab_k90_r64 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptab_k90_r64_Anone

# Exp 18: ptab_k90_r64 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptab_k90_r64_A8

# Exp 19: ptab_k90_r64 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptab_k90_r64_A4g128

# Exp 20: ptab_k90_r64 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptab_k90_r64_A4g128_downA8
```

### Group O: ptab_k90_r128 (Experiments 21–24)

```bash
# Exp 21: ptab_k90_r128 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptab_k90_r128_Anone

# Exp 22: ptab_k90_r128 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptab_k90_r128_A8

# Exp 23: ptab_k90_r128 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptab_k90_r128_A4g128

# Exp 24: ptab_k90_r128 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTAB_K90_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptab_k90_r128_A4g128_downA8
```

---

## 5. Experiment Matrix Quick Reference

| # | Label | Weight Variant | ActQ Descriptor |
|---|-------|---------------|-----------------|
| 1 | ptab_k75_r16_Anone | ptab_k75_r16 (J) | none |
| 2 | ptab_k75_r16_A8 | ptab_k75_r16 (J) | int8 |
| 3 | ptab_k75_r16_A4g128 | ptab_k75_r16 (J) | int4-g128 |
| 4 | ptab_k75_r16_A4g128_downA8 | ptab_k75_r16 (J) | int4-g128+down_proj:int8 |
| 5 | ptab_k75_r64_Anone | ptab_k75_r64 (K) | none |
| 6 | ptab_k75_r64_A8 | ptab_k75_r64 (K) | int8 |
| 7 | ptab_k75_r64_A4g128 | ptab_k75_r64 (K) | int4-g128 |
| 8 | ptab_k75_r64_A4g128_downA8 | ptab_k75_r64 (K) | int4-g128+down_proj:int8 |
| 9 | ptab_k75_r128_Anone | ptab_k75_r128 (L) | none |
| 10 | ptab_k75_r128_A8 | ptab_k75_r128 (L) | int8 |
| 11 | ptab_k75_r128_A4g128 | ptab_k75_r128 (L) | int4-g128 |
| 12 | ptab_k75_r128_A4g128_downA8 | ptab_k75_r128 (L) | int4-g128+down_proj:int8 |
| 13 | ptab_k90_r16_Anone | ptab_k90_r16 (M) | none |
| 14 | ptab_k90_r16_A8 | ptab_k90_r16 (M) | int8 |
| 15 | ptab_k90_r16_A4g128 | ptab_k90_r16 (M) | int4-g128 |
| 16 | ptab_k90_r16_A4g128_downA8 | ptab_k90_r16 (M) | int4-g128+down_proj:int8 |
| 17 | ptab_k90_r64_Anone | ptab_k90_r64 (N) | none |
| 18 | ptab_k90_r64_A8 | ptab_k90_r64 (N) | int8 |
| 19 | ptab_k90_r64_A4g128 | ptab_k90_r64 (N) | int4-g128 |
| 20 | ptab_k90_r64_A4g128_downA8 | ptab_k90_r64 (N) | int4-g128+down_proj:int8 |
| 21 | ptab_k90_r128_Anone | ptab_k90_r128 (O) | none |
| 22 | ptab_k90_r128_A8 | ptab_k90_r128 (O) | int8 |
| 23 | ptab_k90_r128_A4g128 | ptab_k90_r128 (O) | int4-g128 |
| 24 | ptab_k90_r128_A4g128_downA8 | ptab_k90_r128 (O) | int4-g128+down_proj:int8 |

---

## 6. One-Click Shell Script

Save the following as `run_v4_tail_absorb.sh` in the project root and execute with `bash run_v4_tail_absorb.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# W4A Benchmark V4 — Percentile Tail Absorb (r16/r64/r128)
# Generates 6 weight variants + runs 24 PPL evaluations
# ============================================================

# --- Path Configuration ---
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"
# NOTE: V4 uses qwen3_gptq_tail_absorb.py (NOT qwen3_gptq_percentile_tail_spill.py)
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_tail_absorb.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"

export W_PTAB_K75_R16="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K75_R64="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K75_R128="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K90_R16="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K90_R64="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTAB_K90_R128="${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"

cd "${PROJECT_DIR}"

# --- Pre-flight Checks ---
echo "=== Pre-flight Checks ==="
for f in "${SMOOTH_STATE_DICT}" "${QUANT_SCRIPT}" "${EVAL_SCRIPT}"; do
    if [ ! -f "$f" ]; then echo "[FAIL] Missing: $f"; exit 1; fi
done
for d in "${MODEL_DIR}" "${WIKITEXT2_DIR}"; do
    if [ ! -d "$d" ]; then echo "[FAIL] Missing dir: $d"; exit 1; fi
done
echo "[OK] All pre-flight checks passed."

# ============================================================
# Phase 1: Generate 6 Weight Variants (6 tasks)
# ============================================================
PHASE1_TOTAL=6
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

# Variant J: k=75, rank=16
run_quant "Variant J: k75_r16" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r16 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --percentile-k 75 --tail-rank 16

# Variant K: k=75, rank=64
run_quant "Variant K: k75_r64" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r64 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --percentile-k 75 --tail-rank 64

# Variant L: k=75, rank=128
run_quant "Variant L: k75_r128" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k75_r128 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --percentile-k 75 --tail-rank 128

# Variant M: k=90, rank=16
run_quant "Variant M: k90_r16" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r16 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --percentile-k 90 --tail-rank 16

# Variant N: k=90, rank=64
run_quant "Variant N: k90_r64" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r64 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --percentile-k 90 --tail-rank 64

# Variant O: k=90, rank=128
run_quant "Variant O: k90_r128" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_tail_absorb/from_smooth_k90_r128 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --percentile-k 90 --tail-rank 128

echo ""
echo "================================================================"
echo "  [Phase 1] ALL ${PHASE1_TOTAL} WEIGHT VARIANTS GENERATED"
echo "================================================================"

# Verify Phase 1 outputs
echo "=== Phase 1 Output Verification ==="
for variant in "from_smooth_k75_r16" "from_smooth_k75_r64" "from_smooth_k75_r128" "from_smooth_k90_r16" "from_smooth_k90_r64" "from_smooth_k90_r128"; do
    dir="${PROJECT_DIR}/output/exp_tail_absorb/${variant}"
    pt="${dir}/qwen3-4b-instruct-2507-gptq-4bit.pt"
    meta="${dir}/metadata.json"
    if [ -f "${pt}" ] && [ -f "${meta}" ]; then
        echo "[OK] ${variant}: weights + metadata found"
    else
        echo "[FAIL] ${variant}: missing files in ${dir}"
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

# === Group J: ptab_k75_r16 ===
run_exp "ptab_k75_r16 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptab_k75_r16_Anone

run_exp "ptab_k75_r16 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptab_k75_r16_A8

run_exp "ptab_k75_r16 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptab_k75_r16_A4g128

run_exp "ptab_k75_r16 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptab_k75_r16_A4g128_downA8

# === Group K: ptab_k75_r64 ===
run_exp "ptab_k75_r64 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptab_k75_r64_Anone

run_exp "ptab_k75_r64 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptab_k75_r64_A8

run_exp "ptab_k75_r64 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptab_k75_r64_A4g128

run_exp "ptab_k75_r64 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptab_k75_r64_A4g128_downA8

# === Group L: ptab_k75_r128 ===
run_exp "ptab_k75_r128 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptab_k75_r128_Anone

run_exp "ptab_k75_r128 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptab_k75_r128_A8

run_exp "ptab_k75_r128 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptab_k75_r128_A4g128

run_exp "ptab_k75_r128 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K75_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptab_k75_r128_A4g128_downA8

# === Group M: ptab_k90_r16 ===
run_exp "ptab_k90_r16 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptab_k90_r16_Anone

run_exp "ptab_k90_r16 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptab_k90_r16_A8

run_exp "ptab_k90_r16 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptab_k90_r16_A4g128

run_exp "ptab_k90_r16 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptab_k90_r16_A4g128_downA8

# === Group N: ptab_k90_r64 ===
run_exp "ptab_k90_r64 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptab_k90_r64_Anone

run_exp "ptab_k90_r64 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptab_k90_r64_A8

run_exp "ptab_k90_r64 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptab_k90_r64_A4g128

run_exp "ptab_k90_r64 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptab_k90_r64_A4g128_downA8

# === Group O: ptab_k90_r128 ===
run_exp "ptab_k90_r128 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptab_k90_r128_Anone

run_exp "ptab_k90_r128 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptab_k90_r128_A8

run_exp "ptab_k90_r128 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptab_k90_r128_A4g128

run_exp "ptab_k90_r128 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTAB_K90_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptab_k90_r128_A4g128_downA8

# === Done ===
echo ""
echo "================================================================"
echo "  ALL TASKS COMPLETED"
echo "  Phase 1: ${PHASE1_TOTAL} weight variants generated"
echo "  Phase 2: ${PHASE2_TOTAL} PPL evaluations completed"
echo "  Results: ${OUTPUT_DIR}/results.txt"
echo "================================================================"
```
