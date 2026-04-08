# W4A Benchmark V3 — Percentile Tail Spill High Rank (r64 / r128)

> **Relationship to V2**: This document is a **supplement** to V2, not a replacement.
> V2 covers the original 20 experiments (5 weight variants × 4 activation formats).
> V3 adds 16 new experiments (4 high-rank weight variants × 4 activation formats).
>
> **Results**: New experiment results will be **appended** to the existing
> `output/benchmark/results.txt`. Existing V2 data will NOT be overwritten.
>
> **Purpose**: Self-contained instruction document for generating 4 new
> Percentile Tail Spill weight variants (rank=64, rank=128) and running
> 16 PPL evaluation experiments on Qwen3-4B-Instruct-2507.
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
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_percentile_tail_spill.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"

# === New Weight Variant Paths (V3 High Rank) ===
export W_PTSS_K75_R64="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTSS_K75_R128="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTSS_K90_R64="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTSS_K90_R128="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
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

Generate 4 new Percentile Tail Spill weight variants with higher tail rank (64 and 128).
All variants use the **smooth state dict** as the initialization base.

### Common GPTQ Hyperparameters

These are identical to V2:

| Parameter | Value |
|-----------|-------|
| `--wbits` | 4 |
| `--nsamples` | 128 |
| `--seqlen` | 2048 |
| `--groupsize` | 128 |
| `--percdamp` | 0.01 |
| `--act-order` | enabled |
| `--true-sequential` | enabled |

### Variant F: k=75, rank=64

```bash
cd "${PROJECT_DIR}"

# Variant F: percentile_k=75, tail_rank=64
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r64 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --act-order \
    --true-sequential \
    --percentile-k 75 \
    --tail-rank 64
```

### Variant G: k=75, rank=128

```bash
# Variant G: percentile_k=75, tail_rank=128
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r128 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --act-order \
    --true-sequential \
    --percentile-k 75 \
    --tail-rank 128
```

### Variant H: k=90, rank=64

```bash
# Variant H: percentile_k=90, tail_rank=64
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r64 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --act-order \
    --true-sequential \
    --percentile-k 90 \
    --tail-rank 64
```

### Variant I: k=90, rank=128

```bash
# Variant I: percentile_k=90, tail_rank=128
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r128 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --act-order \
    --true-sequential \
    --percentile-k 90 \
    --tail-rank 128
```

### Expected Outputs

After Phase 1 completes, verify these files exist:

```bash
echo "=== Phase 1 Output Verification ==="
for variant in "from_smooth_k75_r64" "from_smooth_k75_r128" "from_smooth_k90_r64" "from_smooth_k90_r128"; do
    dir="${PROJECT_DIR}/output/exp_percentile_tail_spill/${variant}"
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

## 4. Phase 2 — PPL Evaluation (16 Experiments)

Run PPL evaluation for each of the 4 new weight variants with 4 activation quantization formats.

> **Note**: Results will be **appended** to the existing `output/benchmark/results.txt`.
> V2 results already present in that file will NOT be overwritten.

### Group F: ptss_k75_r64 (Experiments 1–4)

```bash
cd "${PROJECT_DIR}"

# Exp 1: ptss_k75_r64 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K75_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptss_k75_r64_Anone

# Exp 2: ptss_k75_r64 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K75_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptss_k75_r64_A8

# Exp 3: ptss_k75_r64 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K75_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptss_k75_r64_A4g128

# Exp 4: ptss_k75_r64 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K75_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptss_k75_r64_A4g128_downA8
```

### Group G: ptss_k75_r128 (Experiments 5–8)

```bash
# Exp 5: ptss_k75_r128 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K75_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptss_k75_r128_Anone

# Exp 6: ptss_k75_r128 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K75_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptss_k75_r128_A8

# Exp 7: ptss_k75_r128 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K75_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptss_k75_r128_A4g128

# Exp 8: ptss_k75_r128 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K75_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptss_k75_r128_A4g128_downA8
```

### Group H: ptss_k90_r64 (Experiments 9–12)

```bash
# Exp 9: ptss_k90_r64 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K90_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptss_k90_r64_Anone

# Exp 10: ptss_k90_r64 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K90_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptss_k90_r64_A8

# Exp 11: ptss_k90_r64 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K90_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptss_k90_r64_A4g128

# Exp 12: ptss_k90_r64 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K90_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptss_k90_r64_A4g128_downA8
```

### Group I: ptss_k90_r128 (Experiments 13–16)

```bash
# Exp 13: ptss_k90_r128 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K90_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label ptss_k90_r128_Anone

# Exp 14: ptss_k90_r128 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K90_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label ptss_k90_r128_A8

# Exp 15: ptss_k90_r128 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K90_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label ptss_k90_r128_A4g128

# Exp 16: ptss_k90_r128 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_PTSS_K90_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label ptss_k90_r128_A4g128_downA8
```

---

## 5. Experiment Matrix Quick Reference

| # | Label | Weight Variant | ActQ Descriptor |
|---|-------|---------------|-----------------|
| 1 | ptss_k75_r64_Anone | ptss_k75_r64 (F) | none |
| 2 | ptss_k75_r64_A8 | ptss_k75_r64 (F) | int8 |
| 3 | ptss_k75_r64_A4g128 | ptss_k75_r64 (F) | int4-g128 |
| 4 | ptss_k75_r64_A4g128_downA8 | ptss_k75_r64 (F) | int4-g128+down_proj:int8 |
| 5 | ptss_k75_r128_Anone | ptss_k75_r128 (G) | none |
| 6 | ptss_k75_r128_A8 | ptss_k75_r128 (G) | int8 |
| 7 | ptss_k75_r128_A4g128 | ptss_k75_r128 (G) | int4-g128 |
| 8 | ptss_k75_r128_A4g128_downA8 | ptss_k75_r128 (G) | int4-g128+down_proj:int8 |
| 9 | ptss_k90_r64_Anone | ptss_k90_r64 (H) | none |
| 10 | ptss_k90_r64_A8 | ptss_k90_r64 (H) | int8 |
| 11 | ptss_k90_r64_A4g128 | ptss_k90_r64 (H) | int4-g128 |
| 12 | ptss_k90_r64_A4g128_downA8 | ptss_k90_r64 (H) | int4-g128+down_proj:int8 |
| 13 | ptss_k90_r128_Anone | ptss_k90_r128 (I) | none |
| 14 | ptss_k90_r128_A8 | ptss_k90_r128 (I) | int8 |
| 15 | ptss_k90_r128_A4g128 | ptss_k90_r128 (I) | int4-g128 |
| 16 | ptss_k90_r128_A4g128_downA8 | ptss_k90_r128 (I) | int4-g128+down_proj:int8 |

---

## 6. One-Click Shell Script

Save the following as `run_v3_high_rank.sh` in the project root and execute with `bash run_v3_high_rank.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# W4A Benchmark V3 — Percentile Tail Spill High Rank (r64/r128)
# Generates 4 weight variants + runs 16 PPL evaluations
# ============================================================

# --- Path Configuration ---
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_percentile_tail_spill.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"

export W_PTSS_K75_R64="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTSS_K75_R128="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTSS_K90_R64="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_PTSS_K90_R128="${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"

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
# Phase 1: Generate 4 Weight Variants (4 tasks)
# ============================================================
PHASE1_TOTAL=4
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

# Variant F: k=75, rank=64
run_quant "Variant F: k75_r64" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r64 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --percentile-k 75 --tail-rank 64

# Variant G: k=75, rank=128
run_quant "Variant G: k75_r128" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k75_r128 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --percentile-k 75 --tail-rank 128

# Variant H: k=90, rank=64
run_quant "Variant H: k90_r64" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r64 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --percentile-k 90 --tail-rank 64

# Variant I: k=90, rank=128
run_quant "Variant I: k90_r128" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_percentile_tail_spill/from_smooth_k90_r128 \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --percentile-k 90 --tail-rank 128

echo ""
echo "================================================================"
echo "  [Phase 1] ALL ${PHASE1_TOTAL} WEIGHT VARIANTS GENERATED"
echo "================================================================"

# Verify Phase 1 outputs
echo "=== Phase 1 Output Verification ==="
for variant in "from_smooth_k75_r64" "from_smooth_k75_r128" "from_smooth_k90_r64" "from_smooth_k90_r128"; do
    dir="${PROJECT_DIR}/output/exp_percentile_tail_spill/${variant}"
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
# Phase 2: PPL Evaluation (16 experiments)
# ============================================================
PHASE2_TOTAL=16
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

# === Group F: ptss_k75_r64 ===
run_exp "ptss_k75_r64 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K75_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptss_k75_r64_Anone

run_exp "ptss_k75_r64 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K75_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptss_k75_r64_A8

run_exp "ptss_k75_r64 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K75_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptss_k75_r64_A4g128

run_exp "ptss_k75_r64 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K75_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptss_k75_r64_A4g128_downA8

# === Group G: ptss_k75_r128 ===
run_exp "ptss_k75_r128 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K75_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptss_k75_r128_Anone

run_exp "ptss_k75_r128 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K75_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptss_k75_r128_A8

run_exp "ptss_k75_r128 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K75_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptss_k75_r128_A4g128

run_exp "ptss_k75_r128 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K75_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptss_k75_r128_A4g128_downA8

# === Group H: ptss_k90_r64 ===
run_exp "ptss_k90_r64 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K90_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptss_k90_r64_Anone

run_exp "ptss_k90_r64 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K90_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptss_k90_r64_A8

run_exp "ptss_k90_r64 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K90_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptss_k90_r64_A4g128

run_exp "ptss_k90_r64 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K90_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptss_k90_r64_A4g128_downA8

# === Group I: ptss_k90_r128 ===
run_exp "ptss_k90_r128 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K90_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label ptss_k90_r128_Anone

run_exp "ptss_k90_r128 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K90_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label ptss_k90_r128_A8

run_exp "ptss_k90_r128 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K90_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label ptss_k90_r128_A4g128

run_exp "ptss_k90_r128 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_PTSS_K90_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 --label ptss_k90_r128_A4g128_downA8

# === Done ===
echo ""
echo "================================================================"
echo "  ALL TASKS COMPLETED"
echo "  Phase 1: ${PHASE1_TOTAL} weight variants generated"
echo "  Phase 2: ${PHASE2_TOTAL} PPL evaluations completed"
echo "  Results: ${OUTPUT_DIR}/results.txt"
echo "================================================================"
```
