# W4A Benchmark V8 — Smooth + Head Absorb (r16 / r64 / r128)

> **Purpose**: Self-contained instruction document for generating 3 Smooth-based Head Absorb
> weight variants and running 12 PPL evaluation experiments on Qwen3-4B-Instruct-2507.
>
> **Key Difference from V7**: V7 used Tail Absorb (INT8 on the **least important** columns —
> last tail_rank columns after actorder sorting). V8 uses Head Absorb (INT8 on the **most
> important** columns — first tail_rank columns after actorder sorting).
>
> **Motivation**: V7 showed that rank parameter had minimal impact on PPL (r16/r64/r128 differ
> by only 0.01-0.08), suggesting that INT8 on unimportant columns provides limited benefit.
> V8 tests the opposite: give higher precision (INT8 > INT4) to the **most important** columns.
>
> **Quantization Pipeline**:
> ```
> FP16 原始权重 → SmoothQuant(alpha=1) → GPTQ Head Absorb (最重要列 INT8, 误差正常传播) → 量化权重
> ```
>
> **Results**: All 12 experiment results will be written to a **new independent file**
> `output/benchmark/results_smooth_head_absorb.txt`. Existing result files will NOT be modified.
>
> **Date**: 2026-04-10

---

## 1. Path Configuration

```bash
# === Path Configuration ===
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

# === Scripts ===
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_tail_absorb.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"

# === SmoothQuant Pre-processed Weights (alpha=1) ===
export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"

# === Results File (isolated from all previous experiments) ===
export RESULTS_FILE="results_smooth_head_absorb.txt"

# === New Smooth + Head Absorb Weight Paths (act-order ON) ===
export W_SMOOTH_HA_R16="${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_HA_R64="${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_HA_R128="${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
```

---

## 2. Phase 1 — Generate 3 Smooth + Head Absorb Weight Variants

**Key points:**
- Uses `qwen3_gptq_tail_absorb.py` with the new `--head-absorb` flag
- Uses `--use-standard-quantizer` (standard min/max Quantizer, consistent with V7)
- Starts from **SmoothQuant(alpha=1) weights** via `--init-state-dict`
- All 3 variants use **act-order ON** (default)

### Common GPTQ Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--wbits` | 4 | 4-bit weight quantization |
| `--nsamples` | 128 | Calibration samples |
| `--seqlen` | 2048 | Sequence length |
| `--groupsize` | 128 | Group size |
| `--percdamp` | 0.01 | Hessian damping |
| `--true-sequential` | enabled | Must pass explicitly |
| `--use-standard-quantizer` | enabled | Standard min/max (not percentile) |
| `--init-state-dict` | `${SMOOTH_STATE_DICT}` | Load SmoothQuant(alpha=1) weights |
| `--head-absorb` | enabled | **NEW**: INT8 on most important columns |

### 3 Weight Variants Summary

| # | Variant | tail-rank | act-order | Output Directory |
|---|---------|-----------|-----------|-----------------|
| 1 | smooth_ha_r16 | 16 | ✅ ON | `exp_smooth_head_absorb/from_smooth_r16/` |
| 2 | smooth_ha_r64 | 64 | ✅ ON | `exp_smooth_head_absorb/from_smooth_r64/` |
| 3 | smooth_ha_r128 | 128 | ✅ ON | `exp_smooth_head_absorb/from_smooth_r128/` |

### Variant 1: Smooth + Head Absorb, rank=16

```bash
cd "${PROJECT_DIR}"

python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r16 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 16 \
    --head-absorb
```

### Variant 2: Smooth + Head Absorb, rank=64

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r64 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 64 \
    --head-absorb
```

### Variant 3: Smooth + Head Absorb, rank=128

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r128 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 128 \
    --head-absorb
```

### Expected Outputs

```bash
echo "=== Phase 1 Output Verification ==="
for rank in 16 64 128; do
    dir="${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r${rank}"
    pt="${dir}/qwen3-4b-instruct-2507-gptq-4bit.pt"
    meta="${dir}/metadata.json"
    if [ -f "${pt}" ] && [ -f "${meta}" ]; then
        echo "[OK] from_smooth_r${rank}: weights + metadata found"
        # Verify metadata method field
        method=$(python -c "import json; print(json.load(open('${meta}'))['method'])")
        echo "     method=${method} (expected: head_absorb)"
    else
        echo "[FAIL] from_smooth_r${rank}: missing files in ${dir}"
    fi
done
```

---

## 3. Phase 2 — PPL Evaluation (12 Experiments)

Run PPL evaluation for 3 weight variants × 4 activation quantization formats = 12 experiments.

> All results are written to `output/benchmark/results_smooth_head_absorb.txt`.

### Group 1: smooth_ha_r16 (Experiments 1–4)

```bash
cd "${PROJECT_DIR}"

# Exp 1: smooth_ha_r16 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ha_r16_Anone \
    --results-file ${RESULTS_FILE}

# Exp 2: smooth_ha_r16 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ha_r16_A8 \
    --results-file ${RESULTS_FILE}

# Exp 3: smooth_ha_r16 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ha_r16_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 4: smooth_ha_r16 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ha_r16_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 2: smooth_ha_r64 (Experiments 5–8)

```bash
# Exp 5: smooth_ha_r64 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ha_r64_Anone \
    --results-file ${RESULTS_FILE}

# Exp 6: smooth_ha_r64 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ha_r64_A8 \
    --results-file ${RESULTS_FILE}

# Exp 7: smooth_ha_r64 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ha_r64_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 8: smooth_ha_r64 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ha_r64_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 3: smooth_ha_r128 (Experiments 9–12)

```bash
# Exp 9: smooth_ha_r128 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ha_r128_Anone \
    --results-file ${RESULTS_FILE}

# Exp 10: smooth_ha_r128 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ha_r128_A8 \
    --results-file ${RESULTS_FILE}

# Exp 11: smooth_ha_r128 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ha_r128_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 12: smooth_ha_r128 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_HA_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ha_r128_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

---

## 4. Phase 3 — Summary Comparison

### V8 Head Absorb vs V7 Tail Absorb (act-order ON)

| ActQ | smooth_ta_r16 (V7) | smooth_ha_r16 (V8) | smooth_ta_r64 (V7) | smooth_ha_r64 (V8) | smooth_ta_r128 (V7) | smooth_ha_r128 (V8) |
|------|--------------------|--------------------|--------------------|--------------------|---------------------|---------------------|
| none | 10.3428 | _TBD_ | 10.4042 | _TBD_ | 10.4184 | _TBD_ |
| int8 | 10.6156 | _TBD_ | 10.6748 | _TBD_ | 10.6897 | _TBD_ |
| int4-g128 | 13.2699 | _TBD_ | 13.3250 | _TBD_ | 13.3127 | _TBD_ |
| int4-g128+down:int8 | 12.6241 | _TBD_ | 12.6773 | _TBD_ | 12.6651 | _TBD_ |

---

## 5. Experiment Matrix Quick Reference

| # | Label | Weight Variant | ActQ Descriptor |
|---|-------|---------------|-----------------|
| 1 | smooth_ha_r16_Anone | smooth_ha_r16 | none |
| 2 | smooth_ha_r16_A8 | smooth_ha_r16 | int8 |
| 3 | smooth_ha_r16_A4g128 | smooth_ha_r16 | int4-g128 |
| 4 | smooth_ha_r16_A4g128_downA8 | smooth_ha_r16 | int4-g128+down_proj:int8 |
| 5 | smooth_ha_r64_Anone | smooth_ha_r64 | none |
| 6 | smooth_ha_r64_A8 | smooth_ha_r64 | int8 |
| 7 | smooth_ha_r64_A4g128 | smooth_ha_r64 | int4-g128 |
| 8 | smooth_ha_r64_A4g128_downA8 | smooth_ha_r64 | int4-g128+down_proj:int8 |
| 9 | smooth_ha_r128_Anone | smooth_ha_r128 | none |
| 10 | smooth_ha_r128_A8 | smooth_ha_r128 | int8 |
| 11 | smooth_ha_r128_A4g128 | smooth_ha_r128 | int4-g128 |
| 12 | smooth_ha_r128_A4g128_downA8 | smooth_ha_r128 | int4-g128+down_proj:int8 |

---

## 6. One-Click Shell Script

Save as `run_v8_smooth_head_absorb.sh` in the project root.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# W4A Benchmark V8 — Smooth + Head Absorb (r16 / r64 / r128)
# Generates 3 weight variants + runs 12 PPL evaluations
# ============================================================

export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_tail_absorb.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"
export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"
export RESULTS_FILE="results_smooth_head_absorb.txt"

export W_SMOOTH_HA_R16="${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_HA_R64="${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_HA_R128="${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"

cd "${PROJECT_DIR}"

# --- Pre-flight Checks ---
echo "=== Pre-flight Checks ==="
for f in "${QUANT_SCRIPT}" "${EVAL_SCRIPT}" "${SMOOTH_STATE_DICT}"; do
    if [ ! -f "$f" ]; then echo "[FAIL] Missing: $f"; exit 1; fi
done
for d in "${MODEL_DIR}" "${WIKITEXT2_DIR}"; do
    if [ ! -d "$d" ]; then echo "[FAIL] Missing dir: $d"; exit 1; fi
done
# Verify --head-absorb support
grep -q "head-absorb" "${QUANT_SCRIPT}" && echo "[OK] --head-absorb supported" || { echo "[FAIL] --head-absorb NOT found"; exit 1; }
echo "[OK] All pre-flight checks passed."

# ============================================================
# Phase 1: Generate 3 Weight Variants
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

run_quant "Smooth + Head Absorb r16" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r16 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential --use-standard-quantizer \
    --tail-rank 16 --head-absorb

run_quant "Smooth + Head Absorb r64" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r64 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential --use-standard-quantizer \
    --tail-rank 64 --head-absorb

run_quant "Smooth + Head Absorb r128" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r128 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential --use-standard-quantizer \
    --tail-rank 128 --head-absorb

echo ""
echo "================================================================"
echo "  [Phase 1] ALL ${PHASE1_TOTAL} WEIGHT VARIANTS GENERATED"
echo "================================================================"

# Verify Phase 1 outputs
for rank in 16 64 128; do
    dir="${PROJECT_DIR}/output/exp_smooth_head_absorb/from_smooth_r${rank}"
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
# Phase 2: PPL Evaluation (12 experiments)
# ============================================================
PHASE2_TOTAL=12
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

# === Group 1: smooth_ha_r16 ===
run_exp "smooth_ha_r16 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ha_r16_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r16 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ha_r16_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r16 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ha_r16_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r16 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ha_r16_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 2: smooth_ha_r64 ===
run_exp "smooth_ha_r64 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ha_r64_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r64 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ha_r64_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r64 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ha_r64_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r64 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ha_r64_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 3: smooth_ha_r128 ===
run_exp "smooth_ha_r128 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ha_r128_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r128 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ha_r128_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r128 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ha_r128_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ha_r128 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_HA_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ha_r128_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Done ===
echo ""
echo "================================================================"
echo "  ALL TASKS COMPLETED"
echo "  Phase 1: ${PHASE1_TOTAL} weight variants generated"
echo "  Phase 2: ${PHASE2_TOTAL} PPL evaluations completed"
echo "  Results: ${OUTPUT_DIR}/${RESULTS_FILE}"
echo "================================================================"
```
