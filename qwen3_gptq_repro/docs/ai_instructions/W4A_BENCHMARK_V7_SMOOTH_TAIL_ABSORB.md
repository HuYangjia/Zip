# W4A Benchmark V7 — Smooth + Tail Absorb (r16 / r64 / r128) × act-order 消融

> **Purpose**: Self-contained instruction document for generating 6 Smooth-based Tail Absorb
> weight variants (3 rank × 2 act-order configurations) and running 24 PPL evaluation experiments
> on Qwen3-4B-Instruct-2507.
>
> **Key Difference from V6**: V6 used `GPTQTailSpill` (tail columns did NOT propagate error),
> causing PPL to worsen with higher rank. V7 uses `GPTQTailAbsorb` (tail columns properly
> propagate error via Hessian compensation chain), which should fix the rank-PPL inversion.
>
> **New Dimension**: V7 also tests act-order ON (default) vs OFF (`--no-act-order`) to ablate
> the effect of activation ordering on Tail Absorb performance.
>
> **Quantization Pipeline**:
> ```
> FP16 原始权重 → SmoothQuant(alpha=1) → GPTQ Tail Absorb (误差正常传播) → 量化权重
> ```
>
> **Results**: All 24 experiment results will be written to a **new independent file**
> `output/benchmark/results_smooth_tail_absorb.txt`. The existing `results.txt`,
> `results_stdts.txt`, and `results_smooth_stdts.txt` will NOT be modified.
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
# NOTE: V7 uses qwen3_gptq_tail_absorb.py (NOT qwen3_gptq_percentile_tail_spill.py from V6)
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_tail_absorb.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"

# === SmoothQuant Pre-processed Weights (alpha=1) ===
export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"

# === Results File (isolated from V5, V6, and previous experiments) ===
export RESULTS_FILE="results_smooth_tail_absorb.txt"

# === Existing Baseline Weight Paths (for Phase 3 comparison) ===
export W_RAW_GPTQ="${PROJECT_DIR}/output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_GPTQ="${PROJECT_DIR}/output/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt"

# === New Smooth + Tail Absorb Weight Paths (act-order ON, default) ===
export W_SMOOTH_TA_R16="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R64="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R128="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"

# === New Smooth + Tail Absorb Weight Paths (act-order OFF) ===
export W_SMOOTH_TA_R16_NOACT="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r16_noact/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R64_NOACT="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r64_noact/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R128_NOACT="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r128_noact/qwen3-4b-instruct-2507-gptq-4bit.pt"
```

---

## 2. Pre-flight Checks

Run these checks **before** starting any experiment to ensure all prerequisites are in place.

```bash
cd "${PROJECT_DIR}"

echo "=== Pre-flight Checks ==="

# Check quantization script (V7 uses qwen3_gptq_tail_absorb.py)
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

# Check SmoothQuant pre-processed weights (CRITICAL for V7)
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

# Check existing baseline weights (needed for Phase 2 baseline evaluation)
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

# Verify key script parameters
echo ""
echo "--- Parameter Support Verification ---"
grep -q "use-standard-quantizer" "${QUANT_SCRIPT}" && echo "[OK] --use-standard-quantizer supported" || echo "[FAIL] --use-standard-quantizer NOT found"
grep -q "init-state-dict" "${QUANT_SCRIPT}" && echo "[OK] --init-state-dict supported" || echo "[FAIL] --init-state-dict NOT found"
grep -q "tail-rank" "${QUANT_SCRIPT}" && echo "[OK] --tail-rank supported" || echo "[FAIL] --tail-rank NOT found"
grep -q "no-act-order" "${QUANT_SCRIPT}" && echo "[OK] --no-act-order supported" || echo "[FAIL] --no-act-order NOT found"
grep -q "results-file" "${EVAL_SCRIPT}" && echo "[OK] --results-file supported in eval_ppl.py" || echo "[FAIL] --results-file NOT found"

echo ""
echo "=== All checks passed. Ready to proceed. ==="
```

---

## 3. Phase 1 — Generate 6 Smooth + Tail Absorb Weight Variants

Generate 6 weight variants: 3 ranks (16, 64, 128) × 2 act-order configurations (ON, OFF),
starting from **SmoothQuant(alpha=1) weights**.

**Key points:**
- Uses `qwen3_gptq_tail_absorb.py` (NOT `qwen3_gptq_percentile_tail_spill.py` from V6)
- Uses `--use-standard-quantizer` (standard min/max Quantizer, NOT PercentileQuantizer)
- Starts from **SmoothQuant(alpha=1) weights** via `--init-state-dict ${SMOOTH_STATE_DICT}`
- `qwen3_gptq_tail_absorb.py` **defaults to act-order ON** — do NOT pass `--act-order`
- To disable act-order, pass `--no-act-order`

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

### 6 Weight Variants Summary

| # | Variant | tail-rank | act-order | Output Directory |
|---|---------|-----------|-----------|-----------------|
| 1 | smooth_ta_r16 | 16 | ✅ ON | `exp_smooth_tail_absorb/from_smooth_r16/` |
| 2 | smooth_ta_r64 | 64 | ✅ ON | `exp_smooth_tail_absorb/from_smooth_r64/` |
| 3 | smooth_ta_r128 | 128 | ✅ ON | `exp_smooth_tail_absorb/from_smooth_r128/` |
| 4 | smooth_ta_r16_noact | 16 | ❌ OFF | `exp_smooth_tail_absorb/from_smooth_r16_noact/` |
| 5 | smooth_ta_r64_noact | 64 | ❌ OFF | `exp_smooth_tail_absorb/from_smooth_r64_noact/` |
| 6 | smooth_ta_r128_noact | 128 | ❌ OFF | `exp_smooth_tail_absorb/from_smooth_r128_noact/` |

### Variant 1: Smooth + Tail Absorb, rank=16, act-order ON

```bash
cd "${PROJECT_DIR}"

python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r16 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 16
```

### Variant 2: Smooth + Tail Absorb, rank=64, act-order ON

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r64 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 64
```

### Variant 3: Smooth + Tail Absorb, rank=128, act-order ON

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r128 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 128
```

### Variant 4: Smooth + Tail Absorb, rank=16, act-order OFF

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r16_noact \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --no-act-order \
    --tail-rank 16
```

### Variant 5: Smooth + Tail Absorb, rank=64, act-order OFF

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r64_noact \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --no-act-order \
    --tail-rank 64
```

### Variant 6: Smooth + Tail Absorb, rank=128, act-order OFF

```bash
python ${QUANT_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r128_noact \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 \
    --nsamples 128 \
    --seqlen 2048 \
    --groupsize 128 \
    --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --no-act-order \
    --tail-rank 128
```

### Expected Outputs

After Phase 1 completes, verify these files exist:

```bash
echo "=== Phase 1 Output Verification ==="
for rank in 16 64 128; do
    # act-order ON
    dir="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r${rank}"
    pt="${dir}/qwen3-4b-instruct-2507-gptq-4bit.pt"
    meta="${dir}/metadata.json"
    if [ -f "${pt}" ] && [ -f "${meta}" ]; then
        echo "[OK] from_smooth_r${rank} (act-order ON): weights + metadata found"
    else
        echo "[FAIL] from_smooth_r${rank} (act-order ON): missing files in ${dir}"
    fi

    # act-order OFF
    dir="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r${rank}_noact"
    pt="${dir}/qwen3-4b-instruct-2507-gptq-4bit.pt"
    meta="${dir}/metadata.json"
    if [ -f "${pt}" ] && [ -f "${meta}" ]; then
        echo "[OK] from_smooth_r${rank}_noact (act-order OFF): weights + metadata found"
    else
        echo "[FAIL] from_smooth_r${rank}_noact (act-order OFF): missing files in ${dir}"
    fi
done
```

---

## 4. Phase 2 — PPL Evaluation (24 Experiments)

Run PPL evaluation for 6 weight variants × 4 activation quantization formats = 24 experiments.

> **Note**: All results are written to `output/benchmark/results_smooth_tail_absorb.txt` via
> the `--results-file results_smooth_tail_absorb.txt` argument. The existing `results.txt`,
> `results_stdts.txt`, and `results_smooth_stdts.txt` are NOT modified.

### Group 1: smooth_ta_r16 — act-order ON (Experiments 1–4)

```bash
cd "${PROJECT_DIR}"

# Exp 1: smooth_ta_r16 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ta_r16_Anone \
    --results-file ${RESULTS_FILE}

# Exp 2: smooth_ta_r16 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ta_r16_A8 \
    --results-file ${RESULTS_FILE}

# Exp 3: smooth_ta_r16 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ta_r16_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 4: smooth_ta_r16 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R16} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ta_r16_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 2: smooth_ta_r64 — act-order ON (Experiments 5–8)

```bash
# Exp 5: smooth_ta_r64 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ta_r64_Anone \
    --results-file ${RESULTS_FILE}

# Exp 6: smooth_ta_r64 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ta_r64_A8 \
    --results-file ${RESULTS_FILE}

# Exp 7: smooth_ta_r64 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ta_r64_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 8: smooth_ta_r64 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R64} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ta_r64_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 3: smooth_ta_r128 — act-order ON (Experiments 9–12)

```bash
# Exp 9: smooth_ta_r128 | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ta_r128_Anone \
    --results-file ${RESULTS_FILE}

# Exp 10: smooth_ta_r128 | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ta_r128_A8 \
    --results-file ${RESULTS_FILE}

# Exp 11: smooth_ta_r128 | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ta_r128_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 12: smooth_ta_r128 | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R128} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ta_r128_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 4: smooth_ta_r16_noact — act-order OFF (Experiments 13–16)

```bash
# Exp 13: smooth_ta_r16_noact | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R16_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ta_r16_noact_Anone \
    --results-file ${RESULTS_FILE}

# Exp 14: smooth_ta_r16_noact | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R16_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ta_r16_noact_A8 \
    --results-file ${RESULTS_FILE}

# Exp 15: smooth_ta_r16_noact | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R16_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ta_r16_noact_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 16: smooth_ta_r16_noact | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R16_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ta_r16_noact_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 5: smooth_ta_r64_noact — act-order OFF (Experiments 17–20)

```bash
# Exp 17: smooth_ta_r64_noact | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R64_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ta_r64_noact_Anone \
    --results-file ${RESULTS_FILE}

# Exp 18: smooth_ta_r64_noact | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R64_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ta_r64_noact_A8 \
    --results-file ${RESULTS_FILE}

# Exp 19: smooth_ta_r64_noact | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R64_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ta_r64_noact_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 20: smooth_ta_r64_noact | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R64_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ta_r64_noact_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

### Group 6: smooth_ta_r128_noact — act-order OFF (Experiments 21–24)

```bash
# Exp 21: smooth_ta_r128_noact | ActQ=none
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R128_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none \
    --label smooth_ta_r128_noact_Anone \
    --results-file ${RESULTS_FILE}

# Exp 22: smooth_ta_r128_noact | ActQ=int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R128_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 \
    --label smooth_ta_r128_noact_A8 \
    --results-file ${RESULTS_FILE}

# Exp 23: smooth_ta_r128_noact | ActQ=int4-g128
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R128_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --label smooth_ta_r128_noact_A4g128 \
    --results-file ${RESULTS_FILE}

# Exp 24: smooth_ta_r128_noact | ActQ=int4-g128 + down_proj:int8
python ${EVAL_SCRIPT} \
    --model-dir ${MODEL_DIR} \
    --quant-weights ${W_SMOOTH_TA_R128_NOACT} \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 \
    --act-quant-override down_proj:int8 \
    --label smooth_ta_r128_noact_A4g128_downA8 \
    --results-file ${RESULTS_FILE}
```

---

## 5. Phase 3 — Summary Comparison (36 Experiments)

After all 24 V7 experiments complete, combine with the 12 baseline experiments from
`results_smooth_stdts.txt` for a comprehensive 36-experiment comparison.

### 5.1 Baseline Experiments (from `results_smooth_stdts.txt`)

These 12 experiments are already completed and available:

| # | Label | Description | PPL (Anone) |
|---|-------|-------------|-------------|
| 1 | fp16_baseline_Anone | FP16 original (no quantization) | 10.04 |
| 2 | fp16_baseline_A8 | FP16 + INT8 activation | 10.32 |
| 3 | fp16_baseline_A4g128 | FP16 + INT4-g128 activation | 14.16 |
| 4 | fp16_baseline_A4g128_downA8 | FP16 + INT4-g128 + down_proj:INT8 | 13.40 |
| 5 | gptq_4bit_raw_Anone | Standard GPTQ 4-bit (from raw) | 10.38 |
| 6 | gptq_4bit_raw_A8 | Standard GPTQ 4-bit + INT8 act | 10.70 |
| 7 | gptq_4bit_raw_A4g128 | Standard GPTQ 4-bit + INT4-g128 act | 15.31 |
| 8 | gptq_4bit_raw_A4g128_downA8 | Standard GPTQ 4-bit + INT4-g128 + down:INT8 | 14.33 |
| 9 | smooth_gptq_4bit_Anone | Smooth + Standard GPTQ 4-bit | 10.84 |
| 10 | smooth_gptq_4bit_A8 | Smooth + Standard GPTQ + INT8 act | 11.14 |
| 11 | smooth_gptq_4bit_A4g128 | Smooth + Standard GPTQ + INT4-g128 act | 14.18 |
| 12 | smooth_gptq_4bit_A4g128_downA8 | Smooth + Standard GPTQ + INT4-g128 + down:INT8 | 13.46 |

### 5.2 Comparison Template (fill in after experiments)

#### Dimension 1: Tail Absorb vs Baseline (same ActQ, act-order ON)

| ActQ | fp16_baseline | smooth_gptq_4bit | smooth_ta_r16 | smooth_ta_r64 | smooth_ta_r128 |
|------|--------------|-------------------|---------------|---------------|----------------|
| none | 10.04 | 10.84 | _TBD_ | _TBD_ | _TBD_ |
| int8 | 10.32 | 11.14 | _TBD_ | _TBD_ | _TBD_ |
| int4-g128 | 14.16 | 14.18 | _TBD_ | _TBD_ | _TBD_ |
| int4-g128+down:int8 | 13.40 | 13.46 | _TBD_ | _TBD_ | _TBD_ |

#### Dimension 2: act-order Ablation (same rank, ActQ=none)

| Variant | act-order ON | act-order OFF | Δ PPL |
|---------|-------------|---------------|-------|
| smooth_ta_r16 | _TBD_ | _TBD_ | _TBD_ |
| smooth_ta_r64 | _TBD_ | _TBD_ | _TBD_ |
| smooth_ta_r128 | _TBD_ | _TBD_ | _TBD_ |

#### Dimension 3: Rank Trend Verification (ActQ=none)

| Config | r16 | r64 | r128 | Trend (expected: ↓) |
|--------|-----|-----|------|---------------------|
| act-order ON | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| act-order OFF | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

#### Dimension 4: Gap from FP16 Baseline (ActQ=none)

| Variant | PPL | Δ from FP16 (10.04) |
|---------|-----|---------------------|
| fp16_baseline | 10.04 | 0.00 |
| gptq_4bit_raw | 10.38 | +0.34 |
| smooth_gptq_4bit | 10.84 | +0.80 |
| smooth_ta_r16 (act ON) | _TBD_ | _TBD_ |
| smooth_ta_r64 (act ON) | _TBD_ | _TBD_ |
| smooth_ta_r128 (act ON) | _TBD_ | _TBD_ |
| smooth_ta_r16_noact | _TBD_ | _TBD_ |
| smooth_ta_r64_noact | _TBD_ | _TBD_ |
| smooth_ta_r128_noact | _TBD_ | _TBD_ |

---

## 6. Experiment Matrix Quick Reference

| # | Label | Weight Variant | act-order | ActQ Descriptor |
|---|-------|---------------|-----------|-----------------|
| 1 | smooth_ta_r16_Anone | smooth_ta_r16 | ✅ ON | none |
| 2 | smooth_ta_r16_A8 | smooth_ta_r16 | ✅ ON | int8 |
| 3 | smooth_ta_r16_A4g128 | smooth_ta_r16 | ✅ ON | int4-g128 |
| 4 | smooth_ta_r16_A4g128_downA8 | smooth_ta_r16 | ✅ ON | int4-g128+down_proj:int8 |
| 5 | smooth_ta_r64_Anone | smooth_ta_r64 | ✅ ON | none |
| 6 | smooth_ta_r64_A8 | smooth_ta_r64 | ✅ ON | int8 |
| 7 | smooth_ta_r64_A4g128 | smooth_ta_r64 | ✅ ON | int4-g128 |
| 8 | smooth_ta_r64_A4g128_downA8 | smooth_ta_r64 | ✅ ON | int4-g128+down_proj:int8 |
| 9 | smooth_ta_r128_Anone | smooth_ta_r128 | ✅ ON | none |
| 10 | smooth_ta_r128_A8 | smooth_ta_r128 | ✅ ON | int8 |
| 11 | smooth_ta_r128_A4g128 | smooth_ta_r128 | ✅ ON | int4-g128 |
| 12 | smooth_ta_r128_A4g128_downA8 | smooth_ta_r128 | ✅ ON | int4-g128+down_proj:int8 |
| 13 | smooth_ta_r16_noact_Anone | smooth_ta_r16_noact | ❌ OFF | none |
| 14 | smooth_ta_r16_noact_A8 | smooth_ta_r16_noact | ❌ OFF | int8 |
| 15 | smooth_ta_r16_noact_A4g128 | smooth_ta_r16_noact | ❌ OFF | int4-g128 |
| 16 | smooth_ta_r16_noact_A4g128_downA8 | smooth_ta_r16_noact | ❌ OFF | int4-g128+down_proj:int8 |
| 17 | smooth_ta_r64_noact_Anone | smooth_ta_r64_noact | ❌ OFF | none |
| 18 | smooth_ta_r64_noact_A8 | smooth_ta_r64_noact | ❌ OFF | int8 |
| 19 | smooth_ta_r64_noact_A4g128 | smooth_ta_r64_noact | ❌ OFF | int4-g128 |
| 20 | smooth_ta_r64_noact_A4g128_downA8 | smooth_ta_r64_noact | ❌ OFF | int4-g128+down_proj:int8 |
| 21 | smooth_ta_r128_noact_Anone | smooth_ta_r128_noact | ❌ OFF | none |
| 22 | smooth_ta_r128_noact_A8 | smooth_ta_r128_noact | ❌ OFF | int8 |
| 23 | smooth_ta_r128_noact_A4g128 | smooth_ta_r128_noact | ❌ OFF | int4-g128 |
| 24 | smooth_ta_r128_noact_A4g128_downA8 | smooth_ta_r128_noact | ❌ OFF | int4-g128+down_proj:int8 |

---

## 7. One-Click Shell Script

Save the following as `run_v7_smooth_tail_absorb.sh` in the project root and execute with `bash run_v7_smooth_tail_absorb.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# W4A Benchmark V7 — Smooth + Tail Absorb (r16 / r64 / r128)
#                     × act-order ON/OFF ablation
# Generates 6 weight variants + runs 24 PPL evaluations
# ============================================================

# --- Path Configuration ---
export MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
export PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
export OUTPUT_DIR="${PROJECT_DIR}/output/benchmark"
export WIKITEXT2_DIR="${PROJECT_DIR}/data/wikitext2"

# NOTE: V7 uses qwen3_gptq_tail_absorb.py (NOT qwen3_gptq_percentile_tail_spill.py)
export QUANT_SCRIPT="${PROJECT_DIR}/qwen3_gptq_tail_absorb.py"
export EVAL_SCRIPT="${PROJECT_DIR}/benchmark/eval_ppl.py"
export SMOOTH_STATE_DICT="${PROJECT_DIR}/output/smooth/smoothed_model_state_dict.pt"
export RESULTS_FILE="results_smooth_tail_absorb.txt"

export W_RAW_GPTQ="${PROJECT_DIR}/output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_GPTQ="${PROJECT_DIR}/output/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt"

export W_SMOOTH_TA_R16="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r16/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R64="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r64/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R128="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r128/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R16_NOACT="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r16_noact/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R64_NOACT="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r64_noact/qwen3-4b-instruct-2507-gptq-4bit.pt"
export W_SMOOTH_TA_R128_NOACT="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r128_noact/qwen3-4b-instruct-2507-gptq-4bit.pt"

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
# Phase 1: Generate 6 Weight Variants (3 rank × 2 act-order)
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

# --- act-order ON (default, do NOT pass --no-act-order) ---

# Variant 1: rank=16, act-order ON
run_quant "Smooth + Tail Absorb r16 (act-order ON)" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r16 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 16

# Variant 2: rank=64, act-order ON
run_quant "Smooth + Tail Absorb r64 (act-order ON)" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r64 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 64

# Variant 3: rank=128, act-order ON
run_quant "Smooth + Tail Absorb r128 (act-order ON)" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r128 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --tail-rank 128

# --- act-order OFF (pass --no-act-order) ---

# Variant 4: rank=16, act-order OFF
run_quant "Smooth + Tail Absorb r16 (act-order OFF)" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r16_noact \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --no-act-order \
    --tail-rank 16

# Variant 5: rank=64, act-order OFF
run_quant "Smooth + Tail Absorb r64 (act-order OFF)" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r64_noact \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --no-act-order \
    --tail-rank 64

# Variant 6: rank=128, act-order OFF
run_quant "Smooth + Tail Absorb r128 (act-order OFF)" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r128_noact \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --true-sequential \
    --use-standard-quantizer \
    --no-act-order \
    --tail-rank 128

echo ""
echo "================================================================"
echo "  [Phase 1] ALL ${PHASE1_TOTAL} WEIGHT VARIANTS GENERATED"
echo "================================================================"

# Verify Phase 1 outputs
echo "=== Phase 1 Output Verification ==="
for rank in 16 64 128; do
    for suffix in "" "_noact"; do
        dir="${PROJECT_DIR}/output/exp_smooth_tail_absorb/from_smooth_r${rank}${suffix}"
        pt="${dir}/qwen3-4b-instruct-2507-gptq-4bit.pt"
        meta="${dir}/metadata.json"
        if [ -f "${pt}" ] && [ -f "${meta}" ]; then
            echo "[OK] from_smooth_r${rank}${suffix}: weights + metadata found"
        else
            echo "[FAIL] from_smooth_r${rank}${suffix}: missing files in ${dir}"
            exit 1
        fi
    done
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

# === Group 1: smooth_ta_r16 (act-order ON) ===
run_exp "smooth_ta_r16 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ta_r16_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r16 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ta_r16_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r16 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ta_r16_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r16 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R16} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ta_r16_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 2: smooth_ta_r64 (act-order ON) ===
run_exp "smooth_ta_r64 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ta_r64_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r64 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ta_r64_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r64 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ta_r64_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r64 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R64} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ta_r64_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 3: smooth_ta_r128 (act-order ON) ===
run_exp "smooth_ta_r128 | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ta_r128_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r128 | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ta_r128_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r128 | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ta_r128_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r128 | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R128} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ta_r128_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 4: smooth_ta_r16_noact (act-order OFF) ===
run_exp "smooth_ta_r16_noact | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R16_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ta_r16_noact_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r16_noact | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R16_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ta_r16_noact_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r16_noact | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R16_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ta_r16_noact_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r16_noact | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R16_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ta_r16_noact_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 5: smooth_ta_r64_noact (act-order OFF) ===
run_exp "smooth_ta_r64_noact | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R64_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ta_r64_noact_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r64_noact | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R64_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ta_r64_noact_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r64_noact | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R64_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ta_r64_noact_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r64_noact | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R64_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ta_r64_noact_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Group 6: smooth_ta_r128_noact (act-order OFF) ===
run_exp "smooth_ta_r128_noact | none" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R128_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant none --label smooth_ta_r128_noact_Anone --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r128_noact | int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R128_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int8 --label smooth_ta_r128_noact_A8 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r128_noact | int4-g128" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R128_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --label smooth_ta_r128_noact_A4g128 --results-file ${RESULTS_FILE}

run_exp "smooth_ta_r128_noact | int4-g128+down:int8" \
    --model-dir ${MODEL_DIR} --quant-weights ${W_SMOOTH_TA_R128_NOACT} --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --act-quant int4-g128 --act-quant-override down_proj:int8 \
    --label smooth_ta_r128_noact_A4g128_downA8 --results-file ${RESULTS_FILE}

# === Done ===
echo ""
echo "================================================================"
echo "  ALL TASKS COMPLETED"
echo "  Phase 1: ${PHASE1_TOTAL} weight variants generated"
echo "  Phase 2: ${PHASE2_TOTAL} PPL evaluations completed"
echo "  Results: ${OUTPUT_DIR}/${RESULTS_FILE}"
echo "================================================================"
