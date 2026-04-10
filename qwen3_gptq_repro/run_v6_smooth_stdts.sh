#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# W4A Benchmark V6 — Smooth + Standard Tail Spill (r16 / r64 / r128)
# 从 SmoothQuant(alpha=1) 后的权重出发，生成 3 个量化变体 + 运行 24 个 PPL 评估
# ============================================================

# --- 路径配置 ---
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

# --- 预检查 ---
echo "=== Pre-flight Checks ==="
for f in "${QUANT_SCRIPT}" "${EVAL_SCRIPT}" "${W_RAW_GPTQ}" "${W_SMOOTH_GPTQ}" "${SMOOTH_STATE_DICT}"; do
    if [ ! -f "$f" ]; then echo "[FAIL] Missing: $f"; exit 1; fi
done
for d in "${MODEL_DIR}" "${WIKITEXT2_DIR}"; do
    if [ ! -d "$d" ]; then echo "[FAIL] Missing dir: $d"; exit 1; fi
done
echo "[OK] All pre-flight checks passed."

# ============================================================
# Phase 1: 生成 3 个量化权重变体（基于 Smooth 后权重）
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

# 变体 A: Smooth + Standard Tail Spill, rank=16
run_quant "Smooth + Standard Tail Spill r16" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r16 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --use-standard-quantizer \
    --tail-rank 16

# 变体 B: Smooth + Standard Tail Spill, rank=64
run_quant "Smooth + Standard Tail Spill r64" \
    --model-dir ${MODEL_DIR} \
    --output-dir ${PROJECT_DIR}/output/exp_standard_tail_spill/from_smooth_r64 \
    --local-wikitext2-dir ${WIKITEXT2_DIR} \
    --init-state-dict ${SMOOTH_STATE_DICT} \
    --wbits 4 --nsamples 128 --seqlen 2048 --groupsize 128 --percdamp 0.01 \
    --act-order --true-sequential \
    --use-standard-quantizer \
    --tail-rank 64

# 变体 C: Smooth + Standard Tail Spill, rank=128
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

# 验证 Phase 1 输出
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
# Phase 2: PPL 评估（24 个实验）
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

# === Group 1: fp16_baseline（无 --quant-weights）===
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

# === 完成 ===
echo ""
echo "================================================================"
echo "  ALL TASKS COMPLETED"
echo "  Phase 1: ${PHASE1_TOTAL} weight variants generated (from Smooth)"
echo "  Phase 2: ${PHASE2_TOTAL} PPL evaluations completed"
echo "  Results: ${OUTPUT_DIR}/${RESULTS_FILE}"
echo "================================================================"
