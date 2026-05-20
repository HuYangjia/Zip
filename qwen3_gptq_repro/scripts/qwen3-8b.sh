#!/usr/bin/env bash
# =============================================================================
# Smooth Block Mixed V1-V3
# -----------------------------------------------------------------------------
# Runs:
#   block-rows=16 -> output/qwen3_8b_v1
#   block-rows=32 -> output/qwen3_8b_v2
#   block-rows=64 -> output/qwen3_8b_v3
# =============================================================================

set -eE -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="/root/autodl-tmp/model/Qwen3-8B"
LOG_DIR="${SCRIPT_DIR}/logs/qwen3_8b_v1_v3"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/main_${TIMESTAMP}.log"

log_info() {
    printf "[%s][INFO]  %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MAIN_LOG}"
}

log_error() {
    printf "[%s][ERROR] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MAIN_LOG}" >&2
}

on_error() {
    log_error "脚本在第 $1 行失败，退出码 $2。详见 ${MAIN_LOG}"
}
trap 'on_error $LINENO $?' ERR

if [[ ! -d "${MODEL_DIR}" ]]; then
    log_error "模型目录不存在: ${MODEL_DIR}"
    exit 1
fi

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "zip" ]]; then
    log_info "检测到未激活 conda env 'zip'，尝试自动激活..."
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate zip
fi

cd "${REPRO_DIR}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-1024}"
MAX_SEARCH_BATCHES="${MAX_SEARCH_BATCHES:-128}"
SEARCH_EVAL_BATCH_SIZE="${SEARCH_EVAL_BATCH_SIZE:-1}"
GPTQ_BATCH_SIZE="${GPTQ_BATCH_SIZE:-1}"
OUTPUT_ERROR_BATCH_SIZE="${OUTPUT_ERROR_BATCH_SIZE:-1}"

BLOCK_ROWS_LIST=(16 32 64)
OUTPUT_NAMES=(qwen3_8b_v1 qwen3_8b_v2 qwen3_8b_v3)

log_info "============================================================"
log_info "Smooth V1-V3 启动 | timestamp=${TIMESTAMP}"
log_info "REPRO_DIR=${REPRO_DIR}"
log_info "MODEL_DIR=${MODEL_DIR}"
log_info "LOG_DIR=${LOG_DIR}"
log_info "Python: $(which python) | $(python -V 2>&1)"
log_info "CWD=$(pwd)"
log_info "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}"
log_info "nsamples=${NSAMPLES} seqlen=${SEQLEN} max_search_batches=${MAX_SEARCH_BATCHES}"
log_info "search_eval_batch_size=${SEARCH_EVAL_BATCH_SIZE} gptq_batch_size=${GPTQ_BATCH_SIZE} output_error_batch_size=${OUTPUT_ERROR_BATCH_SIZE}"
log_info "============================================================"

run_one() {
    local block_rows="$1"
    local output_name="$2"
    local output_dir="${REPRO_DIR}/output/${output_name}"
    local sub_log="${LOG_DIR}/${output_name}_block_rows_${block_rows}.log"

    mkdir -p "${output_dir}"

    log_info ">>> [START] ${output_name} | block_rows=${block_rows}"
    log_info "    output_dir=${output_dir}"
    log_info "    sub_log=${sub_log}"

    local t0
    t0="$(date +%s)"

    if python qwen3_smooth_block_mixed.py \
        --model-dir "${MODEL_DIR}" \
        --output-dir "${output_dir}" \
        --nsamples "${NSAMPLES}" \
        --seqlen "${SEQLEN}" \
        --block-rows "${block_rows}" \
        --block-cols 128 \
        --groupsize 32 \
        --save-layer-output-errors \
        --output-error-batch-size "${OUTPUT_ERROR_BATCH_SIZE}" \
        --second-path residual_int4 \
        --act-order \
        --percdamp 0.01 \
        --budget-ratio 0.2 \
        --max-search-batches "${MAX_SEARCH_BATCHES}" \
        --search-eval-batch-size "${SEARCH_EVAL_BATCH_SIZE}" \
        --gptq-batch-size "${GPTQ_BATCH_SIZE}" \
        >"${sub_log}" 2>&1; then
        local t1
        t1="$(date +%s)"
        log_info "<<< [DONE ] ${output_name} | elapsed=$((t1 - t0))s"
    else
        local rc=$?
        local t1
        t1="$(date +%s)"
        log_error "<<< [FAIL ] ${output_name} | rc=${rc} | elapsed=$((t1 - t0))s | see ${sub_log}"
        return "${rc}"
    fi
}

for idx in "${!BLOCK_ROWS_LIST[@]}"; do
    run_one "${BLOCK_ROWS_LIST[$idx]}" "${OUTPUT_NAMES[$idx]}"
done

log_info "全部完成。主日志: ${MAIN_LOG}"
