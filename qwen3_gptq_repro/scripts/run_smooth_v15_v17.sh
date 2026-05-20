#!/usr/bin/env bash
# =============================================================================
# Smooth Block Mixed V15-V17
# -----------------------------------------------------------------------------
# Runs:
#   block-rows=16 -> output/smooth_v15_b32
#   block-rows=32 -> output/smooth_v16_b32
#   block-rows=64 -> output/smooth_v17_b32
# =============================================================================

set -eE -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="/root/autodl-tmp/model/Qwen3-4B-Instruct-2507"
LOG_DIR="${SCRIPT_DIR}/logs/smooth_v15_v17"

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

BLOCK_ROWS_LIST=(16 32 64)
OUTPUT_NAMES=(smooth_v15_b32 smooth_v16_b32 smooth_v17_b32)

log_info "============================================================"
log_info "Smooth V15-V17 启动 | timestamp=${TIMESTAMP}"
log_info "REPRO_DIR=${REPRO_DIR}"
log_info "MODEL_DIR=${MODEL_DIR}"
log_info "LOG_DIR=${LOG_DIR}"
log_info "Python: $(which python) | $(python -V 2>&1)"
log_info "CWD=$(pwd)"
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
        --nsamples 128 \
        --seqlen 1024 \
        --block-rows "${block_rows}" \
        --block-cols 128 \
        --groupsize 64 \
        --save-layer-output-errors \
        --second-path residual_int4 \
        --act-order \
        --percdamp 0.01 \
        --budget-ratio 0.2 \
        --max-search-batches 128 \
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
