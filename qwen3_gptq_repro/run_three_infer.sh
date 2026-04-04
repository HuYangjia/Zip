#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507"
PROJECT_DIR="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro"
OUTPUT_DIR="${PROJECT_DIR}/output"

PROMPT="${1:-请用三句话解释大模型量化的核心思想}"
MAX_NEW_TOKENS="${2:-256}"

cd "${PROJECT_DIR}"

python infer_quantized_qwen3.py \
  --model-dir "${MODEL_DIR}" \
  --quantized-weights "${OUTPUT_DIR}/smooth/smoothed_model_state_dict.pt" \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" | tee "${OUTPUT_DIR}/smooth_infer.txt"

python infer_quantized_qwen3.py \
  --model-dir "${MODEL_DIR}" \
  --quantized-weights "${OUTPUT_DIR}/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" | tee "${OUTPUT_DIR}/gptq_from_smooth_infer.txt"

python infer_quantized_qwen3.py \
  --model-dir "${MODEL_DIR}" \
  --quantized-weights "${OUTPUT_DIR}/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt" \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" | tee "${OUTPUT_DIR}/gptq_from_raw_infer.txt"

echo
echo "Done. Outputs saved to:"
echo "  ${OUTPUT_DIR}/smooth_infer.txt"
echo "  ${OUTPUT_DIR}/gptq_from_smooth_infer.txt"
echo "  ${OUTPUT_DIR}/gptq_from_raw_infer.txt"
