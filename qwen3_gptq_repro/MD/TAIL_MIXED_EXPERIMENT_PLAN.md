# Tail Mixed GPTQ Experiment Plan

## Objective

Validate whether preserving the last 5% input columns as INT8 (while the main 95% uses 4-bit GPTQ) reduces layer-level distortion and improves downstream behavior versus pure GPTQ-4bit.

## Fixed Setup

- Tail ratio: `0.05`
- Tail precision option: `int8`
- Budget: `nsamples=32`, `seqlen=1024`
- Baseline: original `qwen3_gptq.py` with 4-bit GPTQ
- Variant: `qwen3_gptq_tail_mixed.py` with `--enable-tail-mixed --tail-ratio 0.05 --tail-quant int8`

## Artifacts

- Baseline output dir: `output/exp_tail_mixed/baseline_gptq4/`
- Tail-mixed output dir: `output/exp_tail_mixed/tail5p_int8_gptq4/`
- This doc set:
  - `MD/TAIL_MIXED_EXPERIMENT_PLAN.md`
  - `MD/TAIL_MIXED_CHANGELOG.md`
  - `MD/TAIL_MIXED_RESULTS.md`

## Reproducibility Notes

- Keep model path, tokenizer, and calibration source identical between baseline and variant.
- Keep GPTQ hyperparameters identical between baseline and variant.
- Only tail behavior should differ.
