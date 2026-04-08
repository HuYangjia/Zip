# Tail Mixed Experiment Record

> 本文件合并自以下三个原始文件：
> - `MD/TAIL_MIXED_EXPERIMENT_PLAN.md`
> - `MD/TAIL_MIXED_CHANGELOG.md`
> - `MD/TAIL_MIXED_RESULTS.md`

---

## Experiment Plan

### Objective

Validate whether preserving the last 5% input columns as INT8 (while the main 95% uses 4-bit GPTQ) reduces layer-level distortion and improves downstream behavior versus pure GPTQ-4bit.

### Fixed Setup

- Tail ratio: `0.05`
- Tail precision option: `int8`
- Budget: `nsamples=32`, `seqlen=1024`
- Baseline: original `qwen3_gptq.py` with 4-bit GPTQ
- Variant: `qwen3_gptq_tail_mixed.py` with `--enable-tail-mixed --tail-ratio 0.05 --tail-quant int8`

### Artifacts

- Baseline output dir: `output/exp_tail_mixed/baseline_gptq4/`
- Tail-mixed output dir: `output/exp_tail_mixed/tail5p_int8_gptq4/`

### Reproducibility Notes

- Keep model path, tokenizer, and calibration source identical between baseline and variant.
- Keep GPTQ hyperparameters identical between baseline and variant.
- Only tail behavior should differ.

---

## Changelog

### Scope

Track all code-level changes for the tail mixed-precision GPTQ experiment.

### Files Added

- `qwen3_gptq_tail_mixed.py`
- `MD/TAIL_MIXED_EXPERIMENT_PLAN.md`
- `MD/TAIL_MIXED_CHANGELOG.md`
- `MD/TAIL_MIXED_RESULTS.md`

### File-by-File Change Record

#### `qwen3_gptq_tail_mixed.py`

- Forked from `qwen3_gptq.py` into an isolated experiment script.
- Added CLI arguments for tail mixed precision:
  - `--enable-tail-mixed`
  - `--tail-ratio`
  - `--tail-quant` (currently `int8`)
- Added tail-column split logic per linear layer:
  - Main columns: GPTQ 4-bit
  - Tail columns: quantized/dequantized as INT8
- Added metadata fields to record:
  - whether tail mixed is enabled
  - tail ratio, tail quant type
  - per-layer tail range and scales

### Experiment Command Log

Commands and outputs are recorded in the Results section below.

---

## Results

### Experiment Matrix

1. Baseline: GPTQ-4bit (`qwen3_gptq.py`)
2. Variant: GPTQ-4bit + tail 5% INT8 (`qwen3_gptq_tail_mixed.py`)

### Shared Settings

- `nsamples=32`
- `seqlen=1024`
- `groupsize=128`
- `percdamp=0.01`
- `act-order=true`
- `true-sequential=true`

### Run Commands

#### Baseline

Pending.

#### Tail Mixed

Pending.

### Artifacts

#### Baseline

Pending.

#### Tail Mixed

Pending.

### Observations

Pending.

### Conclusion

Pending.
