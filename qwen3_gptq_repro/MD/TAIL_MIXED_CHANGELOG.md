# Tail Mixed GPTQ Changelog

## Scope

Track all code-level changes for the tail mixed-precision GPTQ experiment.

## Files Added

- `qwen3_gptq_tail_mixed.py`
- `MD/TAIL_MIXED_EXPERIMENT_PLAN.md`
- `MD/TAIL_MIXED_CHANGELOG.md`
- `MD/TAIL_MIXED_RESULTS.md`

## File-by-File Change Record

### `qwen3_gptq_tail_mixed.py`

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

## Experiment Command Log

Commands and outputs are recorded in `MD/TAIL_MIXED_RESULTS.md`.
