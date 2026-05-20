# Qwen3 Smooth Block Mixed Quantization

> **SmoothQuant alpha search + block-mask selection + W4 FakeQuant pipeline for Qwen3, with PPL / LM-Eval evaluation and plotting scripts.**

## Highlights

- **Main workflow**: download Qwen3, run `qwen3_smooth_block_mixed.py` for Smooth + Block Mixed Fake quantization, then evaluate with PPL and LM-Eval.
- **FakeQuant weights**: quantized weights are saved as a normal PyTorch `state_dict`, so evaluation loads the original HuggingFace model structure and replaces its weights.
- **Diagnostics included**: `smooth_groups.pt`, metadata JSON, optional layer output errors, Hessian-sensitive block observation plots, and paper-style performance figures.
- **Benchmarks**: WikiText-2 perplexity via `benchmark/eval_ppl.py`, and zero-shot task accuracy via `benchmark/eval_lm_eval.py`.

## Quick Start

```bash
conda create -n qwen3-gptq python=3.10 -y
conda activate qwen3-gptq
pip install torch transformers datasets sentencepiece accelerate huggingface_hub openpyxl
pip install "lm_eval[hf]"  # needed for benchmark/eval_lm_eval.py
```

```bash
cd /root/autodl-tmp/Zip
python download/dl_qwen3_4b.py
cd /root/autodl-tmp/Zip/qwen3_gptq_repro

export MODEL_DIR=/root/autodl-tmp/Zip/models/Qwen3-4B-Instruct-2507
export OUT_DIR=/root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_block_mixed_example

python qwen3_smooth_block_mixed.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUT_DIR" \
  --nsamples 128 \
  --seqlen 1024 \
  --block-rows 16 \
  --block-cols 128 \
  --groupsize 64 \
  --budget-ratio 0.2 \
  --second-path residual_int4 \
  --act-order \
  --percdamp 0.01 \
  --max-search-batches 128 \
  --save-layer-output-errors
```

👉 **[Experiment Guide](qwen3_gptq_repro/docs/EXPERIMENT_GUIDE.md)** — 从下载模型、Fake 量化到 PPL / LM-Eval 测评的完整流程

👉 **[Plotting README](qwen3_gptq_repro/docs/PLOTTING_README.md)** — `observation/` 和 `figure/` 中所有画图脚本的用途说明


## Project Structure

```
Zip/
├── model.py                          # Qwen3 model definition
├── gptq/                             # GPTQ core algorithm library
├── download/                         # Model download scripts
│   ├── dl_qwen3_4b.py                # Download Qwen3-4B-Instruct-2507
│   └── dl_modelscope_qwen3.py        # Download Qwen3-8B with ModelScope
└── qwen3_gptq_repro/
    ├── qwen3_gptq.py                 # Baseline GPTQ quantization
    ├── qwen3_smooth.py               # Fixed alpha=1.0 SmoothQuant preprocessing
    ├── qwen3_smooth_block_mixed.py   # Main Smooth + Block Mixed FakeQuant pipeline
    ├── qwen3_gptq_tail_absorb.py     # V7 Tail Absorb
    ├── gptq_tail_absorb.py           # V7 core quantizer
    ├── qwen3_gptq_submatrix_mixed.py # V9 Submatrix Mixed Precision
    ├── gptq_submatrix_mixed.py       # V9 core quantizer
    ├── infer_quantized_qwen3.py      # Inference script
    ├── benchmark/eval_ppl.py         # PPL evaluation
    ├── benchmark/eval_lm_eval.py     # LM-Eval zero-shot evaluation
    ├── observation/                  # Hessian-sensitive block diagnostic plots
    ├── figure/                       # Paper/report performance figure scripts
    ├── docs/                         # Documentation
    └── output/                       # Quantization, benchmark, and plot outputs
```

## Requirements

- Python 3.10+
- Core packages: `torch`, `transformers`, `datasets`, `sentencepiece`, `accelerate`, `huggingface_hub`, `openpyxl`
- LM-Eval: install `lm_eval[hf]` if running `benchmark/eval_lm_eval.py`
- Optional ModelScope support: install `modelscope` if using `download/dl_modelscope_qwen3.py`
- GPU recommended for quantization and observation scripts; CPU evaluation is possible but slow

## Key Results (Qwen3-4B, WikiText-2 PPL)

The current README flow centers on `qwen3_smooth_block_mixed.py`. Its main artifacts are:

| Artifact | Meaning |
|---|---|
| `qwen3-smooth-block-mixed-state_dict.pt` | Smooth + Block Mixed FakeQuant model weights for evaluation. |
| `smooth_groups.pt` | Smooth group alpha / scale / mask information used by observation scripts. |
| `qwen3_smooth_block_mixed_metadata.json` | Run arguments, calibration source, search summary, and quantization summary. |
| `layer_output_errors.json` | Optional per-Linear output error diagnostics from `--save-layer-output-errors`. |
| `output/benchmark/ppl_<label>.json` | PPL result from `benchmark/eval_ppl.py`. |
| `output/lm_eval/lm_eval_<label>.json` | Zero-shot result from `benchmark/eval_lm_eval.py`. |

For historical V7/V9 results and bug notes, see `qwen3_gptq_repro/docs/POSTMORTEM_V7_V9_CLIP_BUG.md`.

## Acknowledgements

- [GPTQ](https://github.com/IST-DASLab/gptq) — Frantar et al., 2022
- [SmoothQuant](https://github.com/mit-han-lab/smoothquant) — Xiao et al., 2023
- [Qwen3](https://github.com/QwenLM/Qwen) — Alibaba Cloud
