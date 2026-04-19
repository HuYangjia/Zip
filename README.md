# Submatrix Mixed Precision GPTQ for Qwen3

> **4-bit GPTQ with adaptive INT8 submatrix blocks — achieving PPL within +0.23 of FP16.**

## Highlights

- **V9 Submatrix Mixed Precision**: Sensitivity-based block selection places INT8 budget where quantization error is highest
- **Best result**: Qwen3-4B PPL = **10.2786** (FP16 = 10.0449, gap = +0.23) with only 10% INT8 budget
- Three verified methods: Baseline GPTQ, V7 Tail Absorb, V9 Submatrix Mixed Precision
- Full benchmark results on WikiText-2 included

## Quick Start

```bash
conda create -n gptq python=3.10 -y && conda activate gptq
pip install torch>=2.1 transformers>=4.40 datasets sentencepiece
```

👉 **[Experiment Guide](qwen3_gptq_repro/docs/EXPERIMENT_GUIDE.md)** — 5 分钟上手运行所有实验

👉 **[AI Technical Reference](qwen3_gptq_repro/docs/AI_TECHNICAL_REFERENCE.md)** — 详细接口文档，供 AI 辅助复现

## Project Structure

```
Zip/
├── model.py                          # Qwen3 model definition
├── gptq/                             # GPTQ core algorithm library
├── download/                         # Model download scripts
└── qwen3_gptq_repro/
    ├── qwen3_gptq.py                 # Baseline GPTQ quantization
    ├── qwen3_smooth.py               # SmoothQuant preprocessing
    ├── qwen3_gptq_tail_absorb.py     # V7 Tail Absorb
    ├── gptq_tail_absorb.py           # V7 core quantizer
    ├── qwen3_gptq_submatrix_mixed.py # V9 Submatrix Mixed Precision
    ├── gptq_submatrix_mixed.py       # V9 core quantizer
    ├── infer_quantized_qwen3.py      # Inference script
    ├── benchmark/eval_ppl.py         # PPL evaluation
    ├── docs/                         # Documentation
    └── output/                       # Benchmark results (JSON)
```

## Requirements

- Python 3.10+
- PyTorch >= 2.1 (CUDA)
- transformers >= 4.40
- datasets, sentencepiece
- GPU: >= 16GB VRAM (Qwen3-4B)

## Key Results (Qwen3-4B, WikiText-2 PPL)

| Method | PPL (no ActQ) | Δ vs FP16 |
|---|---|---|
| FP16 Baseline | 10.0449 | — |
| GPTQ 4-bit | 10.3845 | +0.34 |
| V7 Tail Absorb (r16) | 10.3428 | +0.30 |
| **V9 Mixed (b64×128, r10%)** | **10.2786** | **+0.23** |

## Acknowledgements

- [GPTQ](https://github.com/IST-DASLab/gptq) — Frantar et al., 2022
- [SmoothQuant](https://github.com/mit-han-lab/smoothquant) — Xiao et al., 2023
- [Qwen3](https://github.com/QwenLM/Qwen) — Alibaba Cloud