# Qwen3-4B Smooth(alpha=1) + GPTQ 4-bit 复现步骤（zip环境）

目标：
- 先做 SmoothQuant（`alpha=1.0`）；
- 保存 smooth 后权重与 scale；
- 再基于 smooth 后权重执行 GPTQ。

## 1) 进入环境

```bash
conda activate zip
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 2) 安装依赖

```bash
cd /home/zhou/Documents/yangjia/zip
pip install -U torch torchvision torchaudio transformers datasets accelerate sentencepiece
```

## 3) 目录约定（建议）

- `output/smooth/`：smooth 中间产物
- `output/gptq_from_smooth/`：smooth 后再 GPTQ

## 4) 先运行 Smooth（仅 smooth）

```bash
cd /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro
python qwen3_smooth.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth \
  --alpha 1.0 \
  --nsamples 128 \
  --seqlen 2048
```

smooth 输出：
- `smoothed_model_state_dict.pt`
- `smooth_scales.pt`
- `smooth_metadata.json`

## 5) Smooth后、GPTQ前先做一次推理检查（可选）

```bash
cd /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro
python infer_quantized_qwen3.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --quantized-weights /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth/smoothed_model_state_dict.pt \
  --prompt "请简要介绍一下你自己" \
  --max-new-tokens 256
```

说明：
- 参数名是 `--quantized-weights`，但脚本本质加载的是 `state_dict`，可直接用于 smooth 后权重验证。

## 6) 再运行 GPTQ（基于 smooth 权重）

```bash
cd /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro
python qwen3_gptq.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/gptq_from_smooth \
  --init-state-dict /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth/smoothed_model_state_dict.pt \
  --smooth-scales-path /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth/smooth_scales.pt \
  --smooth-metadata-path /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth/smooth_metadata.json \
  --wbits 4 \
  --nsamples 128 \
  --seqlen 2048 \
  --groupsize 128 \
  --percdamp 0.01 \
  --act-order \
  --true-sequential
```

## 7) 调用原始 GPTQ 需要注意什么

当你从 smooth 结果切回“原始 GPTQ”时，注意以下几点：

- 不要传 `--init-state-dict`，否则就不是原始权重直量化。
- 不要传 `--smooth-scales-path` 和 `--smooth-metadata-path`，避免元数据混淆。
- 建议输出目录改回 `output/gptq_from_raw/`，确保两条实验路径不互相覆盖。
- 两条路径的 `--nsamples`、`--seqlen`、`--groupsize`、`--percdamp`、`--act-order`、`--true-sequential` 尽量保持一致，保证对比公平。
- `--wbits` 必须为 4（当前脚本固定约束）。

## 8) 验证 GPTQ 追踪字段

```bash
python - <<'PY2'
import json
from pathlib import Path
meta = Path('/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/gptq_from_smooth/qwen3_gptq_4bit_metadata.json')
print('meta exists:', meta.exists())
if meta.exists():
    data = json.loads(meta.read_text(encoding='utf-8'))
    print('init_from_state_dict:', data.get('init_state_dict', {}).get('used'))
    print('init_state_path:', data.get('init_state_dict', {}).get('path'))
    print('smooth_scales_path:', data.get('smooth_artifacts', {}).get('smooth_scales_path'))
    print('smooth_metadata_path:', data.get('smooth_artifacts', {}).get('smooth_metadata_path'))
PY2
```
