# Qwen3-4B 单独 GPTQ 4-bit 复现步骤（zip环境）

目标：
- 只做 **4-bit GPTQ**（不做 smooth）；
- 优先使用 **WikiText2** 作为校准集；
- WikiText 失败时自动 fallback 到临时文本。

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

## 3) 运行 GPTQ（原始权重）

### 3.1 快速验证

```bash
cd /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro
python qwen3_gptq.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/gptq_from_raw \
  --wbits 4 \
  --nsamples 1 \
  --seqlen 256 \
  --groupsize 128 \
  --percdamp 0.01 \
  --act-order \
  --true-sequential
```

### 3.2 正式量化

```bash
cd /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro
python qwen3_gptq.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/gptq_from_raw \
  --wbits 4 \
  --nsamples 128 \
  --seqlen 2048 \
  --groupsize 128 \
  --percdamp 0.01 \
  --act-order \
  --true-sequential
```

> 注意：本脚本固定只支持 4-bit，`--wbits` 必须是 4。

## 4) WikiText2 失败时回退逻辑

1. 若设置 `--local-wikitext2-dir`，先读本地 `wikitext-train.arrow`；
2. 本地失败再尝试在线 `wikitext-2-raw-v1`；
3. 再失败则生成 `tmp_generated_calib.txt` 作为临时校准文本。

使用本地 WikiText2 示例：

```bash
cd /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro
python qwen3_gptq.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --output-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/gptq_from_raw \
  --local-wikitext2-dir /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/data/wikitext2 \
  --wbits 4 \
  --nsamples 128 \
  --seqlen 2048 \
  --groupsize 128 \
  --percdamp 0.01 \
  --act-order \
  --true-sequential
```

## 5) 输出文件

`output/gptq_from_raw/` 下主要有：
- `qwen3-4b-instruct-2507-gptq-4bit.pt`
- `qwen3_gptq_4bit_metadata.json`
- `tmp_generated_calib.txt`（仅 fallback 时存在）

## 6) 最小验证

```bash
python - <<'PY2'
import json
from pathlib import Path
meta = Path('/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/gptq_from_raw/qwen3_gptq_4bit_metadata.json')
print('meta exists:', meta.exists())
if meta.exists():
    data = json.loads(meta.read_text(encoding='utf-8'))
    print('wbits:', data.get('wbits'))
    print('calibration:', data.get('calibration', {}).get('calib_source'))
    print('fallback_used:', data.get('calibration', {}).get('fallback_used'))
    print('init_from_state_dict:', data.get('init_state_dict', {}).get('used'))
    print('num_quantized_linear_layers:', data.get('num_quantized_linear_layers'))
PY2
```

## 7) 常见问题

- `ModuleNotFoundError: No module named 'datasets'`：执行 `pip install -U datasets`
- `CUDA is required for this GPTQ script.`：检查 `torch.cuda.is_available()`
- 显存不足：降低 `--nsamples` 或 `--seqlen`
