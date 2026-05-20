# 绘图代码说明

本文简单说明两个目录中的画图脚本：

```text
/root/autodl-tmp/Zip/qwen3_gptq_repro/observation
/root/autodl-tmp/Zip/qwen3_gptq_repro/figure
```

两者定位不同：

| 目录 | 作用 | 数据来源 |
|---|---|---|
| `observation/` | 重新跑模型、校准数据和量化诊断，画 Hessian-sensitive blocks 相关观察图。 | Qwen3 模型、`smooth_groups.pt`、WikiText2 校准数据、诊断中间结果。 |
| `figure/` | 画论文/报告中的性能展示图。 | 多数数据直接写在脚本里，例如 throughput、latency、overhead 数值。 |

---

## 1. observation 目录

`observation/` 里的脚本主要服务于 Hessian-sensitive blocks 的观察和验证。它们通常需要 GPU、模型目录和 smooth 量化输出。

常见输入：

```text
--model-dir          原始 Qwen3 模型目录
--smooth-groups      qwen3_smooth_block_mixed.py 输出的 smooth_groups.pt
--local-wikitext2-dir 可选，本地 WikiText2 数据目录
--output-dir         图像和 JSON/CSV 诊断结果保存目录
```

### 1.1 prove_hessian_sensitive_blocks.py

路径：

```text
observation/prove_hessian_sensitive_blocks.py
```

作用：验证 W4A4 量化误差是否集中在 Hessian-sensitive blocks 上。

主要画图：

| 输出图 | 含义 |
|---|---|
| `plots/concentration_curves.png` | top-k blocks 捕获 exact output error mass 的集中度曲线。 |
| `plots/repair_curves.png` | 按 Hessian/Frobenius/channel/delta/random 选择 block 后，剩余误差比例如何下降。 |
| `plots/score_correlation.png` | Hessian/Frobenius/channel score 与 exact output score 的相关性散点图。 |
| `plots/heatmaps/*.png` | 单个模块的 block-level heatmap，对比 exact output score 和 Hessian diag score。 |

示例：

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro

python observation/prove_hessian_sensitive_blocks.py \
  --smooth-groups output/smooth_v16_b32/smooth_groups.pt \
  --module-filter q_proj,v_proj \
  --layer-filter 24,34 \
  --block-rows 16 \
  --block-cols 128 \
  --groupsize 32 \
  --output-dir observation/output/hessian_sensitive_blocks_example
```

### 1.2 prove_hessian_sensitive_blocks_obs1_layer24_34_vproj.py

路径：

```text
observation/prove_hessian_sensitive_blocks_obs1_layer24_34_vproj.py
```

作用：专门复现 Observation 1 中 Layer 24 和 Layer 34 的 `self_attn.v_proj` 对比图。

主要画图：

| 输出图 | 含义 |
|---|---|
| `plots/observation1/observation1_layers_24_34_self_attn_v_proj.png` | 横向双面板 heatmap，左/右分别是 Layer 24 和 Layer 34 的 `v_proj` block output distortion。 |
| `plots/observation1/observation1_layers_24_34_self_attn_v_proj.pdf` | 同一张图的 PDF 版本。 |

它也会保留基础诊断图，例如 `concentration_curves.png`、`repair_curves.png`、`score_correlation.png` 和 `heatmaps/*.png`。

示例：

```bash
python observation/prove_hessian_sensitive_blocks_obs1_layer24_34_vproj.py \
  --smooth-groups output/smooth_v16_b32/smooth_groups.pt \
  --obs1-fig-width 10 \
  --obs1-fig-height 5 \
  --obs1-font-size 18 \
  --obs1-format both \
  --output-dir observation/output/hessian_sensitive_blocks_obs1_layer24_34_vproj
```

### 1.3 diagnose_hessian_sensitive_blocks_per_layer.py

路径：

```text
observation/diagnose_hessian_sensitive_blocks_per_layer.py
```

作用：在全局和逐层两个粒度上画 Observation 2/3 selection curve。

主要画图：

| 输出图 | 含义 |
|---|---|
| `plots/observation23/observation23_selection_curve.png` | 全局 selection curve。 |
| `plots/observation23/observation23_selection_curve.pdf` | 全局 selection curve 的 PDF 版本。 |
| `plots/observation23/by_layer/layer_XX_observation23_selection_curve.png` | 每一层单独的 selection curve。 |
| `plots/observation23/by_layer/layer_XX_observation23_selection_curve.pdf` | 每一层图的 PDF 版本。 |

这类曲线的横轴是 selected block fraction，纵轴是 captured error reduction / remaining error 相关指标；曲线通常比较 output distortion oracle、weight outlier、residual norm 和 random。

示例：

```bash
python observation/diagnose_hessian_sensitive_blocks_per_layer.py \
  --smooth-groups output/smooth_v16_b32/smooth_groups.pt \
  --module-filter v_proj \
  --layer-filter 0-35 \
  --block-rows 16 \
  --block-cols 128 \
  --groupsize 32 \
  --plot-observation23 \
  --obs23-max-fraction 0.20 \
  --obs23-format both \
  --output-dir observation/output/hessian_sensitive_blocks_vproj_l0_35_example_per_layer
```

### 1.4 plot_vproj_layer24_layer34_observation23.py

路径：

```text
observation/plot_vproj_layer24_layer34_observation23.py
```

作用：不重新跑模型，只读取 `diagnose_hessian_sensitive_blocks_per_layer.py` 生成的 `observation23_summary.json`，把 Layer 24 和 Layer 34 的 `v_proj` selection curve 画成一张双面板图。

主要输入：

```text
observation23_summary.json
```

主要画图：

| 输出图 | 含义 |
|---|---|
| `v_proj_layer24_layer34_observation23_selection_curve.png` | Layer 24/34 双面板 Observation 2/3 曲线。 |
| `v_proj_layer24_layer34_observation23_selection_curve.pdf` | 同一张图的 PDF 版本。 |

示例：

```bash
python observation/plot_vproj_layer24_layer34_observation23.py \
  --summary-json observation/output/hessian_sensitive_blocks_vproj_l0_35_smooth_v16_b32x128_g128_per_layer/observation23_summary.json \
  --layers 24,34 \
  --module-name v_proj \
  --max-fraction 0.20 \
  --format both
```

---

## 2. figure 目录

`figure/` 里的脚本主要用于生成论文/汇报中的性能图。它们大多不依赖模型运行结果，而是直接使用脚本内写好的数值数组。

运行方式通常是：

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro
python figure/<script_name>.py
```

### 2.1 naive_kernel_fusion.py

路径：

```text
figure/naive_kernel_fusion.py
```

作用：画 naive kernel 和 optimized kernel 的 latency breakdown 堆叠柱状图。

图像内容：

| 对比对象 | 组成 |
|---|---|
| `Naive` | Quant、Dense GEMM、Sparse GEMM、Reduce Add |
| `Ours` | Quant、Fused dense+sparse |

输出：

```text
figure/latency_breakdown.pdf
figure/latency_breakdown.png
```

### 2.2 latency.py

路径：

```text
figure/latency.py
```

作用：画两个环境下不同 batch size 的 throughput 柱状图，并使用 broken axis 展示较大的 TwinQuant 数值。

图像内容：

| 维度 | 内容 |
|---|---|
| 横轴 | `bz=1,2,4,8` |
| 方法 | TensorRT-LLM、AWQ、QuaRot、FlatQuant、TwinQuant |
| 面板 | Environment 1 和 Environment 2 |

输出：

```text
figure/throughput_broken_axis.pdf
figure/throughput_broken_axis.png
```

### 2.3 overhead.py

路径：

```text
figure/overhead.py
```

作用：画不同模型上的 memory overhead 和 latency overhead。

图像内容：

| 面板 | 含义 |
|---|---|
| Memory Overhead | 不同模型的显存开销百分比。 |
| Latency Overhead | 不同模型的延迟开销百分比。 |

模型包括 LLaMA3-3B、LLaMA3-8B、Qwen3-4B、Qwen3-8B、Qwen3-14B、Qwen3-32B。

输出：

```text
figure/low_rank_overhead.pdf
figure/low_rank_overhead.png
```

### 2.4 various_budget.py

路径：

```text
figure/various_budget.py
```

作用：画不同 budget 下 memory overhead 和 latency overhead 的变化。

图像内容：

| 横向类别 | Budget |
|---|---|
| `Budget=1%` | 1% |
| `Budget=5%` | 5% |
| `Budget=10%` | 10% |
| `Budget=20%` | 20% |

输出：

```text
budget_overhead.pdf
budget_overhead.png
```

注意：这个脚本当前保存路径是相对当前运行目录的文件名，不是自动保存到 `figure/`。如果你在项目根目录运行，输出会出现在：

```text
/root/autodl-tmp/Zip/qwen3_gptq_repro/budget_overhead.pdf
/root/autodl-tmp/Zip/qwen3_gptq_repro/budget_overhead.png
```

如果你想让它保存到 `figure/`，可以在 `figure/` 目录下运行：

```bash
cd /root/autodl-tmp/Zip/qwen3_gptq_repro/figure
python various_budget.py
```

### 2.5 throughput.py

路径：

```text
figure/throughput.py
```

作用：画 Qwen3-8B mixed workload 的端到端 throughput 柱状图。

图像内容：

| 维度 | 内容 |
|---|---|
| workload | 1 次 prefill + 128 次 decode |
| batch size | 4、8、16、32 |
| 方法 | TensorRT-LLM、ResQ、AWQ、QuaRot、MosaicQuant |
| 标注 | MosaicQuant 相对最强 baseline 的 speedup |

输出：

```text
figure/kernel_mixed_128_optimized.pdf
figure/kernel_mixed_128_optimized.png
```

---

## 3. 简单对应关系

| 想画的内容 | 推荐脚本 |
|---|---|
| Hessian-sensitive blocks 是否集中 | `observation/prove_hessian_sensitive_blocks.py` |
| Observation 1 的 Layer 24/34 `v_proj` heatmap | `observation/prove_hessian_sensitive_blocks_obs1_layer24_34_vproj.py` |
| Observation 2/3 全局和逐层 selection curve | `observation/diagnose_hessian_sensitive_blocks_per_layer.py` |
| 只用已有 JSON 重画 Layer 24/34 selection curve | `observation/plot_vproj_layer24_layer34_observation23.py` |
| kernel fusion latency breakdown | `figure/naive_kernel_fusion.py` |
| 两个环境下的 throughput broken-axis 图 | `figure/latency.py` |
| 不同模型的 memory/latency overhead | `figure/overhead.py` |
| 不同 budget 的 overhead | `figure/various_budget.py` |
| Qwen3-8B mixed workload throughput | `figure/throughput.py` |

