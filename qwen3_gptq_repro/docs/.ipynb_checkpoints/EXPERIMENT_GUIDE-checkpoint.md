# Experiment Guide — Submatrix Mixed Precision GPTQ

> 5 分钟快速上手，运行所有实验。

---

## 0. ⚠️ 铁律：V7 / V9 量化命令的 7 个必选参数

> 本节是 2026-04-23 CLIP-bug 事故后固化的防御规则，详情见
> [`POSTMORTEM_V7_V9_CLIP_BUG.md`](./POSTMORTEM_V7_V9_CLIP_BUG.md)。
>
> **在运行任何 V7 (`qwen3_gptq_tail_absorb.py`) 或 V9 (`qwen3_gptq_submatrix_mixed.py`)
> 命令时，下表 7 条参数必须全部显式出现**。历史上每一次漏掉任一条都导致 PPL 从 ~10.3
> 爆到 ~180~345。基线 `qwen3_gptq.py` 不受此影响（它只需要默认值即可）。

| # | 参数 | 必须写成 | 漏掉后的后果 |
|---|------|---------|-------------|
| 1 | `--nsamples` | `128` | 脚本默认 32，Hessian 欠估计 → PPL×17 |
| 2 | `--seqlen` | `2048` | 脚本默认 1024，同上 |
| 3 | `--groupsize` | `128` | 与 `block-cols`/g128 scale 对齐被破坏 |
| 4 | `--percdamp` | `0.01` | 默认已是 0.01，显式写出便于审计 |
| 5 | `--true-sequential`（V7） | 必传 | 层内投影残差传播顺序偏离基线语义 |
| 6 | `--use-standard-quantizer`（V7/V9） | 必传 | 否则走 `PercentileQuantizer(k=75)`，25% 权重被截断，`clip_ratio=0.2500` 恒定 |
| 7 | act-order | 默认开启（不传 `--no-act-order`） | 跳过 act-order 会让下游等效矩阵错位 |

> 🗑️ **历史遗留说明**：`PercentileQuantizer` 是早期实验代码，已被证明有 bug，
> 后续会从代码库移除。在它被移除前，第 6 条（`--use-standard-quantizer`）是**必须**的；
> 移除之后本条自动失效，但其余 6 条仍需保留。

---

## 1. Quick Start

### 环境配置

```bash
conda create -n gptq python=3.10 -y && conda activate gptq
pip install torch>=2.1 transformers>=4.40 datasets sentencepiece
```

### 下载模型

```bash
python download/dl_qwen3_4b.py  # 或手动下载 Qwen3-4B-Instruct-2507
```

### 运行第一个实验

```bash
cd qwen3_gptq_repro
#--------------------------------------------------------------------------
#初版量化+smooth
# Step 1: SmoothQuant 预处理
python qwen3_smooth.py --model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507 \
    --output-dir /root/autodl-tmp/output


# Step 2: V9 量化（最佳配置）
python qwen3_gptq_submatrix_mixed.py \
    --model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507 \
    --init-state-dict /root/autodl-tmp/model/smooth/smoothed_model_state_dict.pt \
    --output-dir /root/autodl-tmp/output/exp_submatrix_mixed/b128x128_r10_qe \
    --nsamples 128 --seqlen 1024 \
    --block-rows 128 --block-cols 128 --groupsize 128 --percdamp 0.01 \
    --budget-ratio 0.10 --sensitivity-metric quant_error \
    --save-layer-output-errors \
    --use-standard-quantizer

python scripts/analyze_output_error_losses.py \
  --metadata output/smooth_v1_b32/qwen3_smooth_block_mixed_metadata.json \
  --csv-dir output/smooth_v1_b32/loss_csv

#--------------------------------------------------------------------------
#Step 1：网格搜索smooth+GPTQ量化
cd /root/autodl-tmp/Zip/qwen3_gptq_repro

python qwen3_smooth_block_mixed.py \
  --model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507 \
  --output-dir /root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_v11_b32 \
  --nsamples 128 \
  --seqlen 1024 \
  --block-rows 128 \
  --block-cols 128 \
  --groupsize 32 \
  --save-layer-output-errors \
  --second-path residual_int4 \
  --save-layer-output-errors \
  --act-order \
  --percdamp 0.01 \
  --budget-ratio 0.2 \
  --max-search-batches 128

python scripts/analyze_output_error_losses.py \
  --metadata /root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_v3_b32/qwen3_smooth_block_mixed_metadata.json \
  --csv-dir /root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_v3_b32/loss_csv

/root/autodl-tmp/output/exp_submatrix_mixed/b128x128_r10_qe/qwen3-4b-instruct-2507-gptq-4bit.pt

/root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_v3_b32/qwen3-smooth-block-mixed-state_dict.pt

# Step 3: 评估 PPL
python benchmark/eval_ppl.py \
    --model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507 \
    --quant-weights /root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_v17_b32/qwen3-smooth-block-mixed-state_dict.pt \
    --label v9_b16x128_r10_qe_int4g32 --act-quant int4-g64

python benchmark/eval_ppl.py \
    --model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507 \
    --label v9_b128x128_r10_qe_int4g128 --act-quant none

# Step3: Zero-shot精度
python benchmark/eval_lm_eval.py \
  --model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507 \
  --quant-weights /root/autodl-tmp/Zip/qwen3_gptq_repro/output/smooth_v17_b32/qwen3-smooth-block-mixed-state_dict.pt \
  --datasets-offline \
  --dataset-cache-dir /root/autodl-tmp/datasets \
  --tasks arc_challenge,arc_easy,piqa,hellaswag,lambada_openai,winogrande \
  --batch-size auto \
  --act-quant int4-g64 \
  --label int4-g64-br64

scp -rP 36147 /root/autodl-tmp/model/Qwen3-8B root@connect.westd.seetacloud.com:/root/autodl-tmp/model
nLQhlktt6U6e


python benchmark/eval_lm_eval.py \
  --model-dir /root/autodl-tmp/model/Qwen3-4B-Instruct-2507 \
  --datasets-offline \
  --dataset-cache-dir /root/autodl-tmp/datasets \
  --tasks arc_challenge,arc_easy,piqa,hellaswag,lambada_openai,winogrande \
  --batch-size auto \
  --act-quant none \
  --label new_smooth

```

---

## 2. 方法简介

**Baseline GPTQ**: 标准 4-bit GPTQ 量化。可选 SmoothQuant 预处理将激活离群值迁移到权重中。

**V7 Tail Absorb**: 在 GPTQ 的 act-order 排序后，将最不重要的 `tail_rank` 列用 INT8 量化（而非 INT4），其余列保持 INT4。误差传播不变。

**V9 Submatrix Mixed Precision**: 将权重矩阵分割为子矩阵块，用敏感度评分选出 top-k% 的块使用 INT8，其余 INT4。相比 V7 的固定列位置，V9 自适应地将 INT8 预算放在量化误差最大的位置，PPL 更优。

---

## 3. 实验矩阵

| 实验标签 | 方法 | 关键参数 | 预期 PPL (Anone) |
|---|---|---|---|
| `fp16_baseline` | FP16 原始 | — | 10.0449 |
| `gptq_4bit_raw` | GPTQ 4-bit | — | 10.3845 |
| `smooth_gptq_4bit` | Smooth + GPTQ | — | 10.8361 |
| `smooth_ta_r16` | V7 Tail Absorb | rank=16 | 10.3428 |
| `smooth_ta_r64` | V7 Tail Absorb | rank=64 | 10.4042 |
| `smooth_ta_r128` | V7 Tail Absorb | rank=128 | 10.4184 |
| `v9_b128x128_r5_qe` | V9 Mixed | 128×128, 5%, quant_error | 10.4101 |
| `v9_b128x128_r10_qe` | V9 Mixed | 128×128, 10%, quant_error | 10.2813 |
| `v9_b64x128_r5_qe` | V9 Mixed | 64×128, 5%, quant_error | 10.4995 |
| `v9_b64x128_r10_qe` | V9 Mixed | 64×128, 10%, quant_error | **10.2786** |
| `v9_b128x128_r5_hw` | V9 Mixed | 128×128, 5%, hessian_weighted | 10.3958 |
| `v9_b128x128_r5_wn` | V9 Mixed | 128×128, 5%, weight_norm | 10.4412 |

---

## 4. 运行命令速查表

> 所有命令在 `qwen3_gptq_repro/` 目录下执行。`MODEL` 和 `SMOOTH_PT` 需替换为实际路径。

```bash
export MODEL="/path/to/Qwen3-4B-Instruct-2507"
export SMOOTH_PT="output/smooth/qwen3-4b-smooth-state_dict.pt"
```

### 4.1 基线

```bash
# FP16 baseline (无需量化，直接评估)
python benchmark/eval_ppl.py --model-dir $MODEL --label fp16_baseline_Anone --act-quant none

# GPTQ 4-bit raw
python qwen3_gptq.py --model-dir $MODEL --output-dir output/gptq_from_raw
python benchmark/eval_ppl.py --model-dir $MODEL \
    --quant-weights output/gptq_from_raw/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label gptq_4bit_raw_Anone --act-quant none

# SmoothQuant + GPTQ
python qwen3_smooth.py --model-dir $MODEL --output-dir output/smooth
python qwen3_gptq.py --model-dir $MODEL --init-state-dict $SMOOTH_PT \
    --output-dir output/gptq_from_smooth
python benchmark/eval_ppl.py --model-dir $MODEL \
    --quant-weights output/gptq_from_smooth/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label smooth_gptq_4bit_Anone --act-quant none
```

### 4.2 V7 Tail Absorb

> ⚠️ 下列命令包含全部 7 个铁律参数（见 Section 0）。请不要把它们整合到 `COMMON_ARGS`
> 变量里做"简化共用"——历史上 CLIP bug 就是这样被引入的。

```bash
for RANK in 16 64 128; do
    python qwen3_gptq_tail_absorb.py --model-dir $MODEL \
        --init-state-dict $SMOOTH_PT \
        --output-dir output/exp_smooth_tail_absorb/from_smooth_r${RANK} \
        --wbits 4 --nsamples 128 --seqlen 2048 \
        --groupsize 128 --percdamp 0.01 \
        --true-sequential --use-standard-quantizer \
        --tail-rank $RANK
    python benchmark/eval_ppl.py --model-dir $MODEL \
        --quant-weights output/exp_smooth_tail_absorb/from_smooth_r${RANK}/qwen3-4b-instruct-2507-gptq-4bit.pt \
        --label smooth_ta_r${RANK}_Anone --act-quant none
done
```

### 4.3 V9 Submatrix Mixed Precision

> ⚠️ 同样必须包含铁律参数。`--block-cols` 必须等于 `--groupsize`（都 =128），
> 否则会打印 `WARNING: block_cols != groupsize`。

```bash
# 单个实验（b128x128, r=10%, metric=quant_error）
python qwen3_gptq_submatrix_mixed.py --model-dir $MODEL \
    --init-state-dict $SMOOTH_PT \
    --output-dir output/exp_submatrix_mixed/b128x128_r10_qe \
    --nsamples 128 --seqlen 2048 \
    --block-rows 128 --block-cols 128 --groupsize 128 --percdamp 0.01 \
    --budget-ratio 0.10 --sensitivity-metric quant_error \
    --use-standard-quantizer

python benchmark/eval_ppl.py --model-dir $MODEL \
    --quant-weights output/exp_submatrix_mixed/b128x128_r10_qe/qwen3-4b-instruct-2507-gptq-4bit.pt \
    --label v9_b128x128_r10_qe_Anone --act-quant none \
    --output-dir output/benchmark_v9 --results-file results_v9_all.txt
```

---

## 5. 参数调优指南

| 参数 | 含义 | 推荐值 | 说明 |
|---|---|---|---|
| `--block-rows` | 子矩阵行尺寸 | 128 | 与 hidden_size 对齐，越大越粗粒度 |
| `--block-cols` | 子矩阵列尺寸 | 128 | **必须等于 groupsize** 以获最佳精度 |
| `--budget-ratio` | INT8 预算比例 | 0.05-0.10 | 10% 比 5% 提升 0.12-0.20 PPL |
| `--sensitivity-metric` | 敏感度评分方法 | `hessian_weighted` | 5% 预算下全面最优 |
| `--tail-rank` (V7) | INT8 列数 | 16 | 越小越好（V7 中 r16 最优） |
| `--percentile-k` | 量化 scale 百分位 | 75.0 | 截断 outlier，降低 scale 膨胀 |
| `--groupsize` | per-group 量化组大小 | 128 | 标准值，与 block-cols 对齐 |

---

## 6. 已有结果参考 (Qwen3-4B)

| Method | Anone | A8 | A4g128 | A4g128+down:A8 |
|---|---|---|---|---|
| fp16_baseline | 10.0449 | 10.3204 | 14.1556 | 13.4046 |
| gptq_4bit_raw | 10.3845 | 10.7033 | 15.3095 | 14.3323 |
| smooth_gptq_4bit | 10.8361 | 11.1372 | 14.1750 | 13.4645 |
| smooth_ta_r16 | 10.3428 | 10.6156 | 13.2699 | 12.6241 |
| smooth_ta_r64 | 10.4042 | 10.6748 | 13.3250 | 12.6773 |
| smooth_ta_r128 | 10.4184 | 10.6897 | 13.3127 | 12.6651 |
| v9_b128x128_r10_qe | 10.2813 | 10.5519 | **13.0820** | 12.5364 |
| v9_b64x128_r10_qe | **10.2786** | **10.5419** | 13.1487 | **12.5297** |

**全局最优**: v9_b64x128_r10_qe, Anone → PPL=10.2786 (距 FP16 仅 +0.23)

---

## 7. 扩展到更大模型

1. 重新运行 `qwen3_smooth.py`（smooth 权重是模型特定的）
2. 调整 `--model-dir` 指向新模型
3. 大模型可能需要降低 `--nsamples`（如 16）和 `--seqlen`（如 512）以节省显存
4. `--block-rows` 可适当增大（如 256），`--block-cols` 保持 128
5. 更新 `--output-dir` 避免覆盖已有结果

---

## 8. 批量实验脚本模板

> ⚠️ **关键约定**：本模板里所有量化命令都**显式**追加 7 个铁律参数（见 Section 0），
> 不使用 `COMMON_ARGS` 做缩写；请不要在后续维护时把这些参数抽到变量里。

```bash
#!/bin/bash
export MODEL="/path/to/model"
export SMOOTH_PT="output/smooth/qwen3-4b-smooth-state_dict.pt"
export MAX_PARALLEL=1  # GPU 数量限制，设为 1 串行执行

for BROW in 64 128; do
for BCOL in 128; do
for RATIO in 0.05 0.10; do
for METRIC in quant_error hessian_weighted weight_norm; do
    TAG="b${BROW}x${BCOL}_r$(echo $RATIO | sed 's/0\.//')_${METRIC:0:2}"
    OUTDIR="output/exp_submatrix_mixed/${TAG}"

    echo "=== Running: $TAG ==="
    # NOTE: 不要删除 --nsamples/--seqlen/--groupsize/--percdamp/--use-standard-quantizer
    #       这些参数漏掉任一项都会触发 CLIP bug（PPL 爆炸 17~33 倍）
    python qwen3_gptq_submatrix_mixed.py --model-dir $MODEL \
        --init-state-dict $SMOOTH_PT --output-dir $OUTDIR \
        --nsamples 128 --seqlen 2048 \
        --block-rows $BROW --block-cols $BCOL \
        --groupsize 128 --percdamp 0.01 \
        --budget-ratio $RATIO --sensitivity-metric $METRIC \
        --use-standard-quantizer

    for AQ in none int8 "int4-g128"; do
        AQ_LABEL=$(echo $AQ | sed 's/none/Anone/' | sed 's/int8/A8/' | sed 's/int4-g128/A4g128/')
        python benchmark/eval_ppl.py --model-dir $MODEL \
            --quant-weights ${OUTDIR}/qwen3-4b-instruct-2507-gptq-4bit.pt \
            --label v9_${TAG}_${AQ_LABEL} --act-quant $AQ \
            --output-dir output/benchmark_v9 --results-file results_v9_all.txt
    done
done
done
done
done
```

---

## 9. FAQ / Troubleshooting

**Q: WikiText-2 加载失败？**
A: 使用 `--local-wikitext2-dir` 指定本地数据集路径。脚本有 fallback 机制会生成临时文本，但不推荐用于正式实验。

**Q: CUDA OOM？**
A: 减小 `--nsamples`（如 16）或 `--seqlen`（如 512）。GPTQ 逐层处理，峰值显存 ≈ 2× 单层大小 + Hessian。

**Q: JSON 中的路径是服务器绝对路径？**
A: 是的，`quant_weights` 字段记录的是生成时的服务器路径。评估时用 `--quant-weights` 指定本地实际路径即可。

**Q: block_cols 必须等于 groupsize 吗？**
A: 强烈推荐。不对齐时脚本会打印 WARNING，量化精度可能下降。

**Q: 如何只跑 FP16 baseline？**
A: `python benchmark/eval_ppl.py --model-dir $MODEL --label fp16_baseline_Anone --act-quant none`（不指定 `--quant-weights`）。

**Q: V7/V9 量化后 PPL 突然从 10.x 飙到 180+ 了？**
A: 命令里很可能漏了 Section 0 里的铁律参数之一。依次检查：
  1. `grep -c "clip_ratio=0.2500" <量化日志>`，如果 >0 → 漏了 `--use-standard-quantizer`。
  2. 确认 `--nsamples 128 --seqlen 2048` 都显式出现，没被默认覆盖。
  3. V7 必须带 `--true-sequential`，V9 要显式 `--groupsize 128`。
  完整复盘见 [`POSTMORTEM_V7_V9_CLIP_BUG.md`](./POSTMORTEM_V7_V9_CLIP_BUG.md)。
