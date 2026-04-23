# 事故复盘 — V7 / V9 的 CLIP 爆炸 Bug

> **类别**：量化配置缺省 → PPL 爆炸 17~33 倍
> **涉及方法**：V7 (GPTQ Tail Absorb)、V9 (GPTQ Submatrix Mixed)
> **影响范围**：基于 `smoothed_model_state_dict.pt` 的所有 smooth-based 高精度补偿实验
> **现象**：`smooth_ta_r16_Anone` PPL 从历史基线 10.34 飙到 345.66 / 179.68
> **根因**：量化脚本的命令行缺少关键参数，退化到脚本默认值路径（`PercentileQuantizer(k=75)` + `nsamples=32` + `seqlen=1024`）
> **最终结论**：不是算法坏了，而是 **命令行参数三重缺省** 叠加导致量化器行为偏离基线

> ⚠️ **历史遗留说明（2026-04-23）**
>
> `PercentileQuantizer` 是 V7/V9 早期开发阶段引入的实验性 scale 估计器，后续被证明在
> Qwen3-4B 上会造成大权重截断（clip_ratio=0.2500 恒定）。
> 该 Quantizer **将在后续提交中从代码库移除**。在它被移除前，本文档描述的防御参数
> （特别是 `--use-standard-quantizer`）对所有 V7/V9 量化命令都是**必须**的。
>
> 移除 `PercentileQuantizer` 之后，本文档的 Bug A 部分即自动失效，但 Bug B / Bug C
> 以及"铁律参数"思路仍然有效，请继续保留本文档作为流程防御的历史记录。

---

## 1. 症状

| 实验 | 预期 PPL | 实际 PPL | 比值 |
|------|:--------:|:--------:|:----:|
| `smooth_ta_r16_Anone` | 10.34  | **345.66 → 179.68** | ×33 → ×17 |
| `smooth_ta_r16_A8`    | 10.62  | 279.06              | ×26 |
| `smooth_ta_r16_A4g128`| 13.27  | 233.27              | ×17 |
| `smooth_ta_r64_*`     | 同级别    | 同级别爆炸            | —    |
| `v9_*`                | 同级别    | 同级别爆炸            | —    |

量化日志里 36 层 × 7 Linear 一共 252 次 `find_params`，**全部**打印：

```
clip_ratio=0.2500
```

同时各层的 `gptq_loss` 分布形如：

```
Layer 0   gptq_loss ~ 100 ~ 1000          (正常)
Layer 1   gptq_loss ~ 2.3e4 ~ 2.9e4       (开始偏大)
Layer 10  gptq_loss ~ 1e6 ~ 1e7           (爆炸)
Layer 35  gptq_loss ~ 1.7e6               (持续爆炸)
```

深层 loss 呈指数级发散 → 典型的 "early-layer 量化 scale 错 → GPTQ Hessian 补偿放大误差 →
后续层见到被污染的 W" 累积传播模式。

---

## 2. 三个叠加的 Bug（缺一就不会这么严重）

### Bug A — `PercentileQuantizer` 默认启用（scale 截断 25% 权重）

> **注**：这是历史遗留 bug，`PercentileQuantizer` 将被删除。本节保留做档案。

| | 基线 `qwen3_gptq.py` | V7 `qwen3_gptq_tail_absorb.py` | V9 `qwen3_gptq_submatrix_mixed.py` |
|---|---|---|---|
| 默认 Quantizer | `Quantizer()`（min/max） | `PercentileQuantizer(k=75)` | `PercentileQuantizer(k=75)` |
| 切换开关 | —（没有 Percentile 路径） | `--use-standard-quantizer` | `--use-standard-quantizer` |

`PercentileQuantizer.find_params` 的 `sym=False` 分支里硬写了：

```python
if self.sym:
    xmin = -xmax
else:
    xmin = -xmax   # 简化处理，对称截断
```

这意味着无论用户传 `--sym` 与否，**实际都做对称截断**，并且使用绝对值第 75 百分位作为 xmax，
**绝对值最大的 25% 权重直接被钉到 `[-xmax, xmax]` 边界** → `clip_ratio=0.25` 在所有层恒定。

每层前 25% 的大权重信息全丢了，而偏偏这些权重是 outlier，对下游激活的贡献远超平均权重。

### Bug B — `nsamples=32, seqlen=1024`（Hessian 严重欠估计）

| 参数 | 基线默认 | V7 / V9 默认 | 后果 |
|------|:--:|:--:|---|
| `--nsamples` | **128** | **32** | 样本 ×4 缩水 |
| `--seqlen`   | **2048** | **1024** | 长度 ×2 缩水 |

GPTQ 的权重 Hessian 估计：`H = (2/N) · Σ xxᵀ`。有效 rank 正比于 `nsamples × seqlen`：

- 基线：`128 × 2048 = 262 144 tokens` 估计一个最多 `9728×9728`（`up_proj`）的 Hessian
- 本次：`32 × 1024 = 32 768 tokens`，**样本量 ×1/8**

当 `nsamples × seqlen` 远小于 `d_in` 时，H 极度 rank-deficient；Cholesky 分解后 `Hinv`
对应的零特征方向数值爆炸 → GPTQ 的误差传播公式 `W[:, i:] -= err · Hinv[i, i:]` 把噪声
放大到后续列 → 后续 `find_params` 看到的是被污染的 W → scale 越估越错 → 循环强化。

### Bug C — `--true-sequential` 默认关闭

V7 脚本的 `--true-sequential` 决定了层内不同投影（q/k/v, gate/up, down, o）是否按基线
GPTQ 的原生顺序 sequential 地传播残差。`run_all.sh` 早期版本 `COMMON_GPTQ_ARGS` 没带这个
flag，相当于**跳过了顺序补偿**，与"V7 = 基线 GPTQ + tail 吸收"的语义不一致。

### 三者叠加的破坏力

| 单独影响 | 叠加后的症状 |
|---|---|
| A 单独：PPL 升到 ~12~15（大权重被截断） | clip_ratio=0.25 恒定 + PPL×17+ |
| B 单独：PPL 升到 ~14~18（Hessian 退化） | 层间 loss 发散、深层 1e7+ |
| C 单独：PPL 升到 ~12~13（顺序被破坏） | 叠加 A/B 后放大 3~5 倍 |

---

## 3. 修复动作（按应用顺序）

| 步骤 | 改动 | 位置 |
|---|---|---|
| 1 | `COMMON_GPTQ_ARGS` 补 `--nsamples 128 --seqlen 2048` | 服务器 `scripts/rerun_v7_v9/run_all.sh` |
| 2 | V7 每条 `quant_v7_ta_*` 子任务追加 `--groupsize 128 --percdamp 0.01 --true-sequential --use-standard-quantizer` | 同上 |
| 3 | V9 每条 `quant_v9_*` 子任务追加 `--use-standard-quantizer` | 同上 |
| 4 | 删除所有已被污染的 `.pt` 与 `ppl_*.json`，重跑 | `output/v7_*` + `output/benchmark/` |
| 5 | 小规模验证（仅 `v7_r16` 全流程）→ PPL 回到 10.34 即验证修复完成 | — |
| 6 | 放开跑其余 4 组 | — |

**验证结果**：`v7_r16` 小实验的 4 组 PPL 全部回到历史基线区间，差异 < 0.02，flag="OK"。

---

## 4. 为什么会踩这 3 个坑（Process Post-mortem）

1. **脚本级默认值分裂**：基线、V7、V9 三个量化脚本各自维护了不同的 `--nsamples / --seqlen`
   默认值，而没有任何一个统一的 "QUANT_DEFAULTS" 表。
2. **开关默认语义不直观**：`PercentileQuantizer` 作为 V7/V9 的默认量化器，而这个选择在
   文档和 CLI `--help` 里都没醒目提示；`--use-standard-quantizer` 的存在是隐式开关，不是必选。
3. **旧 runbook 命令集合不完整**：最早的 AI 指令文档里其实**正确地**显式写了所有 7 个铁律
   参数，但后来整合到 `run_all.sh` 时，为了"共用 COMMON_GPTQ_ARGS"做减法，悄悄丢了
   `--true-sequential` 和 `--use-standard-quantizer`。
4. **首轮观测太短**：只跑了第一组 `Anone` 就发现 PPL=345，花了 2 分钟怀疑是 eval 脚本问题，
   没第一时间去看量化日志里的 `clip_ratio`。如果最开始 grep 一下 `clip_ratio=0.2500`
   的出现次数，能在 10 秒内定位到 Bug A。

---

## 5. 今后避免复发的固化做法

1. **双文档事实源**：
   - 人类版 [`EXPERIMENT_GUIDE.md`](./EXPERIMENT_GUIDE.md) — 包含醒目的 CLIP-bug 警示框与"铁律参数"表。
   - 机器版 [`AI_TECHNICAL_REFERENCE.md`](./AI_TECHNICAL_REFERENCE.md) — 包含 **Mandatory Quantization Arguments** 章节与字面量精确命令。
   AI 跑实验时只认机器版，每条量化命令把 7 个铁律参数逐一列全，不做任何 `COMMON_ARGS` 省略。
2. **pre-flight check 脚本**：在量化启动前用 `grep` 验证脚本是否支持 `--use-standard-quantizer /
   --true-sequential / --tail-rank / --block-rows / --sensitivity-metric` 等 flag，
   eval 脚本是否支持 `--results-file`，不满足立即失败退出。
3. **post-quant health check**：量化结束后自动 `grep -c "clip_ratio=0.2500"` 量化日志，
   命中就告警"CLIP bug 复发"；同时检查每组 `*_Anone` PPL 是否 ≤ 15。
4. **脚本内注释警告**：`run_all.sh` 的 `COMMON_GPTQ_ARGS` 与每个 V7 量化子任务都加
   "本行删改会触发 CLIP bug"的注释。
5. **`PercentileQuantizer` 将被删除**：一旦该类从代码库移除，对应的 `--use-standard-quantizer`
   开关会成为唯一行为（或开关也同步移除），Bug A 的复发可能性从此消失。在那之前，
   本文档与两份主文档的铁律一致：**永远显式传 `--use-standard-quantizer`**。

---

## 6. 一眼速查：是否又中招？

在 autodl2 上执行：

```bash
# 1) 量化日志 clip_ratio 计数（应为 0）
for log in /root/autodl-tmp/Zip/qwen3_gptq_repro/scripts/rerun_v7_v9/logs/quant_v*.log; do
    n=$(grep -c "clip_ratio=0.2500" "$log" 2>/dev/null || echo 0)
    echo "$log : clip_ratio=0.2500 x $n"
done

# 2) 最新一条 eval 的 PPL 是否在 10.x / 13.x 区间
tail -20 /root/autodl-tmp/Zip/qwen3_gptq_repro/output/benchmark/results_v7_v9_rerun.txt
```

- 任意一条量化日志 `x N` 且 `N>0`：**Bug A 复发**，检查对应量化命令是否漏了
  `--use-standard-quantizer`（仅当 `PercentileQuantizer` 尚未删除时适用）。
- 任意一条 `*_Anone` PPL > 15：**Bug A 或 B 复发**，检查 `--nsamples/--seqlen`。
- `*_Anone` PPL 漂移 0.1 左右：一般是随机种子/calibration 采样差异，容忍范围。
