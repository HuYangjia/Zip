# Smooth 实现审查结论（Qwen3）

## 结论

当前 `qwen3_smooth.py` 的核心思路正确，符合 `alpha=1` 的 SmoothQuant 目标：
- 使用校准样本统计 RMSNorm 输出的每通道激活最大值；
- 对应执行 `norm.weight /= s` 与下游线性层 `weight *= s`；
- 保存 smooth 后权重、scale、metadata；
- 可被 `qwen3_gptq.py --init-state-dict` 正常衔接。

## 关键实现点

1. 激活统计：对 `input_layernorm` 和 `post_attention_layernorm` 注册 hook，收集 per-channel `amax`。
2. 平滑变换：
   - attention 路径：`input_layernorm -> (q_proj, k_proj, v_proj)`
   - MLP 路径：`post_attention_layernorm -> (gate_proj, up_proj)`
3. 产物：
   - `smoothed_model_state_dict.pt`
   - `smooth_scales.pt`
   - `smooth_metadata.json`

## Smooth后、GPTQ前的推理命令

`infer_quantized_qwen3.py` 虽然参数名是 `--quantized-weights`，但本质上是加载一个 `state_dict`，所以可以直接用 smooth 后权重做推理检查：

```bash
cd /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro
python infer_quantized_qwen3.py \
  --model-dir /home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507 \
  --quantized-weights /home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/smooth/smoothed_model_state_dict.pt \
  --prompt "请简要介绍一下你自己" \
  --max-new-tokens 256
```

## 风险与改进建议

1. 增加 smooth 前后 logits 等价性检查（建议作为必须项）。
2. 更新权重时可临时用 fp32，减少 fp16 数值误差。
3. 若精度波动大，尝试 percentile 统计替代纯 `amax`。

## 总评

实现基本正确，可用于实验主流程；建议补齐等价性自检后再作为稳定基线。
