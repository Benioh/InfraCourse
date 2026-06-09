# Debug Checklist：L24 量化 Serving

## 1. 先固定现场

- 记录命令、配置文件、git commit、Python/PyTorch/CUDA 版本、GPU 型号和驱动。
- 记录 quant method、weight bits、activation dtype、group_size、zero_point、calibration set、eval set。
- 记录模型大小、batch、context length、concurrency、prompt 分布和随机种子。
- 保存 stdout/stderr、`quant_matrix.json`、`accuracy_drift.json`、resolved config、command snapshot 和评估结果。

## 2. 先判断是哪一类问题

| 问题 | 先看什么 | 常见原因 |
|---|---|---|
| patch 失败 | scale shape、广播方向、int8 范围、roundtrip error | 把 AWQ scale 写成输出通道，忘记除回 awq_scale |
| AWQ load 失败 | qweight/qzeros/scales、group_size、pack_factor | hidden 或 output 维度不对齐，checkpoint 字段缺失 |
| FP8 结果差 | activation_scheme、weight_scale、input_scale、校准集 | scale 未加载，动态/静态 activation 配置错 |
| KV-int8 漂移 | k/v scale、长上下文、prefix cache、decode 步数 | scale metadata 丢失，短 prompt 未覆盖累积误差 |
| 速度没有提升 | kernel/backend、batch、dtype、硬件能力 | 只有存储压缩，没有低精度 GEMM 或 dequant 成本过高 |

## 3. Patch 数学排查

1. 检查 `weight` shape 是 `(out_features, in_features)`。
2. `compute_awq_scale()` 必须沿 `dim=0` 得到 `(in_features,)`。
3. `alpha=0` 时 scale 必须是全 1。
4. `quantize_w8_per_channel()` 中 `weight * awq_scale.unsqueeze(0)` 不能写反。
5. `per_out_scale` 必须沿 `dim=1` 求最大值，shape 是 `(out_features,)`。
6. dequant 后如果传入 `awq_scale`，必须除以 `awq_scale.unsqueeze(0)`。

## 4. Serving 指标排查

1. 用 fp16 baseline 作为比较对象。
2. 同时记录 TTFT、ITL、tokens/s、peak_kv_mem_gb、acc_drop_pp 和 cache_hit_rate。
3. 速度结论要绑定 GPU、kernel/backend、batch、context length 和 concurrency。
4. 准确率结论要绑定 eval set、calibration set 和 metric。
5. CPU smoke 只证明字段和 artifact，不证明低精度 kernel 加速。

## 5. 真实加载器排查

1. AWQ：检查 `weight_bits == 4`、`group_size`、`zero_point`、`qweight`、`qzeros`、`scales`。
2. AWQ：检查 input size 是否能被 group_size 整除，output size 是否能被 pack_factor 整除。
3. FP8：检查 checkpoint 是否 serialized FP8，activation_scheme 是否匹配。
4. FP8：检查 weight_scale、input_scale 或 block scale 是否存在。
5. KV cache：检查 q/k/v/prob scale，确认是否动态计算 per-token-head scale。

## 6. 结束条件

- 问题能被一个最小命令复现。
- 数学合同、加载合同和 serving 指标已经分层。
- artifact 足以恢复模型、quant config、workload、硬件和指标。
- 结论写入 `serving_metrics_template.md`，并标明不能外推的条件。
