# L24：AWQ-lite W8 Per-Channel 量化 Serving

## 0. 本讲目标

- 理解 weight-only quantization 在 serving 中解决的显存和带宽问题。
- 能区分 AWQ input-channel scale、per-output dequant scale、FP8 scale 和 KV cache scale。
- 能实现 AWQ-lite W8 per-channel quant/dequant，并解释每个 shape。
- 能判断量化报告里的显存、速度、accuracy drift 和硬件边界是否完整。
- 能读懂 MiniInfra 和 vLLM 源码中 AWQ、FP8、KV cache quantization 的主要检查点。

## 1. 问题背景：量化 Serving 首先是资源和误差问题

推理服务的显存通常被三类对象占用：

- 模型权重。
- KV cache。
- 临时激活、workspace、通信或 kernel buffer。

当模型权重过大时，服务能留给 KV cache 的显存会减少，长上下文和高并发更容易被挤掉。weight-only quantization 通过低 bit 存储权重，降低权重显存和权重读取带宽。W8A16 表示 weight 用 8 bit 存储或加载，activation 仍用 16 bit 参与计算；W4A16 更省显存，但对 scale 和 kernel 更敏感。

量化的代价是误差。把 fp16/fp32 权重映射到 int8/int4 时，必须选择 scale。scale 太大，小权重会被挤到少数整数刻度；scale 太小，outlier 会被截断。AWQ 的动机来自这里：activation 中经常出现大值的输入通道，对输出更敏感，需要在量化前得到额外保护。

本讲用 W8 而不是 W4，是为了让学生在 CPU 上直接检查数学、shape 和误差门槛。真实 AWQ 常见于 W4A16、per-group scale、packed int4 和专用 GEMM kernel；这些生产复杂度会在源码带读中对照。

## 2. 核心概念

### 2.1 W8A16 / W4A16 / FP8 / KV-int8

**W8A16：**
weight 用 8 bit 表示，activation 保持 fp16/bf16。它主要减少权重存储和读取，速度收益取决于 kernel 是否能避免频繁 dequant 或使用低精度 GEMM。

**W4A16：**
weight 用 4 bit 表示，activation 保持 fp16/bf16。显存更省，但 scale、group_size、pack 格式和 kernel 支持更关键。真实 AWQ/GPTQ 常落在这个区域。

**FP8：**
weight 和/或 activation 使用 8 bit 浮点格式，常见 e4m3 或 e5m2。FP8 的动态范围和硬件路径不同于 INT8，通常要记录 weight scale、activation scale 或 block scale。

**KV-int8：**
量化 KV cache，而不是权重。它降低长上下文和高并发下的 KV cache 显存，但需要保存 scale metadata，并验证 decode 累积误差。

### 2.2 Per-tensor、per-channel、per-group

量化公式可以写成：

```text
int_x = round(x / scale).clamp(qmin, qmax)
x_hat = int_x * scale
```

scale 的粒度决定精度和元数据成本：

- per-tensor：整个 tensor 一个 scale，最简单，但容易被 outlier 主导。
- per-channel：每个通道一个 scale，本讲的 int8 dequant scale 按输出通道记录。
- per-group：每个 group 一个 scale，真实 AWQ/GPTQ 常用，通常以 group_size=128 这类参数出现。

本讲有两个不同的 scale：

- `awq_scale: (in_features,)`，由 activation amax 和每个输入通道的 weight amax 算出。
- `per_out_scale: (out_features,)`，由 `weight * awq_scale` 的每个输出通道最大值算出。

### 2.3 AWQ-lite

**定义：**
AWQ-lite 使用 activation amax 指导输入通道缩放。某个输入通道的 activation amax 越大，该通道更可能影响输出误差，因此在量化前把对应 weight 列乘上更大的 scale。

**输入：**
`weight: (out_features, in_features)`，`act_amax: (in_features,)`，`alpha`。

**中间状态：**
`w_amax_in = weight.abs().max(dim=0).values`，shape 是 `(in_features,)`。随后计算：

```text
awq_scale = clamp(act_amax^alpha / w_amax_in^alpha, min=1.0)
```

**输出：**
`awq_scale`，shape 是 `(in_features,)`。

**边界：**
`alpha=0` 时所有 scale 都是 1，退化为普通 weight-only INT8 quantization。课堂 patch 只允许 scale 下限为 1.0，目标是保护 activation outlier，不主动压小其它通道。

## 3. 机制链路：三步函数

### 3.1 `compute_awq_scale`

第一步从 calibration 统计进入。activation 沿输入通道流入 linear，所以 `act_amax` 和 `w_amax_in` 都按输入通道排列。

```python
w_amax_in = weight.abs().max(dim=0).values
scale = act_amax.clamp(min=eps).pow(alpha) / w_amax_in.clamp(min=eps).pow(alpha)
scale = scale.clamp(min=1.0)
```

如果沿 `dim=1` 求最大值，会得到输出通道统计，shape 变成 `(out_features,)`，后续 `weight * scale.unsqueeze(0)` 会错误或广播错位。

### 3.2 `quantize_w8_per_channel`

第二步对被 AWQ scale 放大的权重做 per-output-channel INT8 量化：

```python
w_scaled = weight * awq_scale.unsqueeze(0)
per_out_amax = w_scaled.abs().max(dim=1).values
per_out_scale = (per_out_amax / 127.0).clamp(min=1e-8)
int_w = (w_scaled / per_out_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
```

这里用 127 作为对称量化分母，测试也要求 int8 值在 `[-127, 127]`。`per_out_scale` 记录每个输出通道如何从整数还原到 `weight * awq_scale` 空间。

### 3.3 `dequantize_w8_per_channel`

第三步先恢复 `weight * awq_scale`，再按输入通道除回：

```python
w = int_weight.float() * per_out_scale.unsqueeze(1)
if awq_scale is not None:
    w = w / awq_scale.unsqueeze(0)
```

如果忘记除回 `awq_scale`，输出权重仍然带着输入通道放大，linear 的 logits 会偏移。测试中的 roundtrip 相对误差会捕捉这类错误。

## 4. 测试为什么这样设计

L24 patch 的测试覆盖五个边界：

- int8 dtype 和 `[-127,127]` 范围。
- quant/dequant 后的平均相对误差。
- AWQ scale shape 必须是 `(in_features,)`。
- activation outlier 通道的 scale 均值要高于非 outlier 通道。
- `alpha=0` 必须返回全 1，保证普通 INT8 退化路径。

这些测试证明数学合同和 shape 合同。它们不证明真实 AWQ W4 checkpoint 能加载，也不证明 GPU 端到端加速。生产结论要继续看 kernel、pack 格式、评估集和硬件。

## 5. MiniInfra：从 patch 回到 serving 指标

MiniInfra 提供三个对照：

1. `calibrator.py` 用样本求 `channel_absmax`、SmoothQuant scale 和 AWQ activation order。它说明 scale 来自 calibration samples，而不是随手设常数。
2. `awq_loader.py` 用 group layout 和字节估算解释真实 AWQ W4A16 checkpoint 中 group_size、scale、zero point 和 compression ratio。
3. `run_engine.py` 的 `quant_payload()` 给出 fp16 baseline、AWQ、FP8、KV-int8 的指标行，包含 TTFT、ITL、tokens/s、peak KV memory、accuracy drop 和 cache hit rate。

这些数字是 validation-only 的教学矩阵。本地 CPU smoke 可以检查字段和 artifact，但不能写成真实 GPU 加速结论。

## 6. 真实 vLLM 源码边界

### 6.1 AWQ

vLLM 的 `AWQConfig` 只接受 4-bit weight quantization，并从 config 中读取 `weight_bits`、`group_size`、`zero_point`。`AWQLinearMethod.create_weights()` 会检查：

- input size 是否能被 group_size 整除。
- output size 是否能被 pack_factor 整除。
- `qweight`、`qzeros`、`scales` 是否按 packed layout 注册。

apply 阶段可能先 dequantize 再 matmul，也可能调用 AWQ GEMM kernel。加载失败时，排查点通常落在 group_size、pack_factor、checkpoint 字段和 tensor parallel 切分。

### 6.2 FP8

`Fp8Config` 关心 checkpoint 是否已经 FP8 serialized、activation_scheme、ignored_layers 和 weight_block_size。FP8 linear method 会创建 FP8 weight、weight_scale、block scale 或 input_scale，并根据动态/静态 activation scheme 选择不同 kernel key。

FP8 的重点不是“也是 8 bit”这么简单。它涉及浮点格式、scale 记录、硬件能力和 activation scheme。H100/H200 上的 FP8 结论不能直接外推到没有原生 FP8 路径的 GPU。

### 6.3 KV cache scale

KV cache quantization 会给 Attention layer 增加 q/k/v/prob scale。真实加载器要判断 checkpoint 是否提供 k/v scale、是否需要动态计算 scale、是否是 per-token-head scale。长上下文和 prefix cache 场景里，KV scale 的错误会在 decode 过程中累积。

## 7. 量化报告怎么写

一次合格的量化 serving 报告至少包含：

| 类别 | 字段 |
|---|---|
| 配置 | quant method、weight bits、activation dtype、group_size、zero_point、calibration set |
| 硬件 | GPU、CUDA、kernel/backend、batch、context length、concurrency |
| 速度 | TTFT、ITL、tokens/s、吞吐、延迟分位数 |
| 显存 | weight memory、KV memory、peak memory |
| 质量 | acc_drop_pp、perplexity、任务指标、长上下文漂移 |
| 证据 | command、resolved config、metrics artifact、git commit |

只报告 tokens/s 容易误导。量化方案可能提升吞吐，同时带来 accuracy drop；也可能降低权重显存，却因没有合适 kernel 而没有速度收益。

## 8. 排查路径

遇到量化 serving 问题时，按顺序查：

1. 固定现场：模型、量化方法、checkpoint 字段、校准数据、GPU、dtype、batch、context。
2. 固定数学：用 patch 的小 tensor 复查 scale shape、int8 范围和 roundtrip error。
3. 固定加载：检查 group_size、pack_factor、qweight/qzeros/scales、weight_scale/input_scale。
4. 固定指标：同时看 TTFT、ITL、tokens/s、peak_kv_mem_gb、acc_drop_pp。
5. 固定质量：用同一评估集比较 fp16 baseline 和量化模型。
6. 固定外推边界：CPU smoke、validation-only 矩阵和真实 GPU benchmark 分开写。

## 9. 小结

L24 的知识链路是：serving 资源压力推动 weight/KV 量化；scale 粒度决定误差和元数据；AWQ-lite 用 activation amax 保护输入通道；W8 per-output-channel quant/dequant 给出可断言的数学合同；MiniInfra 和 vLLM 源码把这个合同放回 checkpoint、kernel、KV scale 和指标报告。掌握这条链后，学生才能判断一个量化服务是只“能跑”，还是已经有足够证据上线。
