# 源码带读：L24 AWQ-lite W8 Per-Channel 量化 Serving

这份带读按主路径组织：先读 patch 和测试，确认数学合同；再读 MiniInfra 的 calibration、AWQ layout、KV-int8 和 smoke 指标；最后读真实 vLLM 的 AWQ、FP8、KV cache scale 处理。

## 0. 源码地图

```text
labs/l23_quant_serving/patch/starter/awq_calibrate.py
labs/l23_quant_serving/patch/reference/awq_calibrate.py
labs/l23_quant_serving/patch/tests/test_patch.py
mini_infra/vllm/quant/calibrator.py
mini_infra/vllm/quant/awq_loader.py
mini_infra/sglang/quant/kv_int8.py
mini_infra/vllm/run_engine.py
github_repo/vllm/vllm/model_executor/layers/quantization/awq.py
github_repo/vllm/vllm/model_executor/layers/quantization/fp8.py
github_repo/vllm/vllm/model_executor/layers/quantization/kv_cache.py
```

## 1. Patch starter：确认三个函数的输入和输出

文件：`labs/l23_quant_serving/patch/starter/awq_calibrate.py`

重点看：

- L18-L27：`compute_awq_scale()` 的输入 shape 和返回 shape。
- L28-L34：AWQ scale 的 TODO 公式。
- L37-L50：W8 per-channel quantization 的参数和返回值。
- L51-L58：`w_scaled`、`per_out_scale`、int8 clamp 的步骤。
- L61-L72：dequant 时先乘 per-output scale，再除回 AWQ scale。

读完要得到的结论：

AWQ scale 是输入通道 scale，per-output scale 是 int8 权重的反量化 scale。两个 scale 不能互换。

可以先跳过：

- 顶部填空规则。
- 具体错误文本。

## 2. Patch reference：看最小实现如何闭合

文件：`labs/l23_quant_serving/patch/reference/awq_calibrate.py`

重点看：

- L8-L17：沿 `dim=0` 求 `w_amax_in`，返回 `(in_features,)` 的 AWQ scale。
- L20-L30：对 `weight * scale` 做 per-output-channel 对称 INT8 量化。
- L33-L41：dequant 后按输入通道除回 AWQ scale。

读完要得到的结论：

reference 的 shape 方向和广播位置是本讲最重要的代码细节。`scale.unsqueeze(0)` 和 `per_out_scale.unsqueeze(1)` 分别对应输入通道和输出通道。

可以先跳过：

- import 和类型注解。

## 3. Patch tests：看验收边界

文件：`labs/l23_quant_serving/patch/tests/test_patch.py`

重点看：

- L17-L19：`IMPL` 选择 starter 或 reference。
- L22-L31：int8 dtype 和 `[-127,127]` 范围。
- L34-L43：roundtrip 相对误差。
- L46-L51：AWQ scale shape 必须是 `(128,)`。
- L54-L65：构造 activation outlier。
- L67-L84：outlier scale 响应和 AWQ roundtrip 误差。
- L87-L95：`alpha=0` 返回全 1。

读完要得到的结论：

测试证明的是最小数学合同。它没有覆盖真实 W4 packed checkpoint、专用 kernel 或任务准确率。

可以先跳过：

- pytest 的导入和 skip 机制。

## 4. MiniInfra calibration：看 scale 从哪里来

文件：`mini_infra/vllm/quant/calibrator.py`

重点看：

- L25-L34：按 channel 求 calibration samples 的绝对最大值。
- L37-L39：SmoothQuant scale 使用 channel amax 和 alpha。
- L42-L44：AWQ activation order 按激活幅度排序。
- L47-L60：summary 把样本数、scale、order 和校准说明打包。

读完要得到的结论：

AWQ 和 SmoothQuant 都依赖 calibration samples。没有真实激活统计，scale 选择只能是启发式。

可以先跳过：

- 顶部历史课程编号。

## 5. MiniInfra AWQ layout：看真实 W4A16 多出的存储约束

文件：`mini_infra/vllm/quant/awq_loader.py`

重点看：

- L43-L57：`group_layout()` 检查 hidden_size 是否能被 group_size 整除，并生成 group metadata。
- L60-L77：`awq_summary()` 估算 fp16 权重、packed int4 权重、scale 开销和压缩比。

读完要得到的结论：

真实 AWQ 不只是一个 W8 roundtrip。W4A16 checkpoint 还要处理 group_size、zero point、packed int4 和 scale metadata。

可以先跳过：

- dataclass 的 `to_dict()`。

## 6. MiniInfra KV-int8：看 KV cache scale metadata

文件：`mini_infra/sglang/quant/kv_int8.py`

重点看：

- L26-L33：对称 int8 scale 使用 `max(abs(x)) / 127`。
- L36-L42：quantize 和 dequantize 都必须保存 scale。
- L45-L62：summary 比较 fp16 KV 和 int8 KV 的显存，并说明 prefix cache key 与 stored block 的边界。

读完要得到的结论：

KV-int8 降低的是 KV cache 显存，不是权重显存。cache key 仍按 token/prefix 管理，但缓存值需要 scale metadata 才能还原。

可以先跳过：

- 顶部教学说明。

## 7. run_engine：看 serving 指标矩阵

文件：`mini_infra/vllm/run_engine.py`

重点看：

- L13-L26：AWQ 指标行包含 TTFT、ITL、tokens/s、peak KV memory、acc_drop 和 cache hit。
- L27-L40：FP8 指标行额外记录硬件边界。
- L41-L53：KV-int8 指标行使用 `kv_int8_summary()` 的内存结果。
- L54-L64：fp16 baseline 是所有量化行的比较对象。
- L79-L87：量化请求会同时写入 quant payload 和 calibration summary。
- L91-L99：run id 存在时写入 `quant_matrix.json` 和 `accuracy_drift.json`。

读完要得到的结论：

量化报告不能只写 tokens/s。至少要同时保存延迟、吞吐、显存、准确率漂移和硬件边界。

可以先跳过：

- spec decode 分支。

## 8. vLLM AWQ：看真实 checkpoint 加载边界

文件：`github_repo/vllm/vllm/model_executor/layers/quantization/awq.py`

重点看：

- L40-L58：`AWQConfig` 保存 weight_bits、group_size、zero_point，并限制当前 AWQ 只支持 4-bit weight。
- L87-L95：从 quant config 读取 bits、group_size、zero_point。
- L192-L204：input size 必须被 group_size 整除。
- L205-L213：output size 必须被 pack_factor 整除。
- L214-L255：注册 `qweight`、`qzeros`、`scales`。
- L268-L286：apply 阶段在 dequantize+matmul 和 AWQ GEMM 之间选择。

读完要得到的结论：

真实 AWQ 加载器消费的是已量化 checkpoint。排查重点是字段、shape、group/pack 对齐和 kernel 路径。

可以先跳过：

- MoE fallback 和 mapper 细节。

## 9. vLLM FP8 与 KV cache：看 activation scheme 和 scale 检查

文件：`github_repo/vllm/vllm/model_executor/layers/quantization/fp8.py`

重点看：

- L96-L112：`Fp8Config` 记录 checkpoint 是否 FP8 serialized、activation_scheme、ignored layers 和 block size。
- L155-L170：从 config 中解析 FP8 参数。
- L312-L356：创建 FP8 weight 和 weight_scale。
- L371-L384：静态 activation 时注册 input_scale，并初始化 FP8 kernel。
- L397-L429：加载后处理 weight、weight_scale 和 input_scale。

文件：`github_repo/vllm/vllm/model_executor/layers/quantization/kv_cache.py`

重点看：

- L32-L45：Attention layer 初始化 q/k/v/prob scale。
- L57-L69：per-token-head KV scale 路径会忽略 checkpoint scale。
- L75-L127：量化 KV cache 使用 checkpoint k/v scale 或默认值。
- L154-L167：q_scale/prob_scale 写入 Attention，并对未校准 FP8 scale 给出 warning。

读完要得到的结论：

FP8 和 KV cache quantization 的核心不只是 dtype。activation scheme、block scale、checkpoint scale、动态 scale 和硬件 backend 都会影响可用性和准确率。

可以先跳过：

- MoE FP8 分支和深层 kernel 选择。

## 读完后的自检问题

1. `compute_awq_scale()` 为什么沿 `dim=0` 求 weight amax？
2. `scale.unsqueeze(0)` 和 `per_out_scale.unsqueeze(1)` 分别对应哪一个通道方向？
3. MiniInfra 的 `quant_payload()` 为什么同时记录 `acc_drop_pp` 和 `peak_kv_mem_gb`？
4. vLLM AWQ 加载失败时，group_size 和 pack_factor 分别约束哪一维？
5. FP8/KV cache 量化为什么必须保存 scale metadata？
