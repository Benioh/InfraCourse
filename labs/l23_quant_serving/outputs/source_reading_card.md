# Source Reading Card：L24 量化 Serving

## 主路径

1. `labs/l23_quant_serving/patch/starter/awq_calibrate.py`
   - 看 `compute_awq_scale()`、`quantize_w8_per_channel()`、`dequantize_w8_per_channel()` 的 TODO。
   - 结论：AWQ scale 是输入通道 scale，per-output scale 是 int8 dequant scale。

2. `labs/l23_quant_serving/patch/reference/awq_calibrate.py`
   - 看 `dim=0`、`scale.unsqueeze(0)`、`per_out_scale.unsqueeze(1)` 和除回 AWQ scale。
   - 结论：shape 方向是本讲最关键的实现细节。

3. `labs/l23_quant_serving/patch/tests/test_patch.py`
   - 看 int8 range、roundtrip error、scale shape、outlier response 和 `alpha=0`。
   - 结论：测试覆盖最小数学合同，不覆盖真实 W4 packed checkpoint。

4. `mini_infra/vllm/quant/calibrator.py`
   - 看 `channel_absmax()`、`smoothquant_scales()` 和 `awq_activation_order()`。
   - 结论：AWQ/SmoothQuant 的 scale 来自 calibration samples。

5. `mini_infra/vllm/quant/awq_loader.py`
   - 看 group layout、group_size 整除检查和 compression ratio。
   - 结论：真实 AWQ W4A16 还要处理 group、zero point、packed int4 和 scale metadata。

6. `mini_infra/sglang/quant/kv_int8.py`
   - 看 symmetric scale、quant/dequant 和 KV memory summary。
   - 结论：KV-int8 降低 KV cache 显存，并需要保存 scale metadata。

7. `mini_infra/vllm/run_engine.py`
   - 看 `quant_payload()` 和量化 artifact 写入。
   - 结论：量化报告要同时记录延迟、吞吐、显存、准确率漂移和硬件边界。

8. `github_repo/vllm/vllm/model_executor/layers/quantization/awq.py`
   - 看 `AWQConfig`、`create_weights()` 和 `apply()`。
   - 结论：真实 AWQ 加载器主要排查 config、shape 对齐、packed parameter 和 kernel 路径。

9. `github_repo/vllm/vllm/model_executor/layers/quantization/fp8.py`
   - 看 `Fp8Config`、weight/input scale 和 activation scheme。
   - 结论：FP8 依赖 scale 记录、硬件能力和动态/静态 activation 配置。

10. `github_repo/vllm/vllm/model_executor/layers/quantization/kv_cache.py`
    - 看 q/k/v/prob scale 初始化和加载后处理。
    - 结论：KV cache 量化的关键是 scale metadata 和 decode 误差边界。

## 快速自检

- 我能否说清 AWQ scale 和 per-output dequant scale 的 shape？
- 我能否解释 `alpha=0` 为什么等价于普通 weight-only INT8？
- 我能否指出真实 AWQ W4A16 比课堂 W8 patch 多出的 checkpoint 字段？
- 我能否说明 CPU smoke 和真实 GPU benchmark 的边界？
- 我能否在 vLLM 源码中找到 AWQ group_size、FP8 input_scale 和 KV k/v scale 的入口？
