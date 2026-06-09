# L24 · AWQ-lite W8 Per-Channel 量化 Serving

<!-- LECTURE_FIRST_START -->

本讲讲推理服务里的 weight quantization。模型权重占用越大，留给 KV cache、并发 batch 和长上下文的显存越少；低精度 scale 选错时，服务仍能启动，但 logits、accuracy drift 和长上下文输出会快速恶化。L24 用一个 CPU-safe 的 AWQ-lite W8 patch 建立量化数学合同，再把它接回 MiniInfra 的 calibration、AWQ/FP8/KV-int8 指标和真实 vLLM 加载器。

## 学习路线

建议按下面顺序走，先把系统讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L24 在 Serving 性能与显存路径中的位置。
2. 读 [lecture.md](lecture.md)：从权重显存、scale 选择、AWQ-lite 三步函数讲到生产量化排查。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra 和真实 vLLM quantization 源码阅读。
4. 跑 notebook：[n17_quant_calibration.ipynb](../../notebooks/n17_quant_calibration.ipynb)，观察 activation amax 和 calibration 直觉。
5. 做 quiz：确认 W8A16、per-channel scale、AWQ、FP8 和 KV scale 边界。
6. 做 patch：实现最小行为合同并通过测试。
7. 跑 smoke：生成量化 serving 指标矩阵和 artifact。
8. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)，沉淀显存、吞吐和准确率漂移结论。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Serving systems / quantized serving |
| 核心瓶颈 | 权重显存、权重读取带宽、KV cache 显存、低精度误差和加载器 shape 约束 |
| 关键机制 | activation-aware input-channel scale、per-output-channel INT8 quant/dequant、calibration 指标和真实加载器 scale metadata |
| 源码落点 | `patch/reference/awq_calibrate.py`、`mini_infra/vllm/quant/*`、`mini_infra/vllm/run_engine.py`、vLLM AWQ/FP8/KV cache quantization |
| lab 检验 | AWQ scale shape、int8 范围、roundtrip 误差、outlier 响应、`alpha=0` 退化路径 |

## 学完后能做什么

- 区分 W8A16、W4A16、W8A8、FP8 和 KV-int8 分别量化了哪些张量。
- 解释 AWQ-lite 为什么用 activation amax 保护输入通道，以及它和 per-output dequant scale 的区别。
- 写出 `compute_awq_scale`、`quantize_w8_per_channel`、`dequantize_w8_per_channel` 的 shape 合同。
- 判断一次量化报告是否同时记录了显存、速度、准确率漂移、校准数据和硬件边界。
- 看懂真实 vLLM AWQ/FP8/KV cache 加载器里 group_size、pack_factor、scale 和 activation_scheme 的排查入口。

## Patch 闭环

```bash
cat labs/l23_quant_serving/patch/task.md
$EDITOR labs/l23_quant_serving/patch/starter/awq_calibrate.py
make patch-test M=l23_quant_serving
```

smoke：

```bash
python labs/l23_quant_serving/scripts/run_smoke.py --run-id l24_smoke --mode smoke
```

量化指标矩阵：

```bash
python labs/l23_quant_serving/scripts/bench_acc.py --eval gsm8k --servers 8001,8002,8003,8004
```

CPU smoke 只能证明配置、指标字段和 artifact 路径。真实速度结论必须在支持对应低精度 kernel 的 GPU 上，用相同模型、workload 和评估集对比。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查量化加载失败、scale shape 错误、accuracy drift 和 KV-int8 漂移 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、MiniInfra 和真实 vLLM quantization 主路径 |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 记录一次量化 serving 的显存、速度、准确率和硬件边界 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L24 后进入 L25 Speculative Decoding。下一讲会继续使用本讲的报告习惯：速度提升必须和准确率、显存、硬件、workload、校准或验证数据一起记录。
