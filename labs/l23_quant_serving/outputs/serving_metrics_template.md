# L24 Quant Serving 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- Python / PyTorch / CUDA：
- git commit：
- 命令：
- 配置文件：
- 模型：
- quant method：
- weight bits / activation dtype：
- group_size / zero_point：
- calibration set：
- eval set：
- workload：batch、context length、concurrency、prompt 分布

## 本次要验证什么

- 比较对象：
- 成功标准：
- 预期显存变化：
- 预期速度变化：
- 允许 accuracy drift：
- 不能外推的条件：

## 指标记录

| method | weight_mem_gb | peak_kv_mem_gb | ttft_p50 | ttft_p99 | itl_p50 | itl_p99 | tokens/s | acc_drop_pp | cache_hit_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 |  |  |  |  |  |  |  |  |  |
| awq |  |  |  |  |  |  |  |  |  |
| fp8 |  |  |  |  |  |  |  |  |  |
| kv-int8 |  |  |  |  |  |  |  |  |  |

## Artifact

| 文件 | 路径 | 检查结果 |
|---|---|---|
| command snapshot |  |  |
| resolved config |  |  |
| quant_matrix.json |  |  |
| accuracy_drift.json |  |  |
| benchmark log |  |  |
| eval result |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| AWQ scale shape / roundtrip | `patch/reference/awq_calibrate.py` |  |
| calibration samples | `mini_infra/vllm/quant/calibrator.py` |  |
| AWQ group layout | `github_repo/vllm/vllm/model_executor/layers/quantization/awq.py` |  |
| FP8 scale / activation scheme | `github_repo/vllm/vllm/model_executor/layers/quantization/fp8.py` |  |
| KV scale metadata | `github_repo/vllm/vllm/model_executor/layers/quantization/kv_cache.py` |  |

## 结论

- 本次能证明：
- 本次不能证明：
- 下一步动作：
