# L23 Attention Benchmark 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- Python / PyTorch / CUDA：
- git commit：
- 命令：
- 配置文件：
- 实现选择：`IMPL=starter` / `IMPL=reference` / 其他
- workload：`B/H/T/D`、dtype、device、causal、mask、iters、warmup

## 本次要验证什么

- 比较对象：
- 成功标准：
- 数值容差：
- 预期 backend：
- 不能外推的条件：

## 指标记录

| seq_len | heads | head_dim | dtype | device | eager_ms | sdpa_ms | speedup | eager_peak_mb | sdpa_peak_mb | max_abs_diff |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |  |  |

## Artifact

| 文件 | 路径 | 检查结果 |
|---|---|---|
| command snapshot |  |  |
| resolved config |  |  |
| metrics.jsonl |  |  |
| bench.csv |  |  |
| bench_summary.json |  |  |
| report.md |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| 数值一致 / 不一致 | `patch/reference/flash_bench.py` |  |
| backend 或 layout 条件 | `github_repo/torchtitan/torchtitan/models/common/attention.py` |  |
| KV cache 输入 | `github_repo/sglang/python/sglang/srt/layers/attention/torch_native_backend.py` |  |
| mask / GQA 参数 | `github_repo/vllm/vllm/v1/attention/backends/cpu_attn.py` |  |

## 结论

- 本次能证明：
- 本次不能证明：
- 下一步动作：
