# L25 Spec Decode Serving Metrics Template

## 1. 实验信息

| 字段 | 填写 |
|---|---|
| 日期 | |
| git revision | |
| 命令 | |
| 框架和版本 | |
| 模型 / tokenizer | |
| spec method | |
| K / num_speculative_tokens | |
| 采样参数 | |
| 硬件 | |
| 并发 / batch | |
| prompt domain | |

## 2. Baseline

| 指标 | 数值 | 备注 |
|---|---:|---|
| tokens/s | | spec 关闭 |
| TTFT p50 / p99 | | |
| ITL p50 / p99 | | |
| GPU memory peak | | |
| 输出质量指标 | | 例如 exact match、first number match 或业务指标 |

## 3. Spec Decode

| 指标 | 数值 | 备注 |
|---|---:|---|
| acceptance rate | | accepted draft tokens / proposed draft tokens |
| mean acceptance length | | 通常包含 bonus |
| per-position acceptance | | 例如 `[0.72, 0.51, 0.33, 0.19]` |
| drafted throughput | | draft tokens/s |
| accepted throughput | | accepted draft tokens/s |
| tokens/s | | 与 baseline 对比 |
| TTFT p50 / p99 | | |
| ITL p50 / p99 | | |
| draft latency | | |
| draft GPU memory | | |
| total GPU memory peak | | |
| 输出质量指标 | | 与 baseline 同一评测集 |

## 4. 分桶分析

| 分桶 | 观察 | 判断 |
|---|---|---|
| accept=0 | | |
| accept=1..K | | |
| in-domain prompt | | |
| out-of-domain prompt | | |
| short prompt | | |
| long prompt | | |
| low concurrency | | |
| high concurrency | | |

## 5. 结论

- 速度结论：
- p99 风险：
- 显存风险：
- 质量风险：
- 建议动作：
- 仍需补充的数据：
