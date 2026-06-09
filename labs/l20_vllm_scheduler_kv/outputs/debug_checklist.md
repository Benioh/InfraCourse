# vLLM Scheduler / KV Cache 排障 Checklist

这张 checklist 用来排查 serving 变慢、TTFT 尾部升高、waiting 队列增长、KV usage 接近上限这类问题。

## 1. 先确认 workload

- [ ] prompt 长度分布是否变化，特别是 p90 / p99 prompt tokens。
- [ ] `max_tokens` 或平均输出长度是否变化。
- [ ] 并发数、到达速率、burst 形态是否变化。
- [ ] 是否打开 streaming，TTFT 的测量口径是否一致。
- [ ] 是否存在大量共享 system prompt 或重复前缀。

## 2. 再看请求生命周期

- [ ] 请求是否已经到达 OpenAI-compatible API。
- [ ] 请求是否进入 engine/core。
- [ ] waiting queue 是否增长。
- [ ] running 请求数是否长期贴近 `max_num_seqs` 或 scheduler running 上限。
- [ ] finished 请求是否持续出现。
- [ ] finished 后 KV usage 是否下降。

## 3. 看 KV cache 压力

- [ ] KV usage 是否接近 1.0。
- [ ] free blocks 是否持续减少。
- [ ] 是否出现 preemption 增多。
- [ ] prefix cache hit 是否下降。
- [ ] 长 prompt 是否让单请求 block 需求变大。
- [ ] `max_model_len` 是否过大，导致可用 KV blocks 变少。

## 4. 看 token budget

- [ ] `max_num_batched_tokens` 是否限制了每 step 可处理 token。
- [ ] 长 prompt 是否挤占了 decode 请求的 step budget。
- [ ] 是否开启 chunked prefill。
- [ ] chunked prefill 打开后，TTFT 和 ITL 是否分阶段改善或恶化。

## 5. 看 admission 参数

- [ ] `max_num_seqs` 是否过大，导致 active set 过宽。
- [ ] `gpu_memory_utilization` 是否过高，留给其他开销的余量不足。
- [ ] `max_model_len`、`max_num_batched_tokens`、`max_num_seqs` 是否一起调过。
- [ ] 真实请求长度是否超过配置设计时的假设。

## 6. 看释放和泄漏

- [ ] finished 请求是否调用 KV free。
- [ ] aborted / cancelled 请求是否释放 KV。
- [ ] preempted 请求是否释放 KV 并回到 waiting。
- [ ] encoder cache 或多模态 cache 是否同步释放。
- [ ] usage 只涨不降时，能否定位到 request owner。

## 7. 形成结论

排查结论建议写成下面格式：

```md
现象：
- TTFT p95 从 __ ms 到 __ ms
- waiting queue p95 从 __ 到 __
- KV usage p99 为 __

判断：
- 主要瓶颈在 queue / prefill / decode / KV admission / output processing

证据：
- 指标 1
- 指标 2
- 源码路径或日志字段

处理：
- 调整参数 / workload 分流 / 打开 chunked prefill / 优化 prefix cache / 降低 max_model_len

风险：
- 对吞吐、ITL、显存余量、preemption 的潜在影响
```

