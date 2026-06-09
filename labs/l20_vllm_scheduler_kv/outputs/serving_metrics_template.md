# Serving 指标复盘模板

完成 `run_serving_drill.sh` 或真实 vLLM benchmark 后，用这份模板复盘。重点是把指标和 scheduler / KV cache 的状态联系起来。

## 1. 实验配置

| 项 | 值 |
|---|---|
| profile |  |
| impl | starter / reference / real vLLM |
| model |  |
| GPU |  |
| num_blocks / KV capacity |  |
| max_num_running_reqs / max_num_seqs |  |
| max_model_len |  |
| max_num_batched_tokens |  |
| prefix cache | on / off |
| chunked prefill | on / off |

## 2. Workload

| 项 | 值 |
|---|---|
| requests_total |  |
| concurrency / arrival pattern |  |
| prompt tokens p50 / p90 / p99 |  |
| max_tokens p50 / p90 / p99 |  |
| repeated prefix ratio |  |
| streaming | on / off |

## 3. 结果

| 指标 | 值 |
|---|---|
| p50 TTFT |  |
| p95 TTFT |  |
| p50 ITL |  |
| p95 ITL |  |
| throughput |  |
| p50 KV usage |  |
| p99 KV usage |  |
| max waiting length |  |
| requests finished / total |  |
| preemptions |  |
| prefix cache hit rate |  |

## 4. 观察

- TTFT 主要受什么影响：
- ITL 主要受什么影响：
- waiting 是否在 KV pressure 下增长：
- finished 后 KV usage 是否下降：
- 是否出现明显 preemption 或重算：

## 5. 结论

本次瓶颈判断：

```text
queue / prefill / decode / KV admission / output processing / client measurement
```

证据：

1.
2.
3.

下一步动作：

1.
2.
3.
