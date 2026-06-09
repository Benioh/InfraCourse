# L22 Serving Metrics 复盘模板

## 1. 运行信息

| 项 | 值 |
|---|---|
| run id |  |
| command |  |
| config |  |
| status | served / validation_only |
| model |  |
| host:port |  |
| hardware |  |
| SGLang importable | yes / no |
| CUDA available | yes / no |

## 2. Workload

| 项 | 值 |
|---|---|
| workload | repeated_prefix / short / high_concurrency |
| shared prefix tokens |  |
| suffix tokens p50/p95 |  |
| request count |  |
| max_new_tokens |  |
| temperature |  |
| namespace / extra_key |  |

## 3. 指标

| 指标 | 值 | 解释 |
|---|---|---|
| TTFT p50 |  |  |
| TTFT p95 |  |  |
| requests/sec |  |  |
| output tokens/sec |  |  |
| cache_hit_rate |  |  |
| matched_prefix_tokens |  |  |
| validation_only |  |  |

## 4. 证据

- `command.sh`：
- `config.resolved.yaml`：
- `serve.log`：
- `metrics.jsonl`：
- token id LCP 证据：
- 源码路径：

## 5. 判断

本次主要瓶颈：

```text
queue / tokenize / prefix cache miss / suffix prefill / decode / output / measurement
```

本次能证明什么：

1.
2.
3.

本次不能证明什么：

1.
2.
3.

下一步动作：

1.
2.
3.
