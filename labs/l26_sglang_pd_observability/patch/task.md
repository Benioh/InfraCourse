# L27 Patch · Prometheus Metrics Exporter

## 你要交付什么

实现一个简化的 Prometheus exposition 文本 exporter。它用于训练 LLM serving 可观测性的最小合同：把运行时状态稳定地导出为可测试、可对比的 metrics 文本。

```python
class MetricsExporter:
    def set_gauge(self, name: str, value: float, labels: dict | None = None): ...
    def inc_counter(self, name: str, by: float = 1, labels: dict | None = None): ...
    def record_event(self, name: str, hit: bool): ...
    def export(self) -> str: ...
```

三类核心 metric：

- `gauge`：当前值，例如 `prefill_queue_depth`、`num_running_reqs`、`token_usage`。
- `counter`：累计事件，例如 `requests_total`、`tokens_total`、`transfer_failed_total`。
- `hit-rate`：由 hits/misses 派生，例如 `prefix_cache_hit_rate`。

限制：

- 禁止使用 `prometheus_client`。
- 可以使用 Python 标准库的 `dict`、`tuple`、`collections` 和字符串处理。
- 补丁规模目标为 60 到 100 行。

## Prometheus 文本格式

本补丁只要求最小文本格式：

```text
# TYPE <metric_name> <type>
<metric_name>[{label="value",...}] <number>
```

示例：

```text
# TYPE prefill_queue_depth gauge
prefill_queue_depth 12.0
# TYPE requests_total counter
requests_total{model="qwen2"} 8421.0
# TYPE prefix_cache_hit_rate gauge
prefix_cache_hit_rate 0.5
```

真实 Prometheus exposition 还可能包含 HELP、timestamp、histogram bucket 和 HTTP scrape endpoint。本补丁暂不实现这些内容。

## 接口合同

```python
m = MetricsExporter()
m.set_gauge("prefill_queue_depth", 12)
m.inc_counter("requests_total", labels={"model": "qwen2"})
m.record_event("prefix_cache", hit=True)
m.record_event("prefix_cache", hit=False)
print(m.export())
```

输出应包含：

- `# TYPE prefill_queue_depth gauge`
- `prefill_queue_depth 12`
- `# TYPE requests_total counter`
- `requests_total{model="qwen2"} 1`
- `# TYPE prefix_cache_hit_rate gauge`
- `prefix_cache_hit_rate 0.5`

## 不变量

1. `set_gauge` 对同一个 `(name, labels)` 覆盖当前值。
2. `inc_counter` 对同一个 `(name, labels)` 累加；不同 labels 是不同 series。
3. `record_event` 维护 hits/misses，导出时计算 `hits / (hits + misses)`。
4. 没有 hit 或只有 miss 时，hit-rate 导出为 `0.0`，不能出现 NaN。
5. labels 必须按 key 字母序输出，避免文本顺序抖动。
6. `export()` 必须包含 `# TYPE` 行，并以换行结尾。

## 怎么验证

```bash
make patch-test M=l26_sglang_pd_observability
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_gauge_set_and_export` | gauge 设置后 export 包含 TYPE 和当前值 |
| `test_counter_increment` | 多次 counter inc 后值正确 |
| `test_hit_rate_calculation` | 3 hit / 1 miss 导出 0.75 |
| `test_zero_events_returns_zero` | 只有 miss 时导出 0.0 |
| `test_labels_sorted` | 多 label 按字母序输出 |

## 写完之后你能做什么

- 读懂 SGLang 和 vLLM 中常见 serving metrics 的类型边界。
- 判断 queue、token usage、cache hit-rate、failed counter 和 latency histogram 应该如何查询。
- 排查 p99 抬高时，把现象拆到 prefill queue、decode queue、KV transfer、cache 或 scrape 配置。
