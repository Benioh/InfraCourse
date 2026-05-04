# L09 Patch · Prometheus Metrics Exporter

## 你要交付什么

实现一个简化的 **Prometheus exposition 格式 metrics exporter**——LLM serving 系统监控的标准接口：

```python
class MetricsExporter:
    def set_gauge(self, name: str, value: float, labels: dict | None = None): ...
    def inc_counter(self, name: str, by: float = 1, labels: dict | None = None): ...
    def record_event(self, name: str, hit: bool): ...  # 用于 ratio metric
    def export(self) -> str:  # 返回 Prometheus exposition 文本
```

3 类核心 metric:
- **gauge** (instantaneous value, e.g. `prefill_queue_depth`)
- **counter** (monotonically increasing, e.g. `requests_total`)
- **ratio** (rolling hit/miss, e.g. `prefix_cache_hit_rate`)

**禁止** 用 `prometheus_client` 库。
**允许** stdlib `dict` / `time` / 字符串拼接。

补丁规模目标：60–100 行。

## Prometheus 文本格式

```
# HELP <metric_name> <description>
# TYPE <metric_name> <type>
<metric_name>[{label="value",...}] <number>
```

例：
```
# TYPE prefill_queue_depth gauge
prefill_queue_depth 12
# TYPE requests_total counter
requests_total{model="qwen2"} 8421
# TYPE prefix_cache_hit_rate gauge
prefix_cache_hit_rate 0.6234
```

## 接口契约

```python
m = MetricsExporter()
m.set_gauge("prefill_queue_depth", 12)
m.inc_counter("requests_total", labels={"model": "qwen2"})
m.record_event("prefix_cache", hit=True)
m.record_event("prefix_cache", hit=False)
print(m.export())
# 应包含 prefill_queue_depth, requests_total{model="qwen2"}, prefix_cache_hit_rate (0.5)
```

## 不变量

1. `set_gauge` 后 `export()` 出现该 metric 名 + 当前值。
2. `inc_counter` 累加；不同 labels 算不同 series。
3. `record_event` 后 hit_rate = hits / (hits + misses)；零事件返回 0.0（不要 NaN）。
4. labels 序列化必须**字母序**稳定输出（防 export 顺序抖动）。
5. export 字符串必须包含 `# TYPE ` 行。

## 怎么验证

```bash
make patch-test M=l26_sglang_pd_observability
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_gauge_set_and_export` | 设置后 export 包含值 |
| `test_counter_increment` | 多次 inc 后值正确 |
| `test_hit_rate_calculation` | 3 hit / 1 miss → 0.75 |
| `test_zero_events_returns_zero` | 没事件 → 0.0 |
| `test_labels_sorted` | 多 label 按字母序输出 |

## 写完之后你能做什么

- 看懂 vLLM / SGLang 的 prometheus metrics 命名约定。
- 给 Capstone Stage B 多模态服务加 prometheus 端点 + Grafana 面板。
- 调试"P99 latency 突然变高"——用 metric history 定位是 prefill queue 堵了还是 decode 抢不到 GPU。
