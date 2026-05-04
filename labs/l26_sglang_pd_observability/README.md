# L09 · SGLang PD：Prometheus Metrics Exporter

> 本关只做一件事：**实现一个支持 gauge / counter / hit-rate 的 Prometheus exposition 文本生成器**。

## 闭环

```bash
cat labs/l26_sglang_pd_observability/patch/task.md
$EDITOR labs/l26_sglang_pd_observability/patch/starter/metrics_exporter.py
make patch-test M=l26_sglang_pd_observability
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_gauge_set_and_export` | 设置后 export 出现 |
| `test_counter_increment` | 多次 inc 后值正确 |
| `test_hit_rate_calculation` | 3 hit / 1 miss → 0.75 |
| `test_zero_events_returns_zero` | 0 events 不 NaN |
| `test_labels_sorted` | 多 label 字母序输出 |

## 卡住怎么办

1. 看 SGLang scheduler / Mooncake PD paper。
2. `make patch-hint M=l26_sglang_pd_observability`。
3. `make patch-show-solution M=l26_sglang_pd_observability`。

## 进入下一关

`make patch-test` 全绿后，下一关 [L09.5 SGLang PD service](../l27_sglang_pd_cache/README.md) 会补 KV transfer 主线。
