# L27 Source Reading Card：SGLang PD Observability

这张卡用于快速回忆 L27 的源码主路径。读源码时先抓状态来源和写入位置，再看扩展分支。

## 1. Patch 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `labs/l26_sglang_pd_observability/patch/starter/metrics_exporter.py` | `_label_key`、`_format_labels`、三张状态表、四个 TODO | exporter 的最小合同是状态维护和稳定文本导出 |
| `labs/l26_sglang_pd_observability/patch/reference/metrics_exporter.py` | `set_gauge`、`inc_counter`、`record_event`、`export` | gauge 覆盖、counter 累加、hit-rate 导出时计算 |
| `labs/l26_sglang_pd_observability/patch/tests/test_patch.py` | 五个 pytest | 最小验收覆盖数值语义、边界和 label 顺序 |

## 2. Drill 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `labs/l26_sglang_pd_observability/scripts/run_pd_lab.py` | `simulate`、配置循环、metrics/report 写入 | CPU drill 训练指标解释，不给出真实性能结论 |
| `mini_infra/observability/evidence.py` | `summarize_run`、`scan` | run 是否可复查取决于 command、config、metrics、artifact、report |
| `mini_infra/sglang/srt/managers/disagg_service.py` | `route_request`、`transfer_kv`、`metrics` | PD 路由要同时观察 worker load、cached prefix、KV transfer 和 active requests |

## 3. SGLang 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `github_repo/sglang/python/sglang/srt/entrypoints/v1_loads.py` | `_compute_aggregate`、`_format_loads_prometheus`、`get_loads` | `/v1/loads` 把 scheduler load 结构化输出，也可导出 Prometheus 文本 |
| `github_repo/sglang/python/sglang/srt/observability/metrics_collector.py` | `SchedulerStats`、Gauge/Counter/Histogram 定义、`log_stats` | 真实 metrics 分类型定义，scheduler stats 统一写入 collector |
| `github_repo/sglang/python/sglang/srt/observability/scheduler_metrics_mixin.py` | `init_metrics`、`report_prefill_stats`、`report_decode_stats` | prefill 和 decode 阶段分别填充 queue、cache、throughput、spec 和 PD stats |

## 4. 读源码时的提问顺序

1. 这个字段描述的是当前状态、累计事件，还是延迟分布？
2. 它的 label 是低基数系统维度，还是高基数请求维度？
3. 状态在哪里产生，在哪里更新，在哪里导出？
4. 指标异常指向 queue、prefill、decode、KV transfer、cache，还是 scrape 配置？
5. 这段代码能支撑课堂 patch 的哪个最小合同？

## 5. 本讲最小不变量

- Gauge 表示当前值，重复写入应覆盖。
- Counter 表示累计事件，重复写入应累加。
- Hit-rate 由 hits 和 misses 计算，分母为 0 时返回 0.0。
- Label 输出要稳定，生产 label 要控制基数。
- Prometheus 文本至少包含 `# TYPE` 和 sample 行。
- PD 排障不能只看 p99，要拆成 queue、prefill、decode、KV transfer 和 cache。
- 一次 run 的结论必须能回到 command、config、metrics、artifact 和 report。
