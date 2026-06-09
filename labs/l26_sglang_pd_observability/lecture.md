# L27：SGLang PD Observability · Prometheus Metrics Exporter

Serving 系统出现 p99 抖动、TTFT 变长或 ITL 变慢时，日志通常只能告诉我们“慢了”。要定位慢在哪里，需要把运行状态拆成指标：请求排队、prefill token、decode batch、KV cache 使用、prefix cache hit-rate、PD queue、KV transfer latency、失败计数。L27 用一个最小 Prometheus exporter 训练这些指标的基本语义，再把它放回 SGLang 的 PD 观测路径。

## 1. 本讲目标

- 写出 Prometheus exposition 文本的最小格式：`# TYPE` 行和 sample 行。
- 区分 gauge、counter 和 hit-rate 的状态更新方式。
- 解释 label 排序和 label 基数为什么会影响可复查性和查询成本。
- 用 TTFT、ITL、queue wait、KV transfer、cache hit-rate 解释 PD 分离的观测边界。
- 读懂 patch、PD lab、SGLang `/v1/loads`、metrics collector 和 scheduler metrics 主路径。

## 2. 问题背景：PD 分离需要阶段化观测

LLM serving 通常分为 prefill 和 decode。Prefill 处理完整 prompt，长 prompt 会影响首 token 前的等待；decode 逐 token 生成，更容易受到 batch 组成、KV cache 读取和显存带宽影响。PD 分离把 prefill 和 decode 放到不同资源池，可以让长 prompt 和长输出分别使用更合适的资源，但也引入路由、队列和 KV transfer 成本。

如果只看一个 p99 latency，无法判断问题来自 prefill queue、decode starvation、prefix cache miss、KV transfer 慢、GPU memory pressure，还是 router 把请求发错 worker。可观测性的价值在于把这些状态变成可对比的时间序列，并和 command、config、workload、日志、artifact 放在同一条证据链里。

本讲 patch 不复刻完整 Prometheus client。它只实现最小 exporter：维护 gauge、counter 和 hit/miss 状态，并导出稳定文本。这个小练习的目标是让学生亲手处理 metric state、series key、label 排序和输出格式。

## 3. Prometheus 文本格式

Prometheus text exposition 最小形态包含类型行和样本行：

```text
# TYPE prefill_queue_depth gauge
prefill_queue_depth{engine="prefill",model="qwen"} 12.0
```

`# TYPE` 告诉查询侧这个 metric 的类型。样本行由 metric name、可选 labels 和数值组成。没有 labels 时直接写 `metric_name value`；有 labels 时写成 `{key="value",...}`。

本讲不要求实现 HELP 行、timestamp、histogram bucket 或 HTTP server，但学生要知道它们在真实系统里存在。SGLang 真实 metrics collector 会创建 gauge、counter、histogram 和 summary；本讲 exporter 只保留最小机制。

## 4. Gauge、Counter 和 Hit-rate

Gauge 表示当前值，可以覆盖。running requests、queue depth、token usage、KV available tokens、cache hit-rate 都适合用 gauge。Counter 表示累计事件，通常只增加。requests total、generated tokens total、bootstrap failed total、transfer failed total 更适合 counter。

Hit-rate 可以由两个计数派生。L27 patch 用 `record_event(name, hit)` 维护 `(hits, misses)`，导出时输出 `<name>_hit_rate = hits / (hits + misses)`。没有事件时输出 0.0，只有 miss 时也输出 0.0，避免除零和 NaN。

Labels 用来区分同名 metric 的维度，例如 `model_name`、`engine_type`、`tp_rank`、`dp_rank`。本讲要求 label 按字母序输出，原因是同一个 label set 应该得到稳定文本，便于测试、diff 和人眼检查。生产里还要控制 label 基数，不能把 request id、prompt hash 这类高基数字段随意放进 label。

## 5. Patch 合同

`MetricsExporter` 的状态很小：

```python
self.gauges[(name, label_key)] = value
self.counters[(name, label_key)] = value
self.ratios[name] = (hits, misses)
```

`_label_key(labels)` 把字典变成按 key 排序的 tuple。这样 dict 输入顺序不同，也会得到同一个 series key。`_format_labels` 负责把 labels 还原成 Prometheus 文本。

四个方法的合同是：

1. `set_gauge(name, value, labels)`：覆盖当前值。
2. `inc_counter(name, by, labels)`：在原值上累加。
3. `record_event(name, hit)`：更新 hits 或 misses。
4. `export()`：按 metric name 排序输出 gauge、counter 和 hit-rate。

测试覆盖五个边界：gauge 输出、counter 累加、hit-rate 3/4 等于 0.75、只有 miss 时 hit-rate 为 0.0、labels 按 `a,m,z` 排序。测试不解析完整 Prometheus 文本，也不检查 histogram；它只守住本讲最小合同。

## 6. PD Lab：从指标形状理解资源分配

`scripts/run_pd_lab.py` 使用三个配置：`unified_tp8`、`pd_2p6d`、`pd_4p4d`。模拟函数里，prefill GPU 越多，`ttft_ms_p50` 越低；decode GPU 越多，`itl_ms_p50` 越低；prefill/decode 资源越不平衡，`queue_wait_ms` 越高。这个模型训练的是观测字段的解释方式，不能当作真实性能结论。

真实 PD 排障至少要看这些组：

| 阶段 | 指标方向 |
|---|---|
| prefill queue | prefill prealloc queue、inflight queue、TTFT |
| decode queue | decode prealloc queue、transfer queue、ITL/TPOT |
| KV transfer | transfer latency、speed、size、failed counters |
| cache | cache hit-rate、cached tokens、KV available/used tokens |
| runtime | running requests、queue requests、token usage、throughput |

PD 分离不是一个单向优化开关。它把资源池拆开后，必须让 workload 与 prefill/decode 配比匹配；否则 queue wait 或 KV transfer 可能吃掉收益。

## 7. SGLang 真实指标路径

SGLang 的 `/v1/loads` endpoint 可以返回 scheduler load，也可以按 Prometheus text 格式导出。它从 dataclass metadata 生成 metric name、HELP、TYPE 和 per-rank sample。这个路径和本讲 exporter 的相同点是：都要把内部状态变成稳定文本；不同点是：真实实现要处理 optional sections、DP rank、HTTP Response 和 dataclass schema。

`metrics_collector.py` 定义了生产级 metrics。基础 gauge 包括 running requests、used tokens、token usage、queue requests、cache hit-rate、KV available/used tokens；spec decode 还有 acceptance 指标；PD 部分包括 prefill/decode queue、bootstrap/transfer failed counters、KV transfer speed/latency/size histograms。

`scheduler_metrics_mixin.py` 是指标写入路径。Prefill 统计会计算 input throughput、cache hit-rate、queue 和 PD prefill queue；decode 统计会记录 generation throughput、spec acceptance、decode queue 和 routing key 分布。读源码时要沿着“状态在哪里生成、写入哪个 metric、用什么 label 导出”的顺序走。

## 8. Debug 路线

Metrics missing 时，先查服务是否打开 metrics，再查 endpoint 和 scrape target。SGLang 需要相应 server args 打开 metrics；`/v1/loads` 要确认 format、include 和 dp_rank 参数；Prometheus 抓取要确认 target、path 和 label 过滤。不要先假设业务没流量。

PD decode starvation 时，先看 decode queue、transfer queue、ITL/TPOT 和 running requests，再看 prefill 侧是否堆积、KV transfer 是否慢、decode GPU 是否不足。若 prefill queue 很低但 decode queue 很高，说明瓶颈不在 prompt 处理；若 transfer latency 或 failed counters 增加，要查网络、connector 和 KV size。

Label 问题要看 series 数量。若每个 request id 都变成 label，Prometheus 会产生大量 series，查询和存储都会变慢。生产 label 应该是低基数维度，例如 model、engine_type、rank、stage、status。

## Lab 验收边界

本讲 patch 命令：

```bash
make patch-test M=l26_sglang_pd_observability
```

patch 验收的是最小 exporter，不包括 HTTP server、histogram bucket、scrape interval、PromQL、Grafana dashboard 或多进程 registry。测试通过后，还要跑 PD lab，检查 `metrics.jsonl`、`pd_comparison.json` 和 `report.md` 是否能支持“哪个配置降低 TTFT、哪个配置降低 ITL”的判断。

## 9. 小结

L27 的核心是把服务状态变成可靠指标。Exporter 的代码很短，但它训练的是可观测性基本功：选对 metric 类型、维护正确状态、稳定输出 label、控制证据链。进入真实 SGLang PD 后，同样的原则会扩展到 queue、KV transfer、cache、throughput 和 p99 排障。
