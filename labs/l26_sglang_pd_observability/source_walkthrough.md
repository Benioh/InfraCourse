# 源码带读：L27 SGLang PD Observability

这份带读按“最小 exporter -> 本地 PD 证据 -> SGLang 真实指标”的顺序走。不要从文件顶部一路读到底，先按下面的锚点把状态来源、写入位置和导出格式串起来。

## 0. 源码地图

```text
labs/l26_sglang_pd_observability/patch/starter/metrics_exporter.py
  -> labs/l26_sglang_pd_observability/patch/reference/metrics_exporter.py
  -> labs/l26_sglang_pd_observability/patch/tests/test_patch.py

labs/l26_sglang_pd_observability/scripts/run_pd_lab.py
  -> runs/mini_infra/l26_sglang_pd_observability/<run-id>/
     -> metrics.jsonl
     -> artifacts/pd_comparison.json
     -> report.md

github_repo/sglang/python/sglang/srt/entrypoints/v1_loads.py
github_repo/sglang/python/sglang/srt/observability/metrics_collector.py
github_repo/sglang/python/sglang/srt/observability/scheduler_metrics_mixin.py

mini_infra/observability/evidence.py
mini_infra/sglang/srt/managers/disagg_service.py
```

Patch 负责最小合同，PD lab 负责生成可复查 artifact，SGLang 源码负责展示生产系统如何扩展同一套观测语义。

## 1. Patch starter：先看状态表和 TODO 边界

文件：[patch/starter/metrics_exporter.py](patch/starter/metrics_exporter.py)

先看第 18-28 行。`_label_key` 把 label 字典变成可排序、可作为 dict key 的 tuple；`_format_labels` 把 label 字典打印成 Prometheus 文本。这里要记住一个原则：写入时规范化，导出时稳定打印。

再看第 31-38 行。`MetricsExporter` 只有三张状态表：gauges、counters、ratios。它没有 HTTP server、registry、多进程聚合或 histogram，所以本讲 patch 的边界很清楚：维护最小 metric state，并导出文本。

最后看第 40-82 行。四个 TODO 分别对应 gauge 覆盖写、counter 累加写、hit/miss 计数和 Prometheus 文本导出。写 patch 前先用自己的话复述每个方法的输入、状态变化和输出。

可以先跳过：

- Prometheus HELP 行、timestamp、scrape endpoint 和 PromQL 查询。
- 真实 SGLang 多进程 metrics registry。
- histogram bucket 和 summary 分位数。

## 2. Patch reference：把三张状态表读成一个确定性投影

文件：[patch/reference/metrics_exporter.py](patch/reference/metrics_exporter.py)

重点看第 28-43 行。`set_gauge` 用 `(name, label_key)` 覆盖当前值；`inc_counter` 用同一个 key 累加；`record_event` 只维护 `(hits, misses)`，不提前保存 rate。这里能看到三类指标的更新语义完全不同。

再看第 45-74 行。`export` 先按 metric name 分组，再排序输出 label series；ratio 在导出时才计算 `<name>_hit_rate`。这里的核心是把当前状态稳定地投影成文本，而不只是拼接字符串。

读完 reference 后要能回答：

1. 为什么 label dict 不能直接作为 `self.gauges` 的 key？
2. 为什么 `set_gauge` 不能累加旧值？
3. 为什么 ratio 内部保存 hits/misses，而不是只保存最后一次 True/False？
4. 为什么 `export()` 最后保留一个换行？

## 3. Patch tests：用五个合同锁住最小语义

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

按测试顺序读：

- 第 21-27 行：gauge 写入后要有 TYPE 行和样本行。
- 第 30-39 行：counter 多次 `inc` 后要累加到 5。
- 第 42-52 行：3 hit / 1 miss 要导出 0.75。
- 第 55-62 行：只有 miss 时不能除零，hit-rate 是 0.0。
- 第 65-71 行：乱序 labels 要输出为 `a,m,z`。

测试没有覆盖完整 Prometheus parser、HELP 行、负数 counter、HTTP endpoint 或 histogram。它只守住 L27 最小证据链：数值语义正确，文本稳定，可被后续 artifact 对比。

## 4. PD lab：看模拟指标如何落盘

文件：[scripts/run_pd_lab.py](scripts/run_pd_lab.py)

先看第 26-37 行。`simulate` 接收配置名、prefill GPU 数和 decode GPU 数，输出 TTFT、ITL、queue wait 和 routing errors。这个公式不是性能结论，它只是把 PD 资源配比投影成几类观测字段。

再看第 40-69 行。脚本准备 run 目录，写入 command、prediction、resolved config，然后依次加载 `unified_tp8.yaml`、`pd_2p6d.yaml`、`pd_4p4d.yaml`，每个配置写一行 `metrics.jsonl`。这一步告诉你课堂 drill 的证据在哪里。

最后看第 74-112 行。脚本把最低 TTFT、最低 ITL 写入 report，并保留 `pd_comparison.json`。读完后要能说清：一次 run 的结论必须由命令、配置、metrics、artifact 和 report 共同支撑。

## 5. SGLang `/v1/loads`：结构化负载到 Prometheus 文本

文件：[v1_loads.py](../../github_repo/sglang/python/sglang/srt/entrypoints/v1_loads.py)

先看第 61-87 行。`_compute_aggregate` 把多个 DP rank 的 running、waiting、used tokens、throughput 和 utilization 汇总。它解决的是“多 rank 负载如何形成整体视图”。

再看第 90-130 行。`_format_loads_prometheus` 遍历 dataclass field metadata，为每个字段生成 HELP、TYPE 和带 `dp_rank` label 的样本。它和本讲 exporter 的同构点是：内部状态先有结构化字段，再按 metric 类型导出文本。

最后看第 134-165 行。`get_loads` 接收 `dp_rank`、`include` 和 `format` 参数；当 `format == "prometheus"` 时返回文本格式。读完后要能解释 `/v1/loads` 为什么既能服务 router，也能作为观测入口。

可以先跳过：

- FastAPI 依赖注入细节。
- optional sections 的完整 dataclass 定义。
- 非 Prometheus JSON 响应字段的全部展开。

## 6. SGLang metrics collector：看 metric 类型如何定义

文件：[metrics_collector.py](../../github_repo/sglang/python/sglang/srt/observability/metrics_collector.py)

先看第 77-115 行。`SchedulerStats` 把 running queue、token usage、cache hit、spec decode、PD queue 和 KV transfer 统一放进一个 stats 对象。scheduler 后续会填这个对象，再交给 collector。

再看第 199-316 行。这里创建基础 gauges：running requests、used tokens、token usage、generation throughput、queue requests、cache hit-rate、KV available/used tokens 和 spec acceptance。注意不同 metric 类型对应不同查询语义。

接着看第 347-416 行。PD 相关 metrics 被拆成 queue gauges、failed counters 和 KV transfer histograms。queue 是当前状态，失败数是累计事件，latency/speed/size 需要分布。

最后看第 1019-1058 行。`log_stats` 把 `SchedulerStats` 写入真实 Prometheus metrics。读这段时不要背 metric 名字，重点看每个 stats 字段如何进入 gauge、counter 或 histogram。

## 7. Scheduler metrics mixin：看指标在哪里被填充

文件：[scheduler_metrics_mixin.py](../../github_repo/sglang/python/sglang/srt/observability/scheduler_metrics_mixin.py)

先看第 123-147 行。`init_metrics` 按 model、engine type、tp rank、pp rank、moe ep rank、dp rank 和额外 labels 创建 collector。这里的 label 都是低基数系统维度，不是 request id。

再看第 411-457 行。prefill 统计会计算 cache hit-rate、queue、token usage、PD prefill queue 和 KV transfer 速度/延迟，然后调用 `metrics_collector.log_stats`。这是 TTFT 和 prefill 阶段排障的主路径。

最后看第 591-650 行。decode 统计会写 running requests、decode sequence lengths、generation throughput、queue、spec acceptance、PD decode queue 和 routing key 统计。ITL/TPOT 和 decode starvation 要从这条路径找证据。

可以先跳过：

- MFU 估算里的 FLOPs/bytes 公式。
- CUDA graph、LoRA、HiCache 和 streaming session 的完整分支。
- 多模态 encoder transfer 的暂时分支。

## 8. MiniInfra evidence 和 disagg service：把证据链收回来

文件：[mini_infra/observability/evidence.py](../../mini_infra/observability/evidence.py)

第 11-29 行说明一个 run 是否可复查：command、config、report、metrics 数量、latest metric 和 artifact 数量都要能被扫描出来。这个工具的价值是把“我看到了一个慢 run”变成可比较的记录。

文件：[mini_infra/sglang/srt/managers/disagg_service.py](../../mini_infra/sglang/srt/managers/disagg_service.py)

第 68-108 行展示最小 PD 路由：校验请求、计算 prefill tokens、选择 prefill/decode worker、记录 KV transfer 和 route。第 125-153 行把 transfers、tokens、cached prefix、active requests 和 worker loads 转成 metrics dict。它是下一讲 L28 的前置线索。

## 读完后的自检问题

1. L27 patch 的三张状态表分别保存什么？
2. 一条 Prometheus 样本里，metric name、labels、value 分别来自哪里？
3. 为什么 queue depth 应该是 gauge，failed requests total 应该是 counter？
4. `/v1/loads?format=prometheus` 和本讲 exporter 在导出思路上有什么相同点？
5. SGLang PD 排障时，prefill queue、decode transfer queue、KV transfer latency 和 cache hit-rate 分别指向哪类问题？
6. 一次 drill 生成的 command、config、metrics、artifact、report 缺一项时，结论会缺什么证据？
