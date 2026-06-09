# L27 Debug Checklist：SGLang PD Observability

这张 checklist 面向 metrics missing、p99 抬高、TTFT 变长、ITL 变慢、decode starvation、KV transfer 慢和 label cardinality 失控。

## 1. 固定现场

- [ ] 记录 run id、命令、git commit、Python 环境、服务启动参数和配置文件。
- [ ] 记录模型、并发、prompt 长度分布、输出长度分布和是否 streaming。
- [ ] 保存 `command.sh`、`config.resolved.yaml`、`metrics.jsonl`、`serve.log`、`report.md` 和 artifacts。
- [ ] 明确指标来自 patch、MiniInfra drill、`/metrics`，还是 `/v1/loads?format=prometheus`。
- [ ] 确认本次只改变一个变量，例如 PD 配比、batch、模型、workload 或 scrape 配置。

## 2. Metrics Missing

- [ ] 服务是否打开 metrics 参数。
- [ ] endpoint 路径是否正确，`/metrics` 和 `/v1/loads` 不要混用。
- [ ] `format=prometheus`、`include`、`dp_rank` 参数是否符合预期。
- [ ] Prometheus scrape target、path、端口和网络连通性是否正确。
- [ ] label 过滤是否把目标 series 排除了。
- [ ] 指标是否存在但名字或 label 与 dashboard 查询不一致。

## 3. Exporter 文本稳定性

- [ ] 每个 metric 是否有 `# TYPE` 行。
- [ ] gauge 是否覆盖当前值，counter 是否累加。
- [ ] hit-rate 是否由 hits/misses 计算，只有 miss 时是否为 0.0。
- [ ] label 是否按字母序输出。
- [ ] 相同 label set 是否映射到同一条 series。
- [ ] 是否把 request id、session id、prompt hash 放进 label。

## 4. 阶段化定位

| 现象 | 先看指标 | 可能方向 |
|---|---|---|
| TTFT 变长 | prefill queue、prompt tokens、cache hit-rate、KV available tokens | prefill 堵塞、cache miss、KV 容量压力 |
| ITL/TPOT 变慢 | decode running、decode queue、gen throughput、decode seq lens | decode batch 过宽、显存带宽压力、worker 不足 |
| p99 抖动 | queue wait、per-stage latency、KV transfer histogram | 队列长尾、跨 worker 传输慢、输入长度长尾 |
| PD misroute | engine_type、prefill/decode worker loads、route artifacts | 请求发到错误 worker 或 label 丢失 |
| decode starvation | decode transfer queue、running requests、prefill inflight queue | prefill 侧堆积、KV transfer 慢、decode worker 不足 |

## 5. KV Transfer

- [ ] transfer latency、speed、size 是否同时记录。
- [ ] bootstrap failed、transfer failed、prefill retries 是否增加。
- [ ] KV transfer tokens 是否和 prompt tokens、cached prefix tokens 对得上。
- [ ] transfer queue 增长时，prefill queue 和 decode running 是否同步变化。
- [ ] 网络、connector、rank 和 worker label 是否足够定位。

## 6. 对比 Run

结论建议写成下面格式：

```md
现象：
- TTFT p95:
- ITL p95:
- queue wait p95:
- KV transfer latency p95:

比较对象：
- baseline run:
- current run:
- 只改变的变量:

证据：
- 指标 1:
- 指标 2:
- artifact:
- 源码路径:

判断：
- 主要瓶颈在 queue / prefill / decode / KV transfer / cache / scrape 配置

处理：
- 参数、路由、资源配比或代码改动:

风险：
- 对 TTFT、ITL、吞吐、显存、label cardinality 的影响:
```

## 7. 结束条件

- [ ] 指标异常能由一个最小命令复现。
- [ ] baseline 和 current run 的命令、配置、workload 可比较。
- [ ] 能指出状态在源码哪里生成、哪里写入 metrics、哪里导出。
- [ ] 结论已写入 `serving_metrics_template.md`。
- [ ] 下一步动作只改一个变量。
