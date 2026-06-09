# L27 · SGLang PD Observability：Prometheus Metrics Exporter

<!-- LECTURE_FIRST_START -->

本讲讲推理服务的可观测性，重点放在 SGLang prefill/decode 分离场景。PD 分离把首 token 前的 prefill、后续 decode 和 KV transfer 拆到不同资源池，排障时不能只看一个 p99 或一行日志。L27 用一个 CPU-safe 的 Prometheus exposition exporter patch 建立 gauge、counter、hit-rate 和 label 稳定性的最小合同，再接回 MiniInfra PD 模拟、SGLang `/v1/loads`、scheduler metrics 和 KV transfer 指标。

## 学习路线

建议按下面顺序走，先把观测对象讲清楚，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L27 在 Serving 观测和 PD 分离路径中的位置。
2. 读 [lecture.md](lecture.md)：从 p99 排障、Prometheus 文本、指标类型讲到 SGLang PD 指标。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、PD lab、SGLang metrics collector 和 scheduler 主路径阅读。
4. 做 quiz：确认 gauge/counter/hit-rate、label 顺序、PD 指标和 artifact 边界。
5. 做 patch：实现最小 exporter 并通过测试。
6. 跑 PD lab：生成 unified 与 PD 配置的模拟指标和 report。
7. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)，沉淀一次观测复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Serving systems / observability / SGLang PD |
| 核心风险 | p99 退化无法定位阶段、metric 类型误用、label 不稳定、series 膨胀、PD queue/KV transfer 不可见 |
| 关键机制 | Prometheus text exposition、gauge、counter、hit-rate、label 排序、PD prefill/decode/KV transfer 指标 |
| 源码落点 | `patch/reference/metrics_exporter.py`、`scripts/run_pd_lab.py`、SGLang `/v1/loads`、metrics collector、scheduler metrics mixin |
| lab 检验 | gauge 覆盖写、counter 累加、hit-rate 计算、零命中安全、label 字母序 |

## 学完后能做什么

- 写出最小 Prometheus exposition 文本 exporter，并解释 gauge、counter、hit-rate 的语义。
- 说明 label 顺序为什么必须稳定，以及生产中为什么不能加 request 级高基数 label。
- 把 TTFT、ITL、queue wait、KV transfer latency 和 cache hit-rate 放到 prefill/decode 分离流水线里解释。
- 对照 SGLang metrics collector 找到 running、queue、token usage、cache hit、spec decode 和 PD 指标。
- 用 command/config/metrics/report/artifact 证据链排查 metrics missing、PD misroute 和 decode starvation。

## Patch 闭环

```bash
cat labs/l26_sglang_pd_observability/patch/task.md
$EDITOR labs/l26_sglang_pd_observability/patch/starter/metrics_exporter.py
make patch-test M=l26_sglang_pd_observability
```

PD lab：

```bash
python labs/l26_sglang_pd_observability/scripts/run_pd_lab.py --run-id l27_pd
```

CPU lab 用模拟公式展示 unified、2P6D、4P4D 的指标形状。真实结论必须替换为 SGLang 服务的 `/metrics` 或 `/v1/loads?format=prometheus` 输出，并保留服务启动命令、模型、workload 和硬件条件。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 metrics missing、label 不稳定、PD misroute、decode starvation 和 KV transfer 慢 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、PD lab、SGLang metrics collector 和 scheduler metrics 主路径 |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 记录一次 SGLang/PD 观测复盘的配置、指标、判断和下一步 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L27 后进入 L28 SGLang PD KV Cache / Disaggregation Service。下一讲会继续使用这里建立的观测字段，去解释 KV transfer、prefix cache 和 PD 路由的真实状态。
