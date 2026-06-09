# L28 · SGLang PD Disaggregation：Prefix Cache、KV Transfer 与 Worker Load

<!-- LECTURE_FIRST_START -->

本讲讲 SGLang prefill/decode 分离中的控制面。L27 已经建立了 queue、KV transfer 和 cache hit 的观测语言；L28 继续把这些字段落到一个最小 `DisaggregationService`：请求如何路由到 prefill/decode worker，prefix cache 命中如何减少 prefill 工作量，KV transfer 如何成为 decode 前的交接边界，worker load 如何在 complete 后释放。

## 学习路线

建议按下面顺序走，先理解 PD 流水线，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L28 在 Serving PD 分离、prefix cache 和 KV transfer 链路中的位置。
2. 读 [lecture.md](lecture.md)：从 prefill/decode 资源画像、prefix cache 命中、KV handoff 讲到 metrics 与 debug。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch reference、MiniInfra、SGLang RadixCache、scheduler 和 PD service 主路径阅读。
4. 跑 notebook：[n08_prefill_decode.ipynb](../../notebooks/n08_prefill_decode.ipynb)、[n09_prefix_cache.ipynb](../../notebooks/n09_prefix_cache.ipynb)。
5. 做 quiz：确认 route、prefix cache、KV transfer、metrics 和 complete 的边界。
6. 做 patch：实现最小 `DisaggregationService` 并通过测试。
7. 跑 drill：对比 cache off/on 的 prefill saving、transfer/request 和 worker drain。
8. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)，沉淀一次 PD 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Serving systems / SGLang / PD disaggregation |
| 核心风险 | router misroute、prefix cache 未被利用、KV transfer 丢失或重复、decode starvation、complete 后 load 泄漏 |
| 关键机制 | least-loaded route、cached prefix -> prefill tokens、prefill -> decode KV transfer、worker load metrics、幂等 complete |
| 源码落点 | patch reference、MiniInfra radix/scheduler/disagg、SGLang RadixCache、Req prefix match、scheduler disagg queues、PD bootstrap service |
| lab 检验 | route 校验、worker 选择、KVTransfer 记录、metrics 聚合、transfers 过滤、complete 释放、空状态 |

## 学完后能做什么

- 解释 PD 分离里 prefill worker、decode worker 和 KV transfer 的职责边界。
- 判断 prefix cache 命中减少的是 prefill 工作量，不能自动消除 decode 侧 KV handoff。
- 写出 least-loaded 路由、KV transfer 记录、metrics 聚合和幂等 complete 的最小实现。
- 对照 SGLang RadixCache 与 scheduler disagg queue 找到真实系统中的对应路径。
- 用 drill artifact 排查 misroute、transfer lost、complete double、decode starvation 和 metrics skew。

## Patch 闭环

```bash
cat labs/l27_sglang_pd_cache/patch/task.md
$EDITOR labs/l27_sglang_pd_cache/patch/starter/disagg_service.py
make patch-test M=l27_sglang_pd_cache
```

Drill：

```bash
IMPL=reference python labs/l27_sglang_pd_cache/scripts/run_pd_drill.py \
  --config configs/cpu_smoke.yaml \
  --run-id l28_pd
```

`cpu_smoke.yaml` 使用 2 个 prefill worker、2 个 decode worker、20 个合成请求。它先跑 cache off，再跑 cache on；验收 cache on 是否降低 prefill tokens、每个 request 是否恰好一条 transfer、complete 后 worker loads 是否归零。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 PD route、prefix cache、KV transfer、decode starvation 和 worker load 泄漏 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、MiniInfra、真实 SGLang RadixCache 与 scheduler 主路径 |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 记录一次 PD cache/transfer 复盘的配置、指标、判断和下一步 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L28 后进入 L29 verl RL Baseline。下一章会把服务侧证据链延伸到 RL rollout、reward、training loop 和评测闭环。
