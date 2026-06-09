# L28 Debug Checklist：SGLang PD Disaggregation

这张 checklist 面向 router misroute、prefix cache 未生效、KV transfer lost、decode starvation、complete double 和 metrics skew。

## 1. 固定现场

- [ ] 记录命令、配置文件、git commit、Python 环境、模型、硬件和 run id。
- [ ] 保存 `command.sh`、`config.resolved.yaml`、`metrics.jsonl`、`artifacts/pd_drill.json` 和 `report.md`。
- [ ] 记录 prefill workers、decode workers、request 数、prompt token 分布、cache hit rate 和 cached prefix ratio。
- [ ] 明确本次运行使用 starter、reference，还是学生实现。
- [ ] 每次只改一个变量，例如 cache hit rate、worker 数、route 策略或 complete 行为。

## 2. Route

- [ ] `request_id` 是否为空或重复。
- [ ] `prompt_token_count` 是否为正。
- [ ] `cached_prefix_tokens` 是否在 `[0, prompt_token_count]`。
- [ ] prefill worker 是否按当前 `load_tokens` 和 `active_requests` 选择。
- [ ] decode worker 是否也参与 least-loaded 选择。
- [ ] route 后是否保存 `DisaggRoute`。

## 3. Prefix Cache

- [ ] `prefill_tokens = prompt_token_count - cached_prefix_tokens`。
- [ ] cache on 相比 cache off 是否降低 `prefill_tokens`。
- [ ] `cached_prefix_tokens` 是否被重复计入。
- [ ] 真实 SGLang 中是否使用相同 namespace、LoRA/cache salt 和 tokenizer。
- [ ] RadixCache 是否因 page alignment、eviction 或 namespace 导致命中变少。

## 4. KV Transfer

- [ ] 每个 request 是否恰好一条 `KVTransfer`。
- [ ] `kv_transfer_tokens` 是否等于 prompt token 数。
- [ ] `tokens_transferred` 是否等于所有 transfer token 之和。
- [ ] `transfers_for_request` 是否能按插入顺序找回记录。
- [ ] 真实系统中 decode transfer queue 是否堆积。

## 5. Complete 与 Worker Load

- [ ] complete 是否从 `routes` 删除 request。
- [ ] prefill worker 是否减去 `prefill_tokens`。
- [ ] decode worker 是否减去 `kv_transfer_tokens`。
- [ ] active request 计数是否归零。
- [ ] unknown request complete 是否 no-op。
- [ ] complete 后是否保留 transfer history。

## 6. 常见故障定位

| 现象 | 先看什么 | 可能方向 |
|---|---|---|
| cache on 没有 saving | cached_prefix_tokens、prefill_tokens、RadixCache namespace | cache match 失败或公式错误 |
| prefill 完成但 decode 不开始 | transfer/request、decode transfer queue、bootstrap port | transfer 丢失或 decode 等 KV |
| decode worker 长期满 | decode worker_loads、active_requests、ITL/TPOT | decode starvation |
| worker load 只涨不降 | complete_request、routes、active_requests | complete 漏释放 |
| metrics 中 prefill_tokens 大于 prompt 总量 | 重复 route、重复计数、cached prefix 越界 | metrics skew |

## 7. 结束条件

- [ ] cache off/on 的 prefill saving 能被 `pd_drill.json` 复查。
- [ ] 每个 request 的 transfer 数可核对。
- [ ] complete 后 worker loads 归零。
- [ ] 真实 SGLang 对照时，能指出 RadixCache、Req prefix match、disagg queues 和 bootstrap service 的源码位置。
- [ ] 结论已写入 `serving_metrics_template.md`。
