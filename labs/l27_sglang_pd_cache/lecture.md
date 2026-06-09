# L28：SGLang PD Disaggregation · Prefix Cache、KV Transfer 与 Worker Load

Serving 系统把 prefill 和 decode 拆开后，一个请求不再只是在同一条 engine loop 里前进。它会先经过路由，进入 prefill worker，复用 prefix cache 能复用的部分，再把 KV 交给 decode worker。任何一段状态记错，都会表现成 TTFT 变长、decode 等 KV、worker 负载失衡或指标看起来“省了 prefill”但实际没有可复查证据。L28 用一个最小 `DisaggregationService` 把这条链写成可测试代码。

## 1. 本讲目标

- 解释 prefill/decode 分离为什么需要显式 route 和 KV handoff。
- 区分 `prompt_tokens`、`cached_prefix_tokens`、`prefill_tokens` 和 `kv_transfer_tokens`。
- 实现 least-loaded worker 选择、KVTransfer 记录、metrics 聚合和幂等 complete。
- 用 patch tests 与 PD drill 验证 route、transfer、cache saving 和 worker drain。
- 对照 SGLang RadixCache、Req prefix match、scheduler disagg queue 和 bootstrap service。

## 2. 问题背景：PD 分离把一个请求拆成跨 worker 流水线

Prefill 处理完整 prompt，通常更像大块计算；decode 逐 token 生成，持续读取历史 KV，更容易受到显存带宽、batch 组成和 KV 访问影响。PD disaggregation 把两段放到不同 worker 或资源池，让长 prompt 和长输出可以被不同策略处理。

拆开以后，系统多了三个必须观测的边界。

第一是 route。请求要进入哪个 prefill worker、哪个 decode worker，不能只看单侧负载。prefill 侧轻但 decode 侧满，用户仍然会等。

第二是 prefix cache。重复前缀命中后，prefill worker 需要新算的 token 数减少，但 decode worker 仍要拥有覆盖完整上下文的 KV。

第三是 KV transfer。prefill 完成后，decode 侧要拿到可用 KV 才能开始。transfer 丢失、重复或变慢，都会变成新的尾部延迟来源。

## 3. Patch 数据模型

L28 patch 只建模控制面和记账面，不启动真实 SGLang server，也不分配 GPU KV page。它保留五个对象。

`Worker` 表示一个 prefill 或 decode worker，包含 `worker_id`、`role`、`load_tokens` 和 `active_requests`。

`KVTransfer` 表示一次 prefill 到 decode 的 KV 交接，包含 request id、prefill worker、decode worker 和 token count。

`DisaggRoute` 表示一次 active route，记录 prompt tokens、cached prefix tokens、prefill tokens 和 KV transfer tokens。

`self.routes` 保存当前活跃请求，用于 metrics 和 complete。

`self.transfers` 保存历史 transfer，用于审计和 debug。complete 不应清空它。

这个模型的关键不变量是：active route 表示当前占用，transfer list 表示历史交接。两者用途不同。

## 4. Prefix Cache 的影响范围

Prefix cache 复用重复 prompt 前缀的 KV。MiniInfra 的 RadixCache 用 token id 序列做 key，`match_prefix` 返回最长命中的前缀长度，`cache_finished_req` 把完成请求插入缓存。真实 SGLang 会进一步处理 `extra_key` namespace、page 对齐、tree node split、host/device 命中和 eviction。

在本讲整数模型里，cache match 的结果被压缩成 `cached_prefix_tokens`。公式是：

```text
prefill_tokens = prompt_token_count - cached_prefix_tokens
kv_transfer_tokens = prompt_token_count
```

如果 prompt 有 128 个 token，命中 32 个 prefix token，prefill worker 只新增计算 96 个 token。decode worker 仍需要完整 prompt 对应的 KV 视图，所以 transfer token 数仍记为 128。

这个区分很重要。若把 `kv_transfer_tokens` 也减成 96，drill 会低估 decode 侧 handoff 成本；若把 `prefill_tokens` 写成 128，cache saving 会消失。

## 5. Route 的正确顺序

`route_request` 按六步执行。

1. 校验 `request_id` 非空。
2. 检查同一个 request 不能重复 route。
3. 校验 `prompt_token_count > 0`。
4. 校验 `cached_prefix_tokens` 在 `[0, prompt_token_count]`。
5. 计算 `prefill_tokens`，选择 prefill 和 decode 两侧 least-loaded worker。
6. 更新 worker load，调用 `transfer_kv`，保存并返回 `DisaggRoute`。

Least-loaded 的排序 key 是 `(load_tokens, active_requests, worker_id)`。第一项控制主要负载，第二项处理 token 数相同但 active request 数不同的情况，第三项保证平局时 deterministic。

顺序不能随意调换。先更新 load 再发现输入非法，会污染 worker 状态；只 route 不 transfer，会让 decode 缺少 handoff 证据；保存 route 前不记录 complete 所需字段，会导致后续释放错误。

## 6. Transfer、Metrics 与 Complete

`transfer_kv` 的合同是校验 request id、worker id 和正的 token count，然后 append 一个 `KVTransfer`。`transfers_for_request` 只过滤历史列表，并保持插入顺序。这个小约束能帮助 debug 多次 handoff 或重复 transfer。

`metrics()` 聚合两类信息。

当前活跃状态：

- `prefill_tokens`
- `cached_prefix_tokens`
- `active_requests`
- `worker_loads`

历史 transfer 状态：

- `kv_transfers`
- `tokens_transferred`
- `tokens_by_prefill_worker`
- `tokens_by_decode_worker`

`complete_request` 负责释放 active 状态。它从 `self.routes` 弹出 route；如果 request 不存在，直接返回。存在 route 时，prefill worker 减 `prefill_tokens`，decode worker 减 `kv_transfer_tokens`，两侧 active request 都减 1，并用 `max(0, ...)` 防止负数。

complete 后，`active_requests` 可以归零，但 `kv_transfers` 仍然大于零。这说明请求已经结束，历史 handoff 仍可复查。

## 7. Patch Tests 与 Drill

Patch tests 覆盖七个合同。

| 测试 | 验证 |
|---|---|
| route records worker and transfer | route 字段、KVTransfer 和 worker load |
| selects least-loaded workers | 多 worker 下根据负载分配 |
| validates required fields | 输入边界 |
| metrics aggregate tokens | cache、prefill、transfer 和 worker load 聚合 |
| transfers_for_request filters | 按 request 保持插入顺序 |
| complete releases load | active load 归零，unknown request no-op |
| empty metrics are zero | 空服务状态 |

`scripts/run_pd_drill.py` 把这些函数组合成一个小 workload。它先生成 cache off 请求，再生成 cache on 请求；每个请求 route 后写入 `metrics.jsonl`，然后 complete；最后检查三件事：cache on 的 prefill saving 是否达到阈值，每个 request 是否恰好一条 transfer，complete 后 worker loads 是否归零。

CPU smoke 的默认条件是 20 个请求、prompt tokens 在 128 到 1024 之间、cache hit rate 为 0.5、命中时缓存 60% prefix，要求 prefill saving 至少 0.20。这些数值只用于验证机制形状，真实性能要用 SGLang benchmark。

## 8. 真实 SGLang 对照

真实 SGLang 的 RadixCache 会用 `RadixKey` 表达 token ids、`extra_key` 和 bigram 视图。`match_prefix` 会先做 page alignment，再查 radix tree；命中结果会返回 device indices 和 last node。`cache_finished_req` 会把完成请求的 token ids 和 KV indices 插回树，并释放重复或未对齐的 KV。

Req 的 `init_next_round_input` 会用 tree cache 做 prefix match，把 `prefix_indices`、`last_node`、`last_host_node`、`host_hit_length` 和 `cache_protected_len` 写回请求状态。课堂的 `cached_prefix_tokens` 可以理解为这些 prefix match 结果的整数化摘要。

Scheduler 的 disagg 初始化会根据 mode 创建 decode transfer queue、decode prealloc queue、prefill bootstrap queue 和 prefill inflight queue。请求进入队列时，普通模式进入 waiting queue；PREFILL 模式进入 bootstrap queue；DECODE 模式进入 decode prealloc queue。idle 判断也要看这些 queue，说明 PD 模式下“有没有活”不能只看 waiting queue。

`start_disagg_service` 展示了生产系统会为 prefill mode 启动 bootstrap server，并根据 transfer backend 创建对应 KV class。真实传输涉及后端、端口、rank、metadata buffer 和故障处理；L28 patch 只保留“每次 route 必须有显式 handoff”这个核心不变量。

## 9. Debug 路线

Misroute 时，先看 route 记录和 worker load，再看 cached prefix 是否被用到。若 cache hit 很高但 prefill tokens 没下降，检查 `prefill_tokens = prompt - cached_prefix_tokens`。若某个 worker load 长期偏高，检查 least-loaded 选择和 complete 是否释放。

Transfer lost 时，先看每个 request 的 transfer 数。prefill 完成但 decode 不开始，常见原因是 route 没调用 `transfer_kv`、transfer token 数为 0、或 decode transfer queue 堆积。

Decode starvation 时，看 decode worker load、decode transfer queue、running requests 和 ITL/TPOT。prefill 侧 saving 达标，不代表 decode 侧没有瓶颈。

Metrics skew 时，把 `prefill_tokens + cached_prefix_tokens` 和 prompt token 总量对齐。若 prefill tokens 大于 prompt tokens，通常是重复 route、重复计数或 complete 后 active route 没释放。

## 10. 小结

L28 的核心是把 PD 分离里的跨 worker 状态写清楚。Prefix cache 改变 prefill 工作量，KV transfer 定义 prefill 到 decode 的交接边界，worker load 决定后续 route，complete 保证状态可回收。写对这些不变量后，学生才能把真实 SGLang 的 RadixCache、disagg queue 和 metrics 放到同一条证据链里。
