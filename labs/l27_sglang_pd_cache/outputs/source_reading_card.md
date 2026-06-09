# L28 Source Reading Card：SGLang PD Disaggregation

这张卡用于快速回忆 L28 的源码主路径。读源码时先抓状态不变量，再看真实系统扩展。

## 1. Patch 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `labs/l27_sglang_pd_cache/patch/reference/disagg_service.py` | `route_request`、`transfer_kv`、`metrics`、`complete_request` | 最小 PD 控制面由 route、handoff、metrics 和 release 组成 |
| `labs/l27_sglang_pd_cache/patch/tests/test_patch.py` | 七个 pytest | 验证 worker 选择、token 公式、transfer 记录、metrics 聚合和 complete 幂等 |
| `labs/l27_sglang_pd_cache/scripts/run_pd_drill.py` | workload、run_pass、acceptance | cache off/on 的证据链来自 metrics、snapshot、transfer/request 和 drained loads |

## 2. MiniInfra 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `mini_infra/sglang/srt/mem_cache/radix_cache.py` | `match_prefix`、`cache_finished_req` | cached prefix tokens 来自最长前缀命中 |
| `mini_infra/sglang/srt/managers/scheduler.py` | `run_batch` | scheduler 在 prefill 前查询 prefix cache，并把 matched prefix 写入记录 |
| `mini_infra/sglang/srt/managers/disagg_service.py` | `route_request`、`metrics`、`complete_request` | MiniInfra 与 patch reference 同构 |

## 3. SGLang 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py` | `RadixKey`、`match_prefix`、`cache_finished_req` | 真实 prefix cache 处理 namespace、page alignment、tree node 和 KV indices |
| `github_repo/sglang/python/sglang/srt/managers/schedule_batch.py` | `Req.init_next_round_input` | prefix match 结果进入 `prefix_indices`、last node 和 protected length |
| `github_repo/sglang/python/sglang/srt/managers/scheduler.py` | `init_disaggregation`、`_add_request_to_queue`、idle 判断 | PD 模式创建 prefill/decode 专用 queue，队列状态影响运行与健康判断 |
| `github_repo/sglang/python/sglang/srt/managers/disagg_service.py` | `start_disagg_service` | prefill mode 会启动 KV bootstrap service |

## 4. 读源码时的提问顺序

1. 这个字段表示 prompt 总量、cached prefix、prefill 工作量，还是 transfer 成本？
2. 这段代码更新 active state，还是保留 history？
3. 请求进入的是 waiting queue、prefill bootstrap queue、decode prealloc queue，还是 decode transfer queue？
4. complete 后哪些状态应释放，哪些状态应保留？
5. 这个源码片段能解释哪类故障：misroute、transfer lost、starvation、load leak，还是 metrics skew？

## 5. 本讲最小不变量

- `prefill_tokens = prompt_token_count - cached_prefix_tokens`。
- `kv_transfer_tokens = prompt_token_count`。
- route 必须同时选择 prefill 和 decode worker。
- route 必须记录一条 KV transfer。
- metrics 必须拆出 prefill、cached、transferred 和 worker load。
- complete 必须释放 active load，并对 unknown request no-op。
- transfer history 是审计证据，complete 后仍保留。
