# 源码带读：L28 SGLang PD Disaggregation

这份带读按“patch reference -> MiniInfra -> 真实 SGLang”的顺序走。目标是抓住四个问题：prefix cache 命中在哪里产生，route 如何记账，KV transfer 在哪里成为边界，complete 如何释放 load。

## 0. 源码地图

```text
labs/l27_sglang_pd_cache/patch/reference/disagg_service.py
  -> labs/l27_sglang_pd_cache/patch/tests/test_patch.py
  -> labs/l27_sglang_pd_cache/scripts/run_pd_drill.py

mini_infra/sglang/srt/mem_cache/radix_cache.py
mini_infra/sglang/srt/managers/scheduler.py
mini_infra/sglang/srt/managers/disagg_service.py

github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py
github_repo/sglang/python/sglang/srt/managers/schedule_batch.py
github_repo/sglang/python/sglang/srt/managers/scheduler.py
github_repo/sglang/python/sglang/srt/managers/disagg_service.py
```

Patch 负责最小控制面，MiniInfra 负责可读主线，真实 SGLang 负责展示生产复杂度。

## 1. Patch reference：先读最小不变量

文件：[patch/reference/disagg_service.py](patch/reference/disagg_service.py)

先看第 8-32 行，确认 `Worker`、`KVTransfer` 和 `DisaggRoute` 的字段。这里的字段就是后面 metrics 和 complete 的事实来源。

再看第 54-68 行。`_select_least_loaded` 用 load、active request 和 worker id 排序；`_worker_by_id` 在 complete 时找回 worker。

接着看第 70-107 行。`route_request` 先校验，再算 `prefill_tokens`，然后选择两侧 worker，更新 load，调用 `transfer_kv`，保存 route。

最后看第 109-172 行。`transfer_kv` 追加历史 handoff；`metrics` 聚合 active 和 history；`complete_request` 释放 active load。读完后要能复述：route 写入哪些状态，complete 删除哪些状态，transfer 为什么保留。

## 2. Patch tests：七个合同逐个对齐

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

按测试顺序读：

- 第 22-33 行：单请求 route 后，route 字段、KVTransfer 和 worker_loads 都要对。
- 第 36-46 行：第二个请求应选择另一组 least-loaded worker。
- 第 49-56 行：空 request、非正 prompt、越界 cached prefix 都要失败。
- 第 59-82 行：metrics 要同时聚合 transfer、prefill、cache、active 和 worker load。
- 第 85-90 行：`transfers_for_request` 保持插入顺序。
- 第 93-100 行：complete 后 active load 归零，unknown complete no-op。
- 第 103-113 行：空状态也要返回完整 metrics shape。

测试覆盖的是状态合同，不覆盖真实网络传输、KV page、超时重试或 router 发现服务。

## 3. Drill：看函数组合后的证据链

文件：[scripts/run_pd_drill.py](scripts/run_pd_drill.py)

先看第 48-58 行。`_generate_workload` 按配置生成 prompt token 数和 cached prefix token 数。cache off/on 的差异来自 `cache_hit_rate` 和 `cached_prefix_ratio_when_hit`。

再看第 97-133 行。`run_pass` 创建服务，逐请求 route，写入 route metrics，随后 complete 所有请求并返回 complete 前后快照。这里能看到 run artifact 如何证明 route、transfer 和 worker drain。

最后看第 138-168 行。脚本计算 prefill saving，检查每个 request 的 transfer 数，检查 worker load 是否归零，并写入 `pd_drill.json`。真实 benchmark 也要保留类似证据。

## 4. MiniInfra：从 prefix cache 到 PD service

文件：[mini_infra/sglang/srt/mem_cache/radix_cache.py](../../mini_infra/sglang/srt/mem_cache/radix_cache.py)

重点看第 60-72 行和第 83-95 行。`match_prefix` 返回最长命中的前缀长度，`cache_finished_req` 把完成请求插回缓存。它解释了 `cached_prefix_tokens` 的来源。

文件：[mini_infra/sglang/srt/managers/scheduler.py](../../mini_infra/sglang/srt/managers/scheduler.py)

重点看第 30-50 行。scheduler 从 waiting 取请求，查询 prefix cache，把 `matched_prefix_tokens` 写入 prefill 记录，然后追加一个输出 token 并缓存完成请求。

文件：[mini_infra/sglang/srt/managers/disagg_service.py](../../mini_infra/sglang/srt/managers/disagg_service.py)

重点看第 68-108 行、第 125-153 行和第 160-169 行。这里和 patch reference 同构：route、metrics 和 complete 构成最小 PD 控制面。

## 5. 真实 SGLang RadixCache 与 Req prefix match

文件：[radix_cache.py](../../github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py)

先看第 71-120 行。`RadixKey` 不只是 token ids，还包含 `extra_key` 和 bigram 视图。真实 cache 命中要考虑 namespace 和 page 对齐。

再看第 398-466 行。`match_prefix` 会处理空 key、page alignment、radix tree 查找，并返回 device indices 和 last node。

最后看第 488-523 行。`cache_finished_req` 从 request 取 committed KV、构造 `RadixKey` 和 KV indices，插入 cache，并释放重复 KV。

文件：[schedule_batch.py](../../github_repo/sglang/python/sglang/srt/managers/schedule_batch.py)

重点看第 971-1039 行。Req 初始化下一轮输入时，会根据 `fill_ids` 计算可查的 token ids，调用 tree cache match，并把 `prefix_indices`、last node、host hit 和 `cache_protected_len` 写回 request。

## 6. 真实 SGLang PD queues 与 bootstrap service

文件：[scheduler.py](../../github_repo/sglang/python/sglang/srt/managers/scheduler.py)

先看第 1133-1238 行。`init_disaggregation` 根据 disaggregation mode 创建 decode transfer queue、decode prealloc queue、prefill bootstrap queue 和 prefill inflight queue。

再看第 2144-2166 行。普通模式进入 waiting queue；PREFILL 模式进入 bootstrap queue；DECODE 模式进入 decode prealloc queue。这个分支就是真实系统的 route 边界之一。

最后看第 3178-3193 行。idle 判断必须同时检查 waiting queue、prefill inflight/bootstrap queue、decode prealloc/transfer queue。PD 模式下，只看主 waiting queue 会误判系统空闲。

文件：[disagg_service.py](../../github_repo/sglang/python/sglang/srt/managers/disagg_service.py)

重点看第 14-44 行。真实 SGLang 会在 prefill mode 下启动 KV bootstrap server，并根据 transfer backend 选择对应 KV class。L28 patch 不实现这些后端，只保留每次 route 必须有 handoff 的不变量。

## 读完后的自检问题

1. `cached_prefix_tokens` 在 MiniInfra 和真实 SGLang 中分别来自哪里？
2. `prefill_tokens` 和 `kv_transfer_tokens` 为什么可以不同？
3. route 时更新了哪些 worker 状态，complete 时要撤销哪些状态？
4. complete 后为什么不能清空 `self.transfers`？
5. 真实 SGLang 的 PREFILL 和 DECODE mode 分别使用哪些 queue？
6. CPU drill 的 saving、transfer/request 和 drained worker loads 分别验证哪个不变量？
