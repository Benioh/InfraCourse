# L22 Source Reading Card

## Patch 主线

| 文件 | 只看什么 | 结论 |
|---|---|---|
| `patch/starter/radix_cache.py` | `_Node`、`RadixCache`、三个 TODO | 学生需要维护 trie 节点、LRU 时间、节点计数和叶子删除 |
| `patch/reference/radix_cache.py` | `match_prefix`、`insert`、`evict`、`_collect_leaves` | reference 展示最长前缀、重复插入、容量驱逐和共享前缀保护 |
| `patch/tests/test_patch.py` | 7 个行为测试 | patch-test 验证 trie 合同，不验证真实 server 性能 |

## MiniInfra 主线

| 文件 | 只看什么 | 结论 |
|---|---|---|
| `mini_infra/sglang/srt/mem_cache/radix_cache.py` | `RadixKey`、`MatchPrefixParams`、`match_prefix`、`cache_finished_req` | MiniInfra 保留真实 SGLang 的结构化 API 和 namespace 形状 |
| `mini_infra/sglang/srt/managers/scheduler.py` | `Scheduler.run_batch` | scheduler 在 prefill 前 match，完成后写回 cache |
| `mini_infra/sglang/run_scheduler.py` | repeated-prefix smoke | 共享 prompt 的请求应在第二轮产生 cache hit |

## 真实 SGLang 主线

| 文件 | 只看什么 | 结论 |
|---|---|---|
| `github_repo/sglang/python/sglang/srt/mem_cache/base_prefix_cache.py` | `MatchPrefixParams`、`InsertParams`、`MatchResult` | 真实返回值包含 KV device indices 和命中节点 |
| `github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py` | `TreeNode`、`match_prefix`、`insert`、`cache_finished_req`、`evict` | 真实 cache 处理 page alignment、lock ref、KV indices、host cache 和 eviction strategy |
| `github_repo/sglang/python/sglang/srt/managers/schedule_policy.py` | `_compute_prefix_matches`、longest-prefix sort | prefix hit 会进入 waiting queue 的调度策略 |

## Benchmark 主线

| 文件 | 只看什么 | 结论 |
|---|---|---|
| `scripts/run_server.py` | server validation | 记录 SGLang/CUDA 可用性和启动命令 |
| `scripts/bench_sglang.py` | `call_server`、metrics row | server 不可达时写 `validation_only`，不能证明真实性能 |

## 自检

1. 我能说清 token id 前缀、KV indices 和 page 的关系。
2. 我能解释 `extra_key` 为什么会改变 cache key。
3. 我能指出 patch trie 和真实 SGLang radix tree 的同构点和差异。
4. 我能判断一个 repeated-prefix artifact 是 served 结果还是 validation-only。
