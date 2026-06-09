# L22 · SGLang Serving Core: RadixCache Prefix Reuse

L22 接在 vLLM scheduler 之后，讨论 SGLang serving 的核心差异：当大量请求共享 system prompt、few-shot examples、工具 schema、RAG 模板或多轮对话历史时，RadixCache 怎样用 token 前缀索引已经算过的 KV，减少重复 prefill。

本讲的 patch 只实现一个教学版 trie。讲授重点是 prefix cache 的命中边界、namespace 隔离、LRU 叶子驱逐、和 scheduler/benchmark 中能看到的证据。真实 SGLang 会把这些机制接到 KV page、reference count、page alignment 和 cache-aware scheduling。

## 学习路线

建议按下面顺序走，先把 serving 问题讲清，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L22 在推理服务主线中的位置。
2. 读 [lecture.md](lecture.md)：理解 repeated-prefix workload、tokenized prefix、RadixCache trie、驱逐和真实 SGLang 的生产复杂度。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、真实 SGLang、benchmark 的顺序带着问题看源码。
4. 跑 notebook：[n09_prefix_cache.ipynb](../../notebooks/n09_prefix_cache.ipynb)。
5. 做 quiz：确认 prefix hit/miss、PagedAttention 区别、LRU 边界和 validation-only 报告边界。
6. 做 patch：实现最小 `RadixCache`。
7. 跑 repeated-prefix smoke：观察本地是否是 validation-only，或真实 server 是否返回 TTFT/cache 指标。
8. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)，沉淀一次 prefix-cache 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Serving data plane / cache-aware scheduling |
| 它解决什么问题 | 多请求共享长前缀时，怎样避免重复 prefill 同一段 KV |
| 它连接哪些指标 | TTFT、prefill tokens、cache hit rate、cache nodes、KV usage、validation-only status |
| 它连接哪些源码 | patch `RadixCache`、MiniInfra SGLang scheduler、真实 SGLang `BasePrefixCache` / `RadixCache` / schedule policy |
| lab 检验什么 | token 前缀匹配、新增节点计数、LRU 叶子驱逐和共享前缀保护 |

## 你会学到什么

- 为什么 prefix cache 按 token id 前缀命中，不按字符串或语义命中。
- RadixCache 与 PagedAttention 的分工：一个做内容索引，一个做 KV 显存分页。
- `match_prefix`、`insert`、`evict` 的输入、状态变化、输出和边界。
- namespace / `extra_key` 为什么能隔离不同 adapter、cache version 或 retrieval context。
- 真实 SGLang 为什么要处理 page alignment、KV indices、lock ref、host cache 和 cache-aware scheduling。
- repeated-prefix benchmark 如何固定 workload，避免把 validation-only 当作真实性能结论。

## Patch 闭环

```bash
cat labs/l21_sglang_serving_core/patch/task.md
$EDITOR labs/l21_sglang_serving_core/patch/starter/radix_cache.py
make patch-test M=l21_sglang_serving_core
```

patch 通过后跑一次本地 serving 路径验证：

```bash
python labs/l21_sglang_serving_core/scripts/run_server.py --config configs/4090_debug.yaml --run-id l22_server_validation
python labs/l21_sglang_serving_core/scripts/bench_repeated_prefix.py --run-id l22_prefix_validation
```

如果本地没有 SGLang server，benchmark 会写出 `validation_only`，这只能证明命令、配置和 artifact 路径有效，不能证明真实 TTFT 改善。

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_empty_cache_match_zero` | 空 cache 不命中，节点数为 0 |
| `test_insert_then_match_full` | 插入后完整 token 序列命中 |
| `test_partial_prefix_match` | query 可以命中已缓存序列的短前缀 |
| `test_repeated_insert_no_double_count` | 重复插入不重复增加节点 |
| `test_total_tokens_correct` | 共享前缀只计一份节点 |
| `test_evict_removes_lru` | 最近访问过的序列不先被删 |
| `test_evict_preserves_shared_prefix` | 删除叶子时保留仍被其它路径依赖的共享前缀 |

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 cache miss、TTFT 高、validation-only 和 metrics 缺失 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速回忆 patch、MiniInfra 和真实 SGLang 的源码主路径 |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 跑 repeated-prefix benchmark 后记录 workload、指标和结论边界 |

## 进入下一关

`make patch-test`、本地 server validation 和 repeated-prefix smoke 都通过后，进入 [L23 FlashAttention v2](../l22_flash_attn_v2_bench/README.md)。
