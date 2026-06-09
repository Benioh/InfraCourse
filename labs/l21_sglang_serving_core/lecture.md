# L22：SGLang RadixCache Prefix Reuse

L21 讲了 vLLM scheduler 怎样在 KV block 容量下接纳请求。L22 继续留在推理服务，但问题换成 repeated-prefix workload：大量请求带着同一段 system prompt、few-shot examples、工具 schema、RAG 模板或多轮对话历史。如果每个请求都重新 prefill 这段共享前缀，服务端会重复计算同一批 KV，TTFT 会被这部分重复工作拖住。

RadixCache 的目标是把已经算过的前缀 KV 复用起来。它按 token id 序列建索引：请求到来时查最长已缓存前缀，只对剩余 suffix 做 prefill；请求完成或 chunk 结束后，把新的可复用 KV 插回 cache。课堂 patch 用一个 dict-of-dict trie 训练这个合同。真实 SGLang 会把 trie 节点关联到 KV page indices、lock reference、page alignment、namespace 和调度策略。

## 1. 本讲学完要能回答什么

学完后，你应该能回答这些问题：

1. prefix cache 复用的对象是文本、token id，还是 attention KV？
2. 两个 prompt 人眼看起来相同，为什么仍可能 cache miss？
3. `match_prefix`、`insert`、`evict` 分别读写哪些状态？
4. RadixCache 和 PagedAttention 的分工是什么？
5. 为什么 LRU evict 要从叶子开始删？
6. `extra_key` 或 namespace 解决什么隔离问题？
7. 真实 SGLang 的 `MatchResult.device_indices` 比课堂版 matched length 多表达了什么？
8. repeated-prefix benchmark 里哪些变量必须固定，哪些结果只能算 validation-only？

本讲属于 serving data plane / cache-aware scheduling。它不讲 HTTP 协议，也不深入 attention kernel。我们关注的是 prefill 前缀复用怎样进入 scheduler 和 KV cache 管理。

## 2. 从 repeated-prefix workload 进入

考虑一个生产服务：

- 每个请求都有同一个 3k token system prompt。
- RAG 模板里有固定的指令、格式约束和 citation schema。
- 工具调用时，每轮都带同一批 tool definitions。
- 多轮 chat 的第 N 轮 prompt 包含前 N-1 轮对话历史。

这些请求的共享前缀可能很长。如果没有 prefix cache，首 token 前要为每个请求重新跑 prefill，生成同样的 KV。GPU 看起来在忙，requests/sec 也许没有立刻崩掉，但用户等待 first token 的时间会被重复 prefill 放大。

Prefix cache 的收益来自少算一段 prompt。它不会让 decode 消失，也不会改变模型权重。命中后，系统仍要处理 suffix、排队、采样、输出和网络返回。排查时要把 TTFT 拆成 queue、tokenize、prefill、prefix cache lookup、suffix prefill、first decode 和 output processing。

## 3. Prefix hit 的判定对象

RadixCache 以 token id 序列为 key。两个 prompt 的字符串相同，不保证 token id 相同。常见破坏命中的来源包括：

- chat template 版本变化；
- BOS/EOS、role marker、空格、换行变化；
- RAG 文档排序不同；
- 工具 schema 字段顺序不同；
- tokenizer 版本变化；
- 请求被放进不同 namespace 或 `extra_key`。

因此 cache miss 排查第一步应比较 token id 的最长公共前缀，而不是停在字符串对比。只有 token id 前缀一致，RadixCache 才能复用对应 KV。

`extra_key` 解决的是隔离问题。不同 LoRA adapter、cache version、retrieval context 或租户即使 token id 一样，也可能不应该共享同一段 KV。把它们放进不同 namespace，可以阻止错误复用。

## 4. RadixCache 和 PagedAttention 的分工

KV cache 是 attention 的 K/V 张量缓存。PagedAttention 关注这些 KV 在显存里怎样切成 page、怎样按需分配、怎样减少碎片。RadixCache 关注哪些 token 前缀已经计算过 KV，怎样用 token 序列找到对应的 KV page。

可以这样分层：

```text
RadixCache
  key: token id prefix + namespace
  value: KV indices / pages for that prefix

PagedAttention / KV manager
  resource: GPU KV pages
  job: allocate, reference, free, compact through page tables
```

生产系统通常需要两层一起工作。RadixCache 命中后要返回 KV indices；KV manager 需要知道这些 page 被引用，不能被错误释放。驱逐时也不能只删 trie 节点，还要处理 KV page 的 reference count 和实际释放。

## 5. 课堂版 trie 的状态

patch 里的 `_Node` 很小：

```python
class _Node:
    children: Dict[int, _Node]
    last_access: int
    parent: _Node | None
    token_from_parent: int | None
```

`children` 表示 token id 到子节点的边。`last_access` 用于 LRU。`parent` 和 `token_from_parent` 用于从叶子回删父节点的 child。`RadixCache` 维护 root、`max_tokens`、单调递增 `_access_counter` 和 `_total`，其中 `_total` 是非 root 节点数，不是序列条数。

这个实现没有 path compression。真实 radix tree 会把一段连续 token 存成一个 edge 或 node key，课堂版让每个 token 对应一个节点，目的是把行为合同暴露清楚。

## 6. `match_prefix` 怎样工作

`match_prefix(token_ids)` 从 root 出发，按 token id 一步一步往下走。能走到第几个 token，就返回多长的匹配前缀。遇到不存在的 child 立即停止。

它还要刷新访问时间。原因是 LRU 驱逐依赖 `last_access`。一个前缀被频繁命中，说明它可能还会继续被使用；如果只在 insert 时更新访问时间，热前缀会被当成冷数据删掉。

输入、状态和输出可以这样看：

| 项 | 内容 |
|---|---|
| 输入 | `token_ids: List[int]` |
| 中间状态 | 当前 node、matched 长度、访问计数 |
| 状态变化 | 被访问节点的 `last_access` 更新 |
| 输出 | 已缓存最长前缀长度 |
| 边界 | 空 cache 返回 0；部分匹配返回已走过长度 |

## 7. `insert` 怎样避免重复计数

`insert(token_ids)` 也从 root 走。如果 child 已存在，就沿着已有节点继续走；如果 child 不存在，就新建节点，并增加 `_total` 和 `new_tokens`。返回值是新增节点数。

例子：

```text
insert([1, 2, 3]) -> new 3, total 3
insert([1, 2, 3]) -> new 0, total 3
insert([1, 2, 4]) -> new 1, total 4
```

这个返回值对应真实服务中的重复 prefill 逻辑：已经在 cache 里的前缀不需要重新计算 KV；只有 suffix 或新分支会增加缓存占用。

插入后如果 `_total > max_tokens`，教学版调用 `evict(self._total - self.max_tokens)`。真实系统会按 token/page 数、lock ref、priority、host cache 和策略决定能释放多少。

## 8. 为什么 evict 从叶子开始

RadixCache 的共享来自前缀。假设 cache 中有 `[1, 2, 3]` 和 `[1, 2, 4]`。节点 `[1, 2]` 是共享前缀。如果为了释放空间直接删掉 `[1]` 或 `[2]`，两条路径都会坏掉。

教学版驱逐策略是：

1. 收集所有非 root 叶子节点。
2. 选 `last_access` 最小的叶子。
3. 从父节点的 `children` 删除这个叶子。
4. 更新 `_total`。
5. 如果还要删，重新收集叶子。

这个过程一次只删除可安全移除的末端节点。父节点如果删完 child 后变成叶子，下一轮才可能被删除。这样可以保护仍被其它请求依赖的共享前缀。

## 9. MiniInfra 怎样把 cache 接到 scheduler

MiniInfra 的 `RadixCache` 使用更接近真实 SGLang 的 API：`RadixKey` 包含 token 序列和可选 `extra_key`；`MatchPrefixParams`、`InsertParams` 和 `MatchResult` 包装输入输出。

MiniInfra scheduler 的流程是：

```text
waiting request
  -> prefix_cache.match_prefix(MatchPrefixParams(RadixKey(prompt_tokens)))
  -> record matched_prefix_tokens
  -> simulate one decode token
  -> prefix_cache.cache_finished_req(req)
  -> return prefill/decode/cache_hits
```

这说明 prefix cache 不是孤立的数据结构。它必须在 prefill 前查询，在请求完成或 chunk 结束后写回。只写 trie 单测而没有接到 scheduler，服务指标不会变化。

## 10. 真实 SGLang 多了哪些生产复杂度

真实 SGLang 的 `BasePrefixCache` 统一了不同 cache 类型的参数和结果。`MatchResult.device_indices` 返回的是已命中前缀对应的 KV cache indices，而不是单纯长度。`last_device_node` 和 `last_host_node` 让调度器和 cache 管理器知道命中落在哪个树节点。

真实 `TreeNode` 包含：

- `key`：路径上这一段 token key；
- `value`：KV indices；
- `lock_ref`：节点是否被运行中请求保护；
- `last_access_time` / `creation_time` / `hit_count`：驱逐策略和指标；
- `host_value` / `hash_value`：HiCache、host cache 和 page hash；
- `priority`：priority-aware eviction。

真实 `match_prefix` 会按 page size 对齐 key，必要时 split node，让命中边界精确落在树结构上。`cache_finished_req` 会把请求的 input ids 和 output ids 对应 KV indices 插入树，释放重复 KV 和未对齐 tail。`evict` 会从可驱逐叶子构建 heap，根据配置策略释放 KV value。

Scheduler 也会利用 prefix match。schedule policy 会提前给 waiting queue 计算 prefix matches，并可按 longest prefix 或 DFS weight 排序。这样 cache 命中不只是减少 prefill 计算，也会影响 admission 顺序。

## 11. Benchmark 和 validation-only 边界

patch-test 只能证明 trie 行为正确。它不能证明真实 server 的 TTFT 下降。要观察服务效果，需要 repeated-prefix benchmark：

- 固定 shared prefix；
- 固定 suffix 构造方式；
- 固定 model、port、max_new_tokens、temperature；
- 记录 server 是否真实可用；
- 记录 TTFT、cache hit rate、requests/sec、metrics 路径；
- 明确本次只改了哪个变量。

`bench_sglang.py` 在 OpenAI-compatible `/v1/chat/completions` 上发请求。server 不可达时，它写出 `status: validation_only`。这种产物可以证明命令、配置、报告路径正常，但不能当成真实性能结果。报告必须区分“本地配置验证”和“真实 server benchmark”。

## 12. Debug 顺序

遇到 cache miss 或 TTFT 高，可以按这个顺序排查：

1. 比较两个请求的 token id 最长公共前缀。
2. 检查 chat template、BOS/EOS、工具 schema、RAG 文档排序和 tokenizer 版本。
3. 检查 namespace / `extra_key` 是否把请求隔离到不同 cache key。
4. 看 server 是否启用 prefix cache，是否有 cache hit metrics。
5. 看 TTFT 拆分：queue、tokenize、prefill、first decode、output。
6. 看 capacity 和 evict：cache 是否太小，热前缀是否被 LRU 删掉。
7. 对齐 artifact：`command.sh`、`config.resolved.yaml`、`serve.log`、`metrics.jsonl` 是否描述同一次运行。

不要把 prefix cache 当成所有 TTFT 问题的解释。命中正常时，queue、suffix prefill、decode batch、网络和输出后处理仍可能是主要瓶颈。

## 13. 本讲小结

L22 的核心链路是：tokenized prompt 决定 prefix key，RadixCache 返回最长命中前缀，scheduler 只 prefill suffix，请求完成后把可复用 KV 插回 cache，容量不足时从冷叶子开始驱逐。

课堂 patch 训练的是这条链路的最小数据结构合同。真实 SGLang 把同一合同扩展到 KV page、reference lock、namespace、page alignment、host cache 和 cache-aware scheduling。做完 patch 后，继续用 repeated-prefix benchmark 和产物边界验证它在服务路径中的意义。
