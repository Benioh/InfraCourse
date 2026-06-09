# 源码带读：L22 SGLang RadixCache

这份带读按“patch trie -> MiniInfra scheduler -> 真实 SGLang -> benchmark”的顺序走。先读 [lecture.md](lecture.md)，再打开源码。每一步只看主路径，跳过和本讲合同无关的框架分支。

## 0. 源码地图

```text
labs/l21_sglang_serving_core/patch/starter/radix_cache.py
  -> labs/l21_sglang_serving_core/patch/reference/radix_cache.py
  -> labs/l21_sglang_serving_core/patch/tests/test_patch.py

mini_infra/sglang/srt/mem_cache/radix_cache.py
  -> mini_infra/sglang/srt/managers/scheduler.py
  -> mini_infra/sglang/run_scheduler.py

github_repo/sglang/python/sglang/srt/mem_cache/base_prefix_cache.py
  -> github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py
  -> github_repo/sglang/python/sglang/srt/managers/schedule_policy.py

labs/l21_sglang_serving_core/scripts/run_server.py
  -> labs/l21_sglang_serving_core/scripts/bench_sglang.py
```

## 1. Patch Starter：先看学生要维护的状态

文件：[radix_cache.py](patch/starter/radix_cache.py)

先看 `_Node`：

```text
L18-L25
```

结论：节点保存 child map、LRU 时间、父节点和从父节点来的 token。`parent` 与 `token_from_parent` 是 evict 能删回父节点的关键。

再看 `RadixCache.__init__` 与 `total_tokens`：

```text
L28-L36
```

结论：root 不计入 `_total`，`_access_counter` 是单调时间源。`total_tokens()` 返回的是非 root 节点数。

然后看三个 TODO：

```text
L38-L51  match_prefix
L53-L69  insert
L71-L92  evict 和 _collect_leaves
```

读这三段时只问三个问题：输入是什么，哪些字段会变，返回值要表达什么。

## 2. Patch Reference：对照正确状态迁移

文件：[radix_cache.py](patch/reference/radix_cache.py)

看 `match_prefix`：

```text
L28-L39
```

它沿 token id 逐步走 trie。每命中一个节点，`matched` 增加，访问计数和节点 `last_access` 更新。遇到 miss 立即停止。

看 `insert`：

```text
L41-L54
```

已有 child 直接复用，缺失 child 才新建。`new_tokens` 是新增节点数，重复插入同一序列时返回 0。插入后如果超过容量，调用 `evict`。

看 `evict`：

```text
L56-L68
```

每轮重新收集叶子，选择 `last_access` 最小的叶子，从父节点删除对应 child，并减少 `_total`。

看 `_collect_leaves`：

```text
L70-L81
```

DFS 遍历整棵树，只返回非 root 且没有 children 的节点。这个限制保护共享前缀。

## 3. Patch Tests：看验收合同

文件：[test_patch.py](patch/tests/test_patch.py)

按下面顺序读：

```text
L21-L32  空 cache 与完整命中
L35-L51  部分前缀命中和重复插入
L54-L73  节点数与 LRU 驱逐
L76-L90  共享前缀保护
```

测试没有评价生成文本质量，也没有接真实 server。它只证明 trie 的候选前缀和容量管理合同正确。

## 4. MiniInfra RadixCache：看 SGLang-shaped API

文件：[radix_cache.py](../../mini_infra/sglang/srt/mem_cache/radix_cache.py)

先看 key 和参数：

```text
L7-L13    RadixKey
L16-L31   MatchPrefixParams / InsertParams
```

结论：MiniInfra 已经把 token ids 和 `extra_key` 放进结构化参数。真实系统用这个额外 key 做 namespace 隔离。

看 `match_prefix`：

```text
L60-L72
```

它沿 `(token, extra_key)` 走树，只在遇到 `node.value is not None` 时更新 best match。这表达了“树节点存在”和“这个前缀有可复用 KV value”之间的区别。

看 `insert` 和 `cache_finished_req`：

```text
L74-L81   insert
L83-L95   cache_finished_req
```

`insert` 把 value 写到路径末端。`cache_finished_req` 从 request 的 prompt tokens 构造 key，并写入一个教学版 `["kv"]` value。

## 5. MiniInfra Scheduler：看 prefix cache 接入点

文件：[scheduler.py](../../mini_infra/sglang/srt/managers/scheduler.py)

看 scheduler 初始化和入队：

```text
L19-L28
```

`Scheduler` 持有 `prefix_cache`、waiting 和 running。新请求先进入 waiting。

看 `run_batch`：

```text
L30-L50
```

关键顺序是：pop waiting、match prefix、记录 `matched_prefix_tokens`、模拟生成 token、`cache_finished_req` 写回 cache、加入 running。

## 6. MiniInfra Smoke：看 repeated-prefix 现象

文件：[run_scheduler.py](../../mini_infra/sglang/run_scheduler.py)

看主路径：

```text
L19-L25
```

前两个请求共享 `"system math tutor question"` 的一段前缀；第三个请求重复第一条 prompt。第二轮应该看到更长 prefix 命中。

如果带 `--run-id`，继续看：

```text
L32-L36
```

smoke 会写 command snapshot 和 `scheduler.json`，这给课后报告提供证据。

## 7. 真实 BasePrefixCache：看统一参数和返回值

文件：[base_prefix_cache.py](../../github_repo/sglang/python/sglang/srt/mem_cache/base_prefix_cache.py)

看参数对象：

```text
L35-L43   MatchPrefixParams
L46-L63   InsertParams
```

真实 cache 不只接 key，还要处理 Mamba、SWA、chunked insert 和 priority。

看结果对象：

```text
L123-L147
```

`device_indices` 是命中的 KV cache indices；`last_device_node` 和 `last_host_node` 是命中落点；`host_hit_length` 服务 HiCache。课堂版只返回长度，真实版要把执行层需要的 KV 位置一起返回。

## 8. 真实 RadixCache：看节点、match、insert、evict

文件：[radix_cache.py](../../github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py)

先看 `TreeNode`：

```text
L211-L233
```

除了 children、parent、key、value，它还有 `lock_ref`、访问时间、host cache、hash、priority。这些字段决定节点能否驱逐、如何观测和怎样跨 host/device 复用。

看初始化里的 eviction strategy：

```text
L321-L335
L342-L360
```

真实 SGLang 支持 LRU、LFU、FIFO、MRU、FILO、priority、SLRU。课堂 patch 固定 LRU。

看 `match_prefix`：

```text
L398-L410
L435-L466
```

`extra_key` 参与 namespace 隔离；key 会按 page size 对齐；返回 `MatchResult` 时带 device indices 和命中节点。

看 `insert` 与 `cache_finished_req`：

```text
L468-L486
L488-L530
```

insert 会做 bigram view、page alignment 和 priority。finished request 会把 input/output ids 对应的 KV indices 插入 tree，并释放重复 KV 和未对齐 tail。

看 `evict`：

```text
L608-L635
```

真实 evict 从可驱逐叶子建 heap，释放 KV value，删除叶子，并在父节点变成可驱逐叶子时继续加入 heap。

## 9. 真实 Schedule Policy：cache hit 进入调度策略

文件：[schedule_policy.py](../../github_repo/sglang/python/sglang/srt/managers/schedule_policy.py)

看 prefix match 预计算：

```text
L185-L203
L204-L214
```

等待队列中的每个请求都会先拿到 `prefix_indices` 和命中节点。

看 in-batch prefix caching：

```text
L216-L243
```

如果多个 waiting 请求共享同一段当前 cache 还没有的前缀，调度器可以临时降低其中一些请求优先级，让先运行的请求把前缀写进 cache。

看 longest-prefix 排序：

```text
L246-L256
```

这说明 prefix cache 会影响调度顺序。命中不是只影响 prefill 计算量。

## 10. Benchmark Scripts：看报告边界

文件：[run_server.py](scripts/run_server.py)

```text
L29-L42
L43-L64
```

`run_server.py` 记录 SGLang 是否可 import、CUDA 是否可用和推荐启动命令。它写的是 server validation，不等于真实 benchmark。

文件：[bench_sglang.py](scripts/bench_sglang.py)

```text
L28-L49   OpenAI-compatible 请求和失败路径
L52-L75   配置、run 目录和 workload 构造
L76-L104  served 与 validation-only 的 metrics 行
```

server 不可达时，`status` 是 `validation_only`。这条边界必须写进报告，不能把它当作真实 TTFT。

## 11. 读完后的自检

1. `match_prefix` 为什么要刷新 `last_access`？
2. 重复插入同一序列时，哪些节点会被复用？
3. LRU evict 为什么只删叶子？
4. MiniInfra 的 `matched_prefix_tokens` 对 prefill 有什么含义？
5. 真实 `MatchResult.device_indices` 给 worker 提供了什么？
6. `extra_key` 能隔离哪些不该共享的请求？
7. schedule policy 为什么要在 waiting queue 上预计算 prefix match？
8. 本地 artifact 是 validation-only 时，报告能支持哪些结论？
