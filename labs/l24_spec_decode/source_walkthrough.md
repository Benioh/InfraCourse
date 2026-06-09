# 源码带读：L25 Speculative Decoding Greedy Verify

这份带读按“patch 合同 -> MiniInfra 现象 -> vLLM 指标 -> SGLang KV 边界”的顺序组织。读源码时不要从真实框架的分支入口开始追，先把一轮 verify 的输入、状态和输出闭合。

## 0. 源码地图

```text
labs/l24_spec_decode/patch/starter/spec_decode.py
labs/l24_spec_decode/patch/reference/spec_decode.py
labs/l24_spec_decode/patch/tests/test_patch.py
mini_infra/vllm/spec_decode/draft_runner.py
mini_infra/vllm/spec_decode/acceptance_tracker.py
mini_infra/vllm/spec_decode/ngram.py
github_repo/vllm/vllm/config/speculative.py
github_repo/vllm/vllm/v1/spec_decode/metrics.py
github_repo/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py
github_repo/sglang/python/sglang/srt/managers/schedule_batch.py
labs/l24_spec_decode/scripts/bench_concurrency.py
```

## 1. Patch starter：确认学生要补的合同

文件：`labs/l24_spec_decode/patch/starter/spec_decode.py`

重点看：`greedy_verify` 的签名、`k`、`target_argmax` 和 TODO 区域。

建议阅读顺序：

- L20-L29：函数输入、输出和返回语义。读完后要能说出 `accepted_tokens` 和 `num_accepted` 的区别。
- L30-L32：先计算 `k` 和长度为 `k+1` 的 target argmax。读完后要能解释 target logits 第一维为什么多 1。
- L34-L45：TODO 注释已经列出连续接受、mismatch 接管和全接受 bonus。读完后手算一个中间 mismatch 例子。

可以先跳过：typing import 和文件头说明。

## 2. Patch reference：看最小正确实现

文件：`labs/l24_spec_decode/patch/reference/spec_decode.py`

重点看：一个循环如何覆盖四个边界。

建议阅读顺序：

- L10-L15：参考实现和 starter 使用同一个签名，并一次性取出 `target_argmax`。
- L16-L23：循环只接受连续前缀；mismatch 分支立刻返回 target argmax；全接受分支追加 `target_argmax[k]`。

读完后的结论：reference 没有模型逻辑，也没有采样逻辑，它只定义 greedy verify 的行为合同。

## 3. Patch tests：用人工 logits 固定边界

文件：`labs/l24_spec_decode/patch/tests/test_patch.py`

重点看：测试如何把 logits 行的 argmax 固定成可手算序列。

建议阅读顺序：

- L17-L19：`IMPL` 决定测试 starter 或 reference。
- L22-L28：`_make_logits` 把每一行 argmax 固定成 `argmax_seq[i]`。
- L31-L37：全接受时返回 k 个 draft 加一个 bonus。
- L40-L47：位置 2 mismatch 时返回前两个 draft 加 target 位置 2 的 argmax。
- L50-L55：k=0 时仍然返回一个 target token。
- L58-L64：首错时只返回 target 位置 0 的 argmax。
- L67-L74：返回值必须是 Python `list[int]` 和 `int`。

可以先跳过：pytest import 和 Path 处理。

## 4. MiniInfra draft runner：把 verify 放进一轮服务指标

文件：`mini_infra/vllm/spec_decode/draft_runner.py`

重点看：候选生成、连续接受和 summary 字段。

建议阅读顺序：

- L29-L40：`verify_candidates` 返回连续接受数量，第一处不一致后停止。
- L43-L57：`draft_step` 生成 n-gram 或模拟 draft 候选，并输出 `kv_rollback_tokens`。
- L60-L72：`spec_decode_summary` 构造 prompt、future、tracker 和 step。
- L73-L86：summary 写出 acceptance、speedup、TTFT、ITL p50/p99、draft latency、draft memory 和 domain。

读完后的结论：MiniInfra 没有真实模型 forward，但它保留了 spec decode 的指标形状，适合用来训练报告字段。

## 5. MiniInfra acceptance tracker：从单轮返回到窗口指标

文件：`mini_infra/vllm/spec_decode/acceptance_tracker.py`

重点看：accepted/proposed 如何进入 rate 和 speedup。

建议阅读顺序：

- L22-L30：tracker 维护固定窗口事件，每个事件是 `(accepted, proposed)`。
- L32-L42：`rate()` 汇总窗口里的 acceptance，`expected_speedup()` 用 draft cost 估算收益。
- L45-L56：`acceptance_summary` 用一组示例事件生成可展示的指标。

可以先跳过：文件头里的教学说明，主路径在类方法里。

## 6. MiniInfra n-gram proposer：最低成本候选器

文件：`mini_infra/vllm/spec_decode/ngram.py`

重点看：prompt lookup 怎样产生候选。

建议阅读顺序：

- L18-L27：`ngram_candidates` 接收 prompt tokens、suffix 和候选长度，suffix 为空时直接返回空列表。
- L28-L32：从 prompt 中找 suffix 出现位置，并取其后 token 作为候选。
- L35-L46：summary 展示 n-gram mode、suffix、候选、acceptance 估计和零 draft GPU memory。

读完后的结论：n-gram proposer 成本低，但依赖 prompt 内重复，不适合所有 workload。

## 7. vLLM speculative config：真实系统的入口参数

文件：`github_repo/vllm/vllm/config/speculative.py`

重点看：method、K、draft model、prompt lookup 和高级 drafting 控制。

建议阅读顺序：

- L73-L85：`SpeculativeConfig` 保存 `num_speculative_tokens`、`model` 和 `method`。
- L91-L95：ngram 需要 prompt lookup 参数，draft TP 只能取受支持的并行度。
- L100-L113：draft model 可能有自己的 quantization、MoE backend 和 attention backend。
- L126-L136：高级控制包含 padding 和 local argmax reduction。
- L138-L154：prompt lookup、tree 和 parallel drafting 影响候选形态。

可以先跳过：post-init 生成的完整 model/parallel config 和长尾兼容字段。

## 8. vLLM metrics：把接受长度变成可观测信号

文件：`github_repo/vllm/vllm/v1/spec_decode/metrics.py`

重点看：accepted tokens、draft tokens、per-position acceptance 和 Prometheus counter。

建议阅读顺序：

- L18-L30：`SpecDecodingStats` 保存 draft 数、draft token 数、accepted token 数和按位置的接受计数。
- L32-L45：`new()` 初始化 per-position 数组，`observe_draft()` 更新一轮 draft 的统计。
- L74-L86：日志聚合 draft 和 accepted throughput。
- L88-L99：日志计算 draft acceptance rate、mean acceptance length 和 per-position acceptance。
- L121-L140：Prometheus 注释给出 acceptance rate、mean acceptance length 和 per-position acceptance 的查询公式。
- L200-L215：`observe()` 把每轮统计写进 counter。

读完后的结论：生产排查要看 accepted/proposed、accepted length 和 per-position 分布，不能只看总 tokens/s。

## 9. SGLang scheduler：accept_lens 与 KV 提交

文件：`github_repo/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py`

重点看：`accept_lens` 如何转成 accepted draft 与 KV committed length。

建议阅读顺序：

- L351-L361：从结果中取 `accept_lens`，并用 `sum(accept_lens) - len(batch.reqs)` 得到 accepted draft 总数。
- L363-L372：每个请求按 `accept_lens[i] - 1` 推进 `kv_committed_len`，并切出本轮预测 token。
- L374-L378：per-request accepted draft 数进入统计和 histogram。

读完后的结论：`accept_lens` 包含 bonus，所以 accepted draft 数要减 1。

## 10. SGLang request state：overallocated KV 和 histogram

文件：`github_repo/sglang/python/sglang/srt/managers/schedule_batch.py`

重点看：request 如何记录 spec 计数和回收范围。

建议阅读顺序：

- L842-L851：request 记录 verify 次数、accepted draft 总数和 acceptance histogram。
- L937-L947：spec decode 可能过量分配 KV，释放时返回 committed 与 allocated 范围。
- L949-L959：按 accepted draft token 数扩展并更新 histogram。

读完后的结论：KV 和指标都依赖 accepted draft count，bonus 计错会污染 request 级统计。

## 11. Bench 脚本：用同一指标形状跑对比

文件：`labs/l24_spec_decode/scripts/bench_concurrency.py`

重点看：不同 batch 下 n-gram 和 draft 两种模式如何输出同一组字段。

建议阅读顺序：

- L12-L20：脚本读取 servers 和 batches，并对每个 batch 生成 n-gram 与 draft 两行 summary。

可以先跳过：JSON 格式化细节。

## 读完后的自检问题

1. `greedy_verify` 在 mismatch 位置返回哪个 token，为什么不能返回 draft token？
2. `target_logits` 为什么需要 `k+1` 行？
3. vLLM 的 acceptance rate、mean acceptance length 和 per-position acceptance 分别由哪些计数字段计算？
4. SGLang 里为什么 `accept_lens - 1` 才是 accepted draft 数？
5. 如果 bench 中 n-gram p50 改善但 p99 变差，你会先查哪三个字段？
