# L25 Source Reading Card

## 主路径

1. `labs/l24_spec_decode/patch/starter/spec_decode.py`：学生补齐 `greedy_verify`，重点是 k、target argmax、连续接受、mismatch 接管和 bonus。
2. `labs/l24_spec_decode/patch/reference/spec_decode.py`：最小正确实现，用一个循环覆盖四个边界。
3. `labs/l24_spec_decode/patch/tests/test_patch.py`：人工 logits 固定 argmax，测试全接受、中间错、k=0、首错和返回类型。
4. `mini_infra/vllm/spec_decode/draft_runner.py`：从候选生成到 acceptance、speedup、TTFT、ITL 和 draft 成本。
5. `mini_infra/vllm/spec_decode/acceptance_tracker.py`：滑动窗口 accepted/proposed 和教学版 speedup 估计。
6. `mini_infra/vllm/spec_decode/ngram.py`：prompt lookup proposer，额外显存低但依赖重复。
7. `github_repo/vllm/vllm/config/speculative.py`：真实 speculative method、K、draft TP、prompt lookup 和 drafting 策略。
8. `github_repo/vllm/vllm/v1/spec_decode/metrics.py`：acceptance rate、mean acceptance length、per-position acceptance 和 Prometheus counter。
9. `github_repo/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py`：`accept_lens` 转 accepted draft 与 KV committed length。
10. `github_repo/sglang/python/sglang/srt/managers/schedule_batch.py`：request 级 spec 计数、overallocated KV 和 histogram。

## 必须记住的字段关系

- `target_logits.shape[0] = len(draft_tokens) + 1`。
- `accepted_tokens = accepted_draft_prefix + [target_bonus]`。
- `num_accepted_drafts` 不包含 bonus。
- 全接受时返回长度是 `k + 1`，accepted draft 数是 k。
- mismatch 在位置 i 时，返回长度是 `i + 1`，accepted draft 数是 i。
- SGLang `accept_lens` 包含 bonus，accepted draft 数要减 1。

## 读源码时先跳过

- 树形 proposer 的完整采样逻辑。
- 分布式 logits 通信和硬件专用 kernel。
- 与本讲 greedy 合同无关的请求生命周期细节。
- Prometheus registry 初始化和服务端暴露格式。

## 自检问题

1. Patch reference 为什么在 mismatch 分支返回 `target_argmax[i]`？
2. `mean acceptance length` 为什么通常比 accepted draft 数多 1？
3. n-gram proposer 和独立 draft model 的成本账差在哪里？
4. vLLM metrics 里哪些字段可以计算 acceptance rate？
5. SGLang 的 overallocated KV 和 spec decode 有什么关系？
