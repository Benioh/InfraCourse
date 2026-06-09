# L25：Speculative Decoding Greedy Verify

Speculative decoding 处理的是 decode 阶段“一步一个 target forward”的成本问题。draft proposer 先给出 k 个候选 token，target 把这些候选接到上下文后一次 forward，得到 k+1 个位置的 logits。greedy verify 从左到右比较 target argmax 与 draft token，只接受连续前缀；第一处 mismatch 由 target argmax 接管；如果 k 个 draft 都被接受，就追加第 k 行 logits 的 argmax 作为 bonus token。

## 1. 本讲目标

- 解释 spec decode 减少的是 target decode step 次数，代价是 draft 计算、候选浪费、额外显存和调度复杂度。
- 写出 greedy verify 的输入、输出、不变量和四个边界：全接受、首错、中间错、k=0。
- 区分返回 token 数、accepted draft 数、mean acceptance length 和 KV committed length。
- 用 acceptance rate、draft cost、K、并发、p99 和显存判断 spec decode 是否值得打开。
- 读懂 patch、MiniInfra、vLLM metrics 和 SGLang KV 提交路径中的主状态。

## 2. 问题背景：decode 慢在 target step 节奏

自回归 LLM 每生成一个 token，都要根据当前上下文做一次 decode step。KV cache 已经缓存历史 token 的 Key/Value，避免重复投影历史上下文，但 target 模型仍然要每步执行一次层堆叠、attention 和输出头。对于大模型服务，decode 阶段的瓶颈常落在大量短小 decode step 的 batch 组织、KV 显存和长尾延迟上，单次矩阵乘吞吐只能解释一部分现象。

Speculative decoding 的思路是让一个更便宜的 proposer 先猜多个 token。proposer 可以是 prompt n-gram lookup、独立 draft model，也可以是 Medusa/EAGLE 这类接在 target 表征上的候选器。target 不信任这些候选，所以仍然要 verify；收益来自一次 target forward 同时验证多个候选位置。如果一轮平均产出多个有效 token，target step 成本就被摊薄。

这节课只讲 greedy verify。greedy decoding 的选择规则是每个位置取 target logits 的 argmax。只要 draft 在第 i 个位置等于 target 在同一位置的 argmax，这个 draft token 就和 target 逐步 greedy decode 的选择一致。第一处不一致意味着后续上下文已经走到不同路径，后面的 draft token 不能继续使用。

## 3. 一轮 verify 的输入和输出

patch 函数是：

```python
def greedy_verify(
    draft_tokens: list[int],
    target_logits: torch.Tensor,
) -> tuple[list[int], int]:
    ...
```

`draft_tokens` 的长度是 k。`target_logits` 的 shape 是 `(k + 1, vocab)`。前 k 行用于检查 draft 第 0 到第 k-1 个位置；第 k 行用于全接受后的 bonus token。函数返回两个值：

- `accepted_tokens`：真正追加到输出序列的 token。它包含被接受的 draft 前缀，并且总是多一个 target bonus token。
- `num_accepted_drafts`：被 target 接受的 draft token 数，不包含 bonus。

四个边界要手算清楚：

| 场景 | 输入关系 | 返回 token | `num_accepted_drafts` |
|---|---|---|---|
| 全接受 | k 个 draft 都等于 target argmax | k 个 draft + `target_argmax[k]` | k |
| 首错 | `draft[0] != target_argmax[0]` | `[target_argmax[0]]` | 0 |
| 中间错 | 前 i 个匹配，第 i 个 mismatch | 前 i 个 draft + `target_argmax[i]` | i |
| k=0 | 没有 draft | `[target_argmax[0]]` | 0 |

例子：`draft=[3,7,11,23]`，`target_argmax=[3,7,99,0,0]`。前两个 draft 被接受，位置 2 mismatch，所以返回 `accepted_tokens=[3,7,99]`，`num_accepted_drafts=2`。位置 3 的 draft `23` 不能再看，因为上下文已经被 `99` 接管。

## 4. Greedy Verify 的算法

实现只有三个状态：`k`、`target_argmax` 和 `accepted`。

```python
k = len(draft_tokens)
target_argmax = target_logits.argmax(dim=-1).tolist()
accepted = []

for i in range(k):
    if target_argmax[i] == draft_tokens[i]:
        accepted.append(draft_tokens[i])
    else:
        return accepted + [target_argmax[i]], len(accepted)

accepted.append(target_argmax[k])
return accepted, k
```

这段逻辑要守住三个不变量。第一，只接受连续前缀，不能跳过 mismatch 再接受后面的 token。第二，mismatch 分支返回 target 的 argmax，不能返回 draft 的错误 token。第三，全接受分支必须追加 `target_argmax[k]`，否则每轮会少产出一个已经由 target 算出的 token。

patch-test 用人工 logits 固定每行 argmax。测试不会依赖真实模型，因此学生可以把注意力放在行为合同上。真实框架中的采样式 rejection、树形 proposer、batch 内不同 K、分布式 logits 通信都比 patch 更复杂，但它们都要回到“target 接受了多少连续候选”这个语义。

## 5. Acceptance 和 Speedup 账本

Acceptance rate 记作 α，含义是 draft token 被 target 接受的比例。无投机时，平均一个输出 token 需要一次 target step。投机时，一轮消耗 draft 成本和一次 target verify，平均产出 `1 + α * k` 个 token。一个常用近似是：

```text
time_per_token_spec = (k * T_draft + T_target) / (1 + α * k)
speedup = T_target / time_per_token_spec
```

MiniInfra 里为了教学还给了一个更粗的窗口估计：`speedup ≈ 1 / ((1 - rate) + draft_cost_ratio)`。这类公式不能直接当生产结论，但它能帮学生判断方向：draft 越贵，需要的 acceptance 越高；K 越大，一处早期 mismatch 浪费的候选越多；prompt 分布变化、高 temperature、tokenizer 不一致、draft 与 target 训练域差异都会压低 acceptance。

报告 spec decode 时至少要记录这些字段：

| 字段 | 解释 |
|---|---|
| baseline ITL / tokens/s | 无 spec 的比较对象 |
| acceptance rate | accepted draft tokens / proposed draft tokens |
| mean acceptance length | 通常包含 bonus，约等于 `1 + accepted_drafts_per_round` |
| draft latency / draft memory | proposer 的成本账 |
| ITL p50/p99 | 平均收益和尾部风险 |
| K / method / sampling params | 复现实验的必要条件 |

只写“开 spec 后更快”没有可复查价值。要说明比较对象、模型、硬件、并发、prompt 分布、K、proposer 类型、采样参数和质量指标。

## 6. Proposer 家族：ngram、draft model、Medusa/EAGLE

n-gram proposer 不加载模型。它在 prompt 中查找与当前 suffix 相同的片段，把片段后的 token 当候选。它的额外 GPU 显存接近 0，适合重复度高的 prompt 或模板化 workload；缺点是短 prompt、开放式生成和域外输入很容易给不出候选或给错候选。

独立 draft model 使用小模型先生成候选。它通常比 target 快，但要加载权重、维护自己的 KV，甚至要考虑 draft tensor parallel、tokenizer/vocab 对齐、quantization 和 attention backend。draft 与 target 越接近，acceptance 越高；draft 越大，成本越高。

Medusa/EAGLE 这类方法把候选器放到 target 的 hidden state 或专门 head 上，目的是降低 draft 成本并提高候选质量。它们部署时仍然要进入 target verify，只是 proposer 的成本结构和配置路径不同。读 vLLM `SpeculativeConfig` 时，要把 `method`、`model`、`num_speculative_tokens`、prompt lookup、draft TP、tree 或 parallel drafting 分开看。

## 7. KV 和 Scheduler 边界

verify forward 会临时计算候选位置的 KV，但最终有效序列只包含被接受的 draft 前缀和一个 target bonus。若接受 m 个 draft，有效输出长度增加 `m + 1`。被拒绝位置之后的候选 KV 不属于后续上下文，实际系统会通过 sequence length、page table、slot mapping 或 overallocated KV 回收来处理。

SGLang 的 `accept_lens` 包含 bonus token。它把 `num_accepted_drafts` 计算为 `accept_lens - 1`，并用 accepted draft 数更新 per-request histogram。这个字段关系非常容易错：bonus 应该推进输出和 KV committed length，但不应该进入 accepted draft token 计数。vLLM metrics 也沿着同一思路聚合 `num_draft_tokens`、`num_accepted_tokens` 和 per-position acceptance。

资源账要同时看 target 和 proposer。独立 draft model 会带来第二份权重、第二份 KV、额外 forward 和临时候选 buffer。高并发下，spec decode 可能降低 p50 ITL，但 reject burst、混合 workload、draft 排队和 KV 回滚会拉高 p99。上线判断必须同时看吞吐、延迟分位数、显存峰值和输出质量。

## 8. 源码落点

本讲先读 patch，再读 MiniInfra，最后读真实框架：

- `labs/l24_spec_decode/patch/starter/spec_decode.py`：学生补 `greedy_verify` 的最小合同。
- `labs/l24_spec_decode/patch/reference/spec_decode.py`：连续前缀、mismatch 接管和全接受 bonus 的参考实现。
- `labs/l24_spec_decode/patch/tests/test_patch.py`：五个 CPU 测试覆盖全接受、首错、中间错、k=0 和返回类型。
- `mini_infra/vllm/spec_decode/draft_runner.py`：用 n-gram 或模拟 draft 生成候选，输出 acceptance、speedup、ITL 和 draft 成本。
- `mini_infra/vllm/spec_decode/acceptance_tracker.py`：用窗口统计 accepted/proposed，并给出教学版 speedup 估计。
- `mini_infra/vllm/spec_decode/ngram.py`：展示 prompt lookup proposer 的输入和边界。
- `github_repo/vllm/vllm/config/speculative.py`：真实 vLLM 的 speculative method、K、draft TP、prompt lookup 和 drafting 策略配置。
- `github_repo/vllm/vllm/v1/spec_decode/metrics.py`：真实 vLLM 的 acceptance、mean acceptance length、per-position acceptance 和 Prometheus 计数器。
- `github_repo/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py`：SGLang 从 `accept_lens` 计算 accepted draft，并推进 KV committed length。
- `github_repo/sglang/python/sglang/srt/managers/schedule_batch.py`：SGLang 记录 per-request spec 计数、overallocated KV 和 acceptance histogram。

## 9. Debug 路线

低接受率先查 proposer 和采样条件。顺序是 tokenizer/vocab 是否对齐、prompt domain 是否偏移、temperature/top_p 是否提高随机性、K 是否过大、draft 是否过弱或过强。如果 `acceptance rate` 低于 draft 成本能承受的范围，优先换 proposer、降 K、按 domain 分流，或对低收益 workload 关闭 spec。

Draft OOM 先算显存账。target 权重、target KV、draft 权重、draft KV、candidate buffer、batch/K 放大项和临时 logits 都要列出来。只降 batch 可能会掩盖问题，真正的边界可能是 draft model 太大、KV block 留量不足或候选链过长。

p99 抖动要把 ITL 按 accept count、method、prompt domain 和 batch size 分桶。若 accept=0 占比高，说明大量轮次只拿到 bonus token，还额外付了 draft 成本。若同一个服务里 in-domain 与 out-of-domain 混在一起，可以考虑分流、动态 K 或滑动窗口开关。

## Lab 验收边界

本讲 patch 命令：

```bash
make patch-test M=l24_spec_decode
```

patch 验收的是 greedy spec decode 的 verify 步骤：target 一次 forward 算出每个位置的 argmax，与 draft 从左到右比对；首个 mismatch 用 target argmax 接管；全接受时追加第 k 行 logits 的 argmax。

测试通过后，还要能把 patch 的返回值映射到 metrics：`num_accepted_drafts` 进入 acceptance，`len(accepted_tokens)` 影响每轮输出 token 数，mismatch 位置决定候选浪费和 KV 回滚范围。

## 10. 小结

L25 的核心是一轮 verify 的语义，而非某个 proposer 名字。draft 可以来自 n-gram、小模型或专门 head；target verify 必须决定连续接受多少 draft，并产出一个 target token 接管下一步。只要这个合同错了，acceptance、mean acceptance length、KV committed length 和 p99 分析都会偏移。把这个合同写对，再去评估 draft 成本、K、并发和真实 workload，才是可上线的 spec decode 判断方式。
