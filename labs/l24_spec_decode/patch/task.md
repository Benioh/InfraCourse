# L25 Patch · Speculative Decoding Greedy Verify

## 你要交付什么

实现 **greedy spec decode 的 verify 步骤**——target 模型一次 forward 验证 k 个 draft token：

```python
def greedy_verify(
    draft_tokens: List[int],          # k 个 draft 模型生成的候选 token
    target_logits: torch.Tensor,      # (k+1, vocab) target 在每个位置的 logits
) -> tuple[List[int], int]:
    """返回 (accepted_tokens, num_accepted_drafts).

    accepted_tokens 长度 = num_accepted_drafts + 1
        （+1 是 target 给的"bonus token"，无论 draft 接受多少都会有）
    num_accepted_drafts ∈ [0, k]
    """
```

**禁止** 用 `transformers.generation.GenerationMixin.assisted_generation`。
**允许** torch.argmax 等基础算子。

补丁规模目标：20–40 行。

## 算法

Greedy verify：target 在每个位置都 argmax，与 draft 比对。

```
for i in 0..k-1:
    target_argmax_i = argmax(target_logits[i])
    if target_argmax_i == draft_tokens[i]:
        accept draft_tokens[i]
        continue
    else:
        # 这一步 target 不认 draft；用 target 自己的 argmax 当 "bonus token"
        return accepted + [target_argmax_i], num_accepted
# 全 k 个都接受：position k 的 argmax 是免费 bonus
return accepted + [argmax(target_logits[k])], k
```

为什么这样数学等价于无 spec decode？
- Greedy 决策只依赖最后 token 的 logits
- target 一次 forward 已经知道所有位置的 logits
- 接受 draft 的本质是"target 在这个位置也会选这个"
- 不接受 draft 的本质是"用 target 的 argmax 接管，从此处重启"

## 不变量

1. 全 k 个 draft 都对：返回 k+1 个 token，num_accepted = k。
2. 第一个就错：返回 1 个 token（target 的 argmax），num_accepted = 0。
3. 第 i 个错（0-indexed）：返回 i+1 个 token（前 i 个 draft + target 在位置 i 的 argmax），num_accepted = i。
4. k=0（没 draft）：返回 1 个 token（target_logits[0] 的 argmax），num_accepted = 0。
5. 所有返回的 token 都来自 vocab（int 类型）。

## 怎么验证

```bash
make patch-test M=l24_spec_decode
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_all_drafts_accepted` | k=4 全对 → 5 token, num_accepted=4 |
| `test_first_mismatch_at_position_2` | 前 2 对，第 3 不对 → 3 token, num_accepted=2 |
| `test_zero_drafts_one_bonus` | k=0 → 1 token, num_accepted=0 |
| `test_all_mismatch` | 全错 → 1 token (target argmax), num_accepted=0 |
| `test_output_dtypes` | 返回 List[int] + int |

## 写完之后你能做什么

- 解释 speculative decoding 的收益来自 target verify 次数下降，以及 draft cost 和 acceptance 的权衡。
- 用 `accepted_tokens` 与 `num_accepted_drafts` 判断 KV 有效长度和回滚边界。
- 看懂 n-gram、draft model、Medusa/EAGLE 这几类 proposer 的入口差异。
