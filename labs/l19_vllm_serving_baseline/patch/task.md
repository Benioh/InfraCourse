# L20 Patch · Typical-p Sampling Filter

实现一个 logits filter：

```python
def typical_p_filter(
    logits: torch.Tensor,
    typical_p: float = 0.9,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    ...
```

输入 `logits` 形状为 `(batch, vocab)`，输出形状必须相同。

## 行为合同

1. `typical_p >= 1.0` 时原样返回 logits。
2. 用 `softmax` 得到概率，用 `-log(p)` 得到 surprisal。
3. 每行 entropy 是 `(probs * surprisal).sum(dim=-1)`。
4. 按 `abs(surprisal - entropy)` 升序排序。
5. 对排序后的概率做累计和，超过 `typical_p` 的位置删除。
6. 每行至少保留 1 个 token。
7. 删除 mask 必须 scatter 回原 vocab 顺序。
8. 被删 token 用 `filter_value` 替换，保留 token 的原 logit 不变。

## 验证

```bash
IMPL=reference make patch-test M=l19_vllm_serving_baseline
```
