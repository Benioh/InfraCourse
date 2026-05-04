# L04.5 Patch · Ring Attention Forward（Online Softmax）

## 你要交付什么

实现 **ring attention forward**——FlashAttention 与 Context Parallel 的核心：
把 K/V 沿 seq 维度切成多 chunk，每读一 chunk 维护"在线 softmax 状态"
（running max + running denom），最终得到与 full attention 数学等价的输出。

```python
def ring_attention_forward(q, k, v, num_chunks=1) -> torch.Tensor:
    """q: (B, H, Sq, D), k/v: (B, H, Sk, D). 返回 (B, H, Sq, D)。

    把 K/V 沿 seq 维分成 num_chunks 块，逐块累加 softmax；
    数学上与 F.scaled_dot_product_attention(q, k, v) 等价。"""
```

**禁止** 调 `F.scaled_dot_product_attention` 或 `nn.MultiheadAttention`。
**允许** `torch.einsum` / `torch.exp` / `torch.maximum` 等基本算子。

补丁规模目标：30–60 行。

## 在线 softmax 数学

完整 softmax: `out = softmax(QK^T / √d) @ V`

分块在线版本：维护 `m`（running max）和 `l`（running denom），每读一 chunk i：

```
S_i = Q @ K_i^T / √d                     # (B, H, Sq, chunk)
m_new = max(m_old, max(S_i, dim=-1))
l_new = l_old * exp(m_old - m_new) + sum(exp(S_i - m_new), dim=-1)
out_new = out_old * exp(m_old - m_new) + exp(S_i - m_new) @ V_i
m_old, l_old = m_new, l_new
```

最后 `out = out / l`。

为什么这样数学等价？因为 `softmax(s)_j = exp(s_j - max) / Σ exp(s_k - max)` 在
分块累加时，只要每次 max 上升就把旧的 `out_old` 和 `l_old` 用 `exp(m_old - m_new)`
重新缩放，效果与一次性算完一致。

## 不变量

1. `num_chunks=1` 时与 `F.scaled_dot_product_attention(q,k,v)` 数值等价（atol=1e-4）。
2. `num_chunks=4` 时与 num_chunks=1 数值等价（验证 online softmax 正确性）。
3. 显存峰值随 num_chunks 增大而降低（O(Sk/num_chunks) 而不是 O(Sk)）。
4. 不实现 causal mask（本关只要 full attention，causal 留 L08.7）。

## 怎么验证

```bash
make patch-test M=l09_long_context_cp
```

5 个测试，CPU 友好：

| 测试 | 验证 |
|---|---|
| `test_matches_full_attention_one_chunk` | num_chunks=1 与 F.SDPA `allclose(atol=1e-4)` |
| `test_matches_full_attention_four_chunks` | num_chunks=4 同上 |
| `test_handles_uneven_chunks` | Sk=7, num_chunks=2（最后一块更小） |
| `test_long_seq` | Sk=1024, num_chunks=8 |
| `test_grads_are_continuous` | backward 不报错（autograd 自动算） |

## 写完之后你能做什么

- 解释 FlashAttention 论文的 forward 算法每一行（你刚写过）。
- 看懂 Megatron 的 context parallel 实现，知道哪一步对应"chunk i"。
- 在 Capstone 的多模态 attention 里支持 4K+ 序列（图像 + 文本拼接超长）。
