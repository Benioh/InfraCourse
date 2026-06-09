# L10 Patch · Ring Attention Forward（Online Softmax）

## 你要交付什么

实现一个教学版 `ring_attention_forward`。它把 K/V 沿序列维切成多个 chunk，并用 online softmax 逐块累加，最终得到与 full attention 等价的输出。

```python
def ring_attention_forward(q, k, v, num_chunks=1) -> torch.Tensor:
    """q: (B, H, Sq, D), k/v: (B, H, Sk, D). 返回 (B, H, Sq, D)."""
```

禁止调用 `torch.nn.functional.scaled_dot_product_attention` 或 `nn.MultiheadAttention`。允许使用 `torch.einsum`、`torch.exp`、`torch.maximum`、`torch.chunk` 等基础算子。

补丁规模目标：30 到 60 行。

## 在线 softmax 数学

完整 attention：

```text
scores = Q @ K^T / sqrt(D)
out = softmax(scores) @ V
```

分块版本每次处理一个 K/V chunk：

```text
scores_i = Q @ K_i^T / sqrt(D)
chunk_max = max(scores_i, dim=-1, keepdim=True)
new_max = max(running_max, chunk_max)
exp_old = exp(running_max - new_max)
exp_chunk = exp(scores_i - new_max)
running_denom = running_denom * exp_old + sum(exp_chunk)
running_out = running_out * exp_old + exp_chunk @ V_i
running_max = new_max
```

最后返回：

```text
running_out / running_denom
```

第一次循环时 `running_max = -inf`，旧状态贡献应为 0。可以用 `torch.where(torch.isfinite(running_max), torch.exp(running_max - new_max), torch.zeros_like(new_max))` 处理这个边界。

## 不变量

1. `num_chunks=1` 时输出与 PyTorch SDPA 对齐。
2. `num_chunks=4` 时输出仍与 PyTorch SDPA 对齐。
3. `Sk` 不能被 `num_chunks` 整除时仍正确。
4. 长 K/V 场景下误差在测试容差内。
5. 输出支持 autograd backward，并产生有限梯度。

## 不要求实现

- causal mask
- dropout
- 自定义 backward
- 多 GPU ring 通信
- Transformer Engine 或 FlashAttention kernel

这些生产复杂度在讲义和源码带读里解释，本关 patch 只验收 forward 数学合同。

## 怎么验证

```bash
make patch-test M=l09_long_context_cp
```

5 个测试，CPU 可运行：

| 测试 | 验证 |
|---|---|
| `test_matches_full_attention_one_chunk` | `num_chunks=1` 与 SDPA 对齐 |
| `test_matches_full_attention_four_chunks` | `num_chunks=4` 与 SDPA 对齐 |
| `test_handles_uneven_chunks` | `Sk=7, num_chunks=2` |
| `test_long_seq` | `Sq=256, Sk=1024, num_chunks=8` |
| `test_grads_are_continuous` | backward 产生有限梯度 |

## 写完之后你要能解释

- 为什么每个 chunk 独立 softmax 后相加会错。
- `running_max`、`running_denom` 和 `running_out` 的 shape。
- 为什么旧分母和旧输出要同时缩放。
- 为什么本地 `num_chunks` 和真实 CP world size 有联系，但不是同一个概念。
- 为什么通过 patch 只能证明 forward 数学，不能证明真实多 GPU CP 性能。
