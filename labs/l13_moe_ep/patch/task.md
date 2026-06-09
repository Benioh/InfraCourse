# L14 Patch · MoE Top-2 Router + Capacity Factor + Aux Loss

## 你要交付什么

实现 MoE 的 router——这是 GShard / Switch / Mixtral / DeepSeekMoE 的核心算法：

```python
def top2_router(
    logits: torch.Tensor,        # (num_tokens, num_experts)
    capacity_factor: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """返回 (dispatch_mask, combine_weights, aux_loss)。

    - dispatch_mask: (num_tokens, num_experts) 0/1，每个 token 选 top-2 expert
    - combine_weights: (num_tokens, num_experts) softmax 权重，未选中位置为 0
    - aux_loss: 标量，Switch Transformer 风格的 load-balance 辅助损失
    """
```

**禁止** 用 `tutel` / `fairscale.moe`。
**允许** `torch.softmax` / `torch.topk` / 张量索引等基础 API。

补丁规模目标：50–80 行。

## 接口契约

每个 token 选 **top-2 expert**（非零 combine_weights = 2，其它 = 0）。

**Capacity factor**：每个 expert 最多接收 `capacity = capacity_factor * num_tokens * 2 / num_experts` 个 token；
超出部分被 drop（dispatch_mask 设 0，token 在 MoE 输出中变 0 向量）。

**Aux loss**（Switch Transformer 公式）：
```
fraction_routed[i] = mean(dispatch_mask[:, i])           # 实际路由分数
fraction_prob[i]   = mean(softmax(logits)[:, i])         # 路由概率均值
aux_loss = num_experts * sum(fraction_routed * fraction_prob)
```
**注意**：Switch 原始公式假设 top-1。在 top-k 时：
- 每个 token 贡献到 k 个 expert，因此 `Σ fraction_routed = k`（不是 1）
- `Σ fraction_prob = 1`（softmax）
- 均匀路由时 aux_loss 的最小值 = **k**（top-2 → 最小 ≈ 2.0；top-1 → 最小 ≈ 1.0）
- 极端不均时 aux_loss 趋近 **num_experts**

我们沿用 Switch 原公式不做 /k 归一化（与 Megatron / DeepSpeed 实现一致）。

## 不变量

1. 每个 token 的 dispatch_mask 沿 expert 维度求和 ≤ 2（top-2，可能因 capacity 被 drop）。
2. 没 capacity 限制（capacity_factor 大）时，每 token combine_weights 之和 ≈ 1（softmax 在 top-2 上的归一化）。
3. 所有 token 路由到少数 expert 时 aux_loss 明显高于均匀值。
4. 完全均匀路由时 top-2 aux_loss ≈ 2.0（top-k 时最优值约为 k）。

## 怎么验证

```bash
make patch-test M=l13_moe_ep
```

5 个测试，CPU 即可：

| 测试 | 验证 |
|---|---|
| `test_each_token_routes_to_two` | top-2 mask 每行求和 ≤ 2 |
| `test_combine_weights_sum_to_one_no_capacity` | capacity 充足时 weights 行和 ≈ 1 |
| `test_capacity_factor_drops_overflow` | capacity=0.5 时部分 token 被 drop |
| `test_aux_loss_low_when_balanced` | 均匀 logits 时 top-2 aux_loss 接近 2 |
| `test_aux_loss_high_when_imbalanced` | 所有 token 都选 expert 0 时 aux_loss 高 |

## 写完之后你能做什么

- 解释 Switch / GShard / Mixtral / DeepSeekMoE 的 router 设计差异。
- 在 Capstone 给多模态模型加 MoE 层（每个模态走不同 expert）。
- 看懂 Megatron `moe_layer.py` 的 routing + dispatch 实现。
