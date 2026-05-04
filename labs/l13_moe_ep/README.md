# L05.5 · MoE：Top-2 Router + Capacity Factor + Aux Loss

> 本关只做一件事：**实现 Switch / GShard / Mixtral 风格的 top-2 router**——含容量限制和负载均衡 aux loss。

## 闭环

```bash
cat labs/l13_moe_ep/patch/task.md
$EDITOR labs/l13_moe_ep/patch/starter/moe_router.py
make patch-test M=l13_moe_ep   # 5 个测试，CPU OK
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_each_token_routes_to_two` | top-2 mask 每行 ≤ 2 |
| `test_combine_weights_sum_to_one_no_capacity` | 充足 capacity 时 weights 行和 ≈ 1 |
| `test_capacity_factor_drops_overflow` | 限制 capacity 时溢出 token 被 drop |
| `test_aux_loss_low_when_balanced` | 均匀 logits 时 top-2 aux ≈ 2 |
| `test_aux_loss_high_when_imbalanced` | 极端偏向时 aux 大 |

## 卡住怎么办

1. 看 `notebooks/n15_moe_router_capacity.ipynb`。
2. `make patch-hint M=l13_moe_ep`。
3. `make patch-show-solution M=l13_moe_ep`。

## 进入下一关

`make patch-test` 全绿后，下一关 [L05.8 Megatron checkpoint](../l15_megatron_parallel_checkpoint/README.md) 会补 distributed checkpoint 与 resume 主线。
