# L11 Patch · Megatron-shaped Train Step

## 你要交付什么

实现一个 Megatron-shaped `train_step`。它是预训练生命周期中最小、可测试的单步合同：清梯度、执行 forward/backward、调用 optimizer、在 update 成功后推进 LR scheduler，并返回训练 metrics。

```python
def train_step(
    forward_backward_func,
    data_iterator,
    model,
    optimizer,
    lr_scheduler,
    iteration,
) -> dict:
    ...
```

补丁规模目标：50 到 90 行。

## 输入合同

`forward_backward_func(data_iterator, model)` 必须返回 dict，支持两种 loss 形式：

```python
{"loss": 1.2}
{"losses": [1.0, 1.4], "tokens": 4096}
```

`optimizer.step()` 可以返回：

```python
True
False
None
{"success": bool, "grad_norm": float}
```

其他返回类型应抛 `TrainStepError`。

## 不变量

1. 如果 optimizer 有 `zero_grad()`，必须在 forward/backward 前调用。
2. `forward_backward_func` 返回值必须是 dict。
3. 多个 microbatch loss 要取平均。
4. 缺少 `loss` / `losses` 或 `losses=[]` 时抛 `TrainStepError`。
5. optimizer update 成功时才调用 `lr_scheduler.step()`。
6. optimizer skip 时返回 `skipped_iter=1`，scheduler 不推进。
7. metrics 至少包含 `iteration`、`loss`、`num_microbatches`、`skipped_iter`、`lr`。
8. 有 `grad_norm` 或 `tokens` 时写入 metrics。

## 怎么验证

```bash
make patch-test M=l10_megatron_pretrain_lifecycle
```

5 个测试，CPU 即可：

| 测试 | 验证 |
|---|---|
| `test_train_step_call_order_and_metrics` | 调用顺序、loss 平均、tokens、grad_norm、lr |
| `test_scheduler_not_stepped_when_optimizer_skips` | optimizer skip 时 scheduler 不推进 |
| `test_accepts_optimizer_step_none_as_success` | optimizer 返回 None 视为成功 |
| `test_rejects_missing_loss` | 缺少 loss 字段抛错 |
| `test_rejects_empty_microbatch_losses` | 空 microbatch loss 列表抛错 |

## 写完之后你要能解释

- 为什么 `zero_grad` 要在 forward/backward 前。
- 为什么 scheduler 只能在 update 成功后推进。
- 为什么多 microbatch loss 进入 metrics 前要平均。
- 为什么 lifecycle drill 还要记录 `metrics.jsonl`、`acceptance.json` 和 checkpoint marker。
