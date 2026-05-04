# L04.8 Patch · Megatron-shaped Train Step

## 你要交付什么

实现一个 Megatron-shaped `train_step`。它是 `mini_infra/megatron/training/training.py` 中训练循环的最小可测切片：清梯度、执行 forward/backward、optimizer step、LR scheduler step、返回训练指标。

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

补丁规模目标：50-90 行。

## 接口契约

`forward_backward_func(data_iterator, model)` 返回：

```python
{"loss": 1.2}
# 或
{"losses": [1.0, 1.4], "tokens": 4096}
```

`optimizer.step()` 可以返回 `True/False`，也可以返回 `{"success": bool, "grad_norm": float}`。

## 不变量

1. 如果 optimizer 有 `zero_grad()`，必须在 forward/backward 前调用。
2. 多个 microbatch loss 要取平均。
3. optimizer step 成功时才调用 lr scheduler。
4. optimizer step 被跳过时返回 `skipped_iter=1`。
5. 返回指标至少包含 `iteration`、`loss`、`num_microbatches`、`skipped_iter`、`lr`。

## 怎么验证

```bash
make patch-test M=l10_megatron_pretrain_lifecycle
```

5 个测试覆盖调用顺序、microbatch loss、optimizer skip、scheduler step 和异常输入。

## 写完之后你能做什么

- 把 L04 的 LR scheduler 放回真实训练 step。
- 解释 Megatron `train_step` 如何连接 forward/backward、optimizer、scheduler 和 metrics。
- 继续阅读 `github_repo/Megatron-LM/megatron/training/training.py` 时有明确锚点。
