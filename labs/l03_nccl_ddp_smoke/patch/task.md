# L04 Patch · 手写 ManualDDP

## 你要交付什么

不用 `torch.nn.parallel.DistributedDataParallel`，自己写一个 `ManualDDP` 类：

```python
class ManualDDP:
    def __init__(self, model, process_group=None): ...
    def synchronize_grads(self): ...  # 在 backward 之后调用，原地更新所有 grads
```

**禁止** `import torch.nn.parallel` 或 `import torch.distributed.fsdp`。
**允许** `torch.distributed.all_reduce / all_reduce_coalesced` 等低层 API。

补丁规模目标：30–80 行。

## 接口契约

```python
ddp = ManualDDP(model, process_group=group)
out = ddp.module(x)             # 等价于 model(x)（不需要在 forward 时通信）
out.sum().backward()
ddp.synchronize_grads()         # 调用后，每个 param.grad = (Σ_ranks local_grad) / world_size
optimizer.step()
```

调用 `synchronize_grads()` 之后：
- 每张 rank 的 `param.grad` 必须等于所有 rank local grad 的 **平均**（不是只 sum）。
- 数值上必须与 `torch.nn.parallel.DistributedDataParallel` 完全一致。

## 不变量

1. `ManualDDP(model)` 不可改变 `model.parameters()`（同一份内存）。
2. `requires_grad=False` 的 param 不参与 all-reduce。
3. 当 `dist` 未初始化或 world_size==1 时，`synchronize_grads()` 是 no-op。
4. backward 之后某些 param 的 grad 仍可能为 None（视模型路径而定），ManualDDP 必须容忍这种情况。
5. 平均除法可以在 all-reduce 之前 `grad / ws` 也可以在之后 `all_reduce(grad) / ws`，结果一致。

## 怎么验证

```bash
make patch-test M=l03_nccl_ddp_smoke
```

5 个**结果对比**测试（CPU gloo, world=2）：

| 测试 | 验证 |
|---|---|
| `test_grads_match_pytorch_ddp` | 与 `torch.nn.parallel.DDP` 在相同 input 下 grad 完全相等 |
| `test_grads_are_averaged_not_summed` | grad 是均值不是和（差一个 world_size 因子）|
| `test_world_size_1_is_noop` | world_size=1 时 grad 不被改动 |
| `test_skips_no_grad_params` | requires_grad=False 的 param 没被通信 |
| `test_handles_partial_grads` | 某些 param 的 grad 为 None 时不报错 |

## 写完之后你能做什么

- 解释为什么 PyTorch DDP 默认用 25MB bucket；为什么大模型用 100MB+。
- 在 L06 给 Megatron 加 gradient bucket overlap 时知道改哪一行。
- 调试"DDP loss 偏大 world_size 倍"——经典 bug：忘了除以 world_size。
