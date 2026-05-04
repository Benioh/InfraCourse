# L05.3 Patch · FSDP2 wrap with mixed-precision policy

## 你要交付什么

实现 `wrap_transformer_blocks_fsdp2`：

```python
def wrap_transformer_blocks_fsdp2(
    model: nn.Module,
    block_cls: type[nn.Module],
    mp_policy: "MixedPrecisionPolicy | None" = None,
    reshard_after_forward: bool = True,
    skip: Callable[[str, nn.Module], bool] | None = None,
) -> WrapReport:
    ...
```

它必须：

1. 遍历 `model` 所有命名子模块，对每一个 `isinstance(child, block_cls)` 的子模块
   调用 `fully_shard(child, mp_policy=..., reshard_after_forward=...)`
2. 对 root `model` **最后**调用一次 `fully_shard`
3. 跳过 `skip(name, child)` 返回 True 的子模块
4. 返回 `WrapReport`：包含 `wrapped_blocks: list[str]`, `root_wrapped: bool`,
   `mp_policy_summary: dict`

**禁止使用** `torch.distributed.fsdp.FullyShardedDataParallel`（那是 FSDP1）。
**允许使用** `torch.distributed._composable.fsdp.fully_shard`、`MixedPrecisionPolicy`。

补丁规模目标：30–60 行 Python。

## 不变量

1. wrap 顺序：所有 block 先 wrap，root 必须最后一个 wrap
2. `mp_policy` 必须原样转发给 `fully_shard`，不能在中间生成新的 policy
3. `reshard_after_forward` 必须传给每个 `fully_shard` 调用
4. `skip` callback 默认是 `None`（不跳过），返回 `True` 时跳过该子模块
5. 返回的 `wrapped_blocks` 顺序 = 遍历到这些 block 的顺序

## 怎么验证

```bash
make patch-test M=l12_fsdp2_llama
```

CPU 测试覆盖 wrap 行为；GPU 测试（需 `RUN_GPU_TESTS=1`）真实跑前向 + 反向 +
SHARDED_STATE_DICT 往返。

## 写完之后你能做什么

- 看懂 TorchTitan `parallelize_llama.py::apply_fsdp2`
- 在自己的训练代码里把 Llama 1B / 7B 切碎跑起来
- 在面试里讲清楚 FSDP1 vs FSDP2 的差异：composable、DTensor-native、no flat-parameter
