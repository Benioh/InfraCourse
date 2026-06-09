# L12 Patch · Bucketed Manual DDP（合并 grad 通信）

## 你要交付什么

在逐参数 ManualDDP 的基础上，加 **bucket 机制**：把 params 按字节数分组，
每个 bucket 一次性 all-reduce 所有 grad（用 `_flatten_dense_tensors`），
而不是每个 param 单独 all-reduce。

```python
class BucketedManualDDP:
    def __init__(self, model, bucket_size_mb=25, process_group=None): ...
    def synchronize_grads(self): ...
```

为什么要 bucket？每次 NCCL all-reduce 有固定启动开销（~10us），param 多时
总开销爆炸（1000 个 param 的 2B 模型 → 10ms 纯开销）。Bucket 把多个 grad
合并成一次大 all-reduce，启动开销摊薄。

**禁止** 用 `torch.nn.parallel.DistributedDataParallel`。
**允许** `torch._utils._flatten_dense_tensors` / `_unflatten_dense_tensors`。

补丁规模目标：50–80 行。

## 接口契约

```python
ddp = BucketedManualDDP(model, bucket_size_mb=25)
out = ddp.module(x)
out.sum().backward()
ddp.synchronize_grads()
optimizer.step()
```

## 不变量

1. **数值上**与逐参数 ManualDDP 完全等价（也与 PyTorch DDP 等价）。
2. params 按 `requires_grad=True` 顺序分组：累加超过 bucket_size_mb 就新开一桶。
3. 单个 param > bucket_size 也可以（自成一桶）。
4. world_size=1 时 `synchronize_grads()` 是 no-op。

## 怎么验证

```bash
make patch-test M=l11_megatron_scale_optimization
```

5 个测试，2-rank gloo CPU：

| 测试 | 验证 |
|---|---|
| `test_grads_match_pytorch_ddp` | 与 PyTorch DDP grad 一致 |
| `test_works_with_huge_param` | 单 param > bucket_size 也正常工作 |
| `test_works_with_tiny_params` | 很多小 param，最终通信次数 < param 数 |
| `test_world_size_1_is_noop` | 单 rank 不改 grad |
| `test_skips_no_grad_params` | 冻结 param 不参与 |

## 写完之后你能做什么

- 解释 PyTorch DDP `gradient_as_bucket_view` 的存储优化。
- 看懂 Megatron `distributed_data_parallel.py` 的 `_buckets` 与 `_buffers`。
- 在 Capstone 多模态训练里调 bucket size 优化通信吞吐。
