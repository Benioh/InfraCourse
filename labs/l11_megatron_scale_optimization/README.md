# L05 · Megatron Scale：Bucketed Manual DDP

> 本关只做一件事：**把 L01.5 的 ManualDDP 升级成 BucketedManualDDP**——按字节数分组合并 all-reduce，摊薄 NCCL 启动开销。

## 闭环

```bash
cat labs/l11_megatron_scale_optimization/patch/task.md
$EDITOR labs/l11_megatron_scale_optimization/patch/starter/bucketed_ddp.py
make patch-test M=l11_megatron_scale_optimization   # 5 个测试，2-rank gloo
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_grads_match_pytorch_ddp` | bucketed grad == PyTorch DDP |
| `test_works_with_huge_param` | 单 param > bucket size 也工作 |
| `test_works_with_tiny_params` | 很多小 param 合并到一桶 |
| `test_world_size_1_is_noop` | 单 rank 不改 grad |
| `test_skips_no_grad_params` | 冻结 param 不参与 |

## 卡住怎么办

1. 看 Megatron `distributed_data_parallel.py` 的 bucket 实现。
2. `make patch-hint M=l11_megatron_scale_optimization`。
3. `make patch-show-solution M=l11_megatron_scale_optimization`。

## 进入下一关

`make patch-test` 全绿后，继续做源码理解口试。下一关 [L05.5 MoE/EP](../l13_moe_ep/README.md) 让你写 top-2 router + capacity factor。
