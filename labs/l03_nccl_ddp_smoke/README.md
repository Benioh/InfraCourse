# L01.5 · 手写 ManualDDP

> 本关只做一件事：**不用 PyTorch DDP，自己写一个 ManualDDP**，在 backward 之后通过 `dist.all_reduce` 把所有 rank 的 grad 取平均。

写完这关你能解释 PyTorch DDP 的全部核心机制（除了 bucket overlap）。

## 闭环

```bash
cat labs/l03_nccl_ddp_smoke/patch/task.md
$EDITOR labs/l03_nccl_ddp_smoke/patch/starter/manual_ddp.py
make patch-test M=l03_nccl_ddp_smoke   # 5 个测试，2-rank gloo CPU
```

## 你要改的文件

```
labs/l03_nccl_ddp_smoke/patch/
├── task.md
├── starter/manual_ddp.py     ★ 唯一要改的文件
├── reference/manual_ddp.py
└── tests/
    ├── conftest.py           ← 2-rank spawn harness
    ├── worker_cases.py
    └── test_patch.py
```

## 测试覆盖（5 个，结果对比）

| 测试 | 验证 |
|---|---|
| `test_grads_match_pytorch_ddp` | 与 `torch.nn.parallel.DDP` 在相同 input 下 grad 完全相等 |
| `test_grads_are_averaged_not_summed` | grad 是均值不是和（差 ws 倍） |
| `test_world_size_1_is_noop` | 单 rank 时不改动 grad |
| `test_skips_no_grad_params` | 冻结 param 不被通信 |
| `test_handles_partial_grads` | 某些 grad=None 时不报错 |

## 卡住怎么办

1. 看 `notebooks/n03_ddp_collectives.ipynb`。
2. `make patch-hint M=l03_nccl_ddp_smoke`。
3. `make patch-show-solution M=l03_nccl_ddp_smoke`。

## 进入下一关

`make patch-test M=l03_nccl_ddp_smoke` 全绿后，继续做源码理解口试。下一关 [L01.7 GPU kernel](../l04_gpu_kernel/README.md) 让你写 Triton 融合 softmax+dropout。
