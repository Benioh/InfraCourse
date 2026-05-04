# L01.7 · GPU Kernel：Triton 行 Softmax

> 本关只做一件事：**用 Triton 写一个数值稳定的行 softmax kernel**，与 `F.softmax` 在 fp32/fp16 上完全等价。

写完这关你能解释 FlashAttention 的 online softmax，并能在 L05.5/L08.7 写自己的 GPU 内核。

## 闭环

```bash
cat labs/l04_gpu_kernel/patch/task.md
$EDITOR labs/l04_gpu_kernel/patch/starter/triton_softmax.py
make patch-test M=l04_gpu_kernel   # 5 个测试，需要 GPU+Triton；无 GPU 自动 skip
```
        
## 你要改的文件

```
labs/l04_gpu_kernel/patch/
├── task.md
├── starter/triton_softmax.py     ★ 唯一要改的文件
├── reference/triton_softmax.py
└── tests/test_patch.py
```

## 测试覆盖（5 个，结果对比，标记 gpu 自动 skip）

| 测试 | 验证 |
|---|---|
| `test_matches_torch_softmax_fp32` | fp32 atol=1e-5 |
| `test_matches_torch_softmax_fp16` | fp16 atol=1e-2 |
| `test_row_sums_to_one` | 每行 sum ≈ 1 |
| `test_short_rows` | n_cols=64 |
| `test_long_rows` | n_cols=2048 |

## 卡住怎么办

1. 看 `notebooks/n12_triton_softmax_walkthrough.ipynb` + `n11_gpu_memory_hierarchy.ipynb`。
2. `make patch-hint M=l04_gpu_kernel`。
3. `make patch-show-solution M=l04_gpu_kernel`。

## 进入下一关

`make patch-test M=l04_gpu_kernel` 全绿后，继续做源码理解口试。下一关 [L02 分布式原语](../l05_distributed_primitives/README.md) 让你手写 Tensor Parallel Linear。
