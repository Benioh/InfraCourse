# L05 · GPU Kernel：Triton 行 Softmax

这一讲解决 GPU kernel 入门里最容易写错的一类问题：给定二维 CUDA tensor，怎样用 Triton 写一个数值稳定的行 softmax，使结果与 `torch.nn.functional.softmax(x, dim=-1)` 对齐。你会在一个小 kernel 里同时看到行级并行、mask load/store、减 max 防溢出、BLOCK_SIZE 选择和 memory-bound 分析。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L05 如何从训练 step 证据进入单算子执行。
2. 读 [lecture.md](lecture.md)：理解稳定 softmax、Triton program、mask、BLOCK_SIZE、memory-bound 和 online softmax。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 MiniInfra 数学骨架、patch reference、tests 和 smoke 路径读源码。
4. 跑 notebook：[n11_gpu_memory_hierarchy.ipynb](../../notebooks/n11_gpu_memory_hierarchy.ipynb) 和 [n12_triton_softmax_walkthrough.ipynb](../../notebooks/n12_triton_softmax_walkthrough.ipynb)。
5. 做 quiz：确认数值稳定、mask identity、BLOCK_SIZE、roofline 边界。
6. 做 patch：实现 `triton_softmax(x)` 和 Triton row kernel。
7. 跑 smoke 或 bench：记录 validation-only artifact 和 CPU 侧 roofline 估算。
8. 填写 [outputs/performance_metrics_template.md](outputs/performance_metrics_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | GPU kernel / training systems performance |
| 它解决什么问题 | 行 softmax 如何在 GPU 上稳定计算，并把 padding、越界、行内 reduction 处理正确 |
| 它连接哪些指标 | max_abs_err、row_sum、dtype、shape、BLOCK_SIZE、bytes_fused、bandwidth_gbs、fallback/validation 边界 |
| 它连接哪些源码 | `patch/reference/triton_softmax.py`、`patch/tests/test_patch.py`、`mini_infra/gpu/triton_softmax.py`、`mini_infra/gpu/memory_model.py` |
| lab 检验什么 | 与 `F.softmax` 在 fp32/fp16 上数值等价；每行和接近 1；短行和长行都能处理 |

## 你会学到什么

- 为什么 softmax 要先减行最大值再 `exp`。
- `tl.program_id(0)`、`tl.arange`、`mask`、`tl.load`、`tl.max`、`tl.sum` 和 `tl.store` 各自承担什么。
- 为什么 `n_cols` 不是 2 的幂时要用 `BLOCK_SIZE = next_power_of_2(n_cols)` 和 mask。
- 为什么 mask 外填充值在 max reduction 中应为 `-inf`。
- 为什么 softmax 常按 memory-bound 分析，而不能只盯着 `exp`。
- GPU patch-test skip、CPU bench、真实 CUDA/Triton 通过分别能证明什么。

## Patch 闭环

```bash
cat labs/l04_gpu_kernel/patch/task.md
$EDITOR labs/l04_gpu_kernel/patch/starter/triton_softmax.py
make patch-test M=l04_gpu_kernel
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_matches_torch_softmax_fp32` | fp32 输出与 `F.softmax` 对齐 |
| `test_matches_torch_softmax_fp16` | fp16 输出在较宽容忍度内对齐 |
| `test_row_sums_to_one` | 每行概率和接近 1 |
| `test_short_rows` | `n_cols=64` 的短行 |
| `test_long_rows` | `n_cols=2048` 的长行 |

没有 CUDA 时测试会 skip。skip 只能说明当前环境缺少验证条件，不能写成 kernel 已通过。

## Smoke 和 Bench

```bash
python labs/l04_gpu_kernel/scripts/run_smoke.py --mode smoke
python labs/l04_gpu_kernel/scripts/bench_softmax.py --seq 4096 --block 1024 --batch 8
```

`run_smoke.py` 走课程通用的 half-mission runner，默认是 validation-only。`bench_softmax.py` 调用 MiniInfra 的 CPU 侧模拟，输出 online softmax 误差和 roofline 估算。它帮助理解算法和性能预算，不替代 GPU patch-test。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查数值错误、mask 越界、资源不足和 benchmark 误读 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 softmax kernel、tests 和 roofline 主路径 |
| [outputs/performance_metrics_template.md](outputs/performance_metrics_template.md) | 记录一次 Triton softmax 功能和性能复盘 |

## 进入下一讲

`make patch-test M=l04_gpu_kernel` 在 GPU/Triton 环境里通过，并完成一次 bench 复盘后，进入 [L06 分布式原语](../l05_distributed_primitives/README.md)。下一讲会把单算子直觉带回 Tensor Parallel Linear。
