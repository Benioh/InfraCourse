# L05 课后产物使用说明

本目录保存 Triton 行 softmax 的复用材料。它们用于复盘一次 kernel 实现或性能排查，重点是把功能正确性、环境状态和性能估算分开。

## 文件说明

| 文件 | 用途 |
|---|---|
| `debug_checklist.md` | 排查数值错误、mask 越界、资源不足、测试 skip 和 benchmark 误读 |
| `source_reading_card.md` | 复习 starter、reference、tests、MiniInfra softmax 和 memory model 主路径 |
| `performance_metrics_template.md` | 记录一次功能/性能复盘，包括 shape、dtype、GPU、max diff、row sum 和 timing |

## 建议使用顺序

1. 先用 `source_reading_card.md` 回忆源码主路径。
2. 在 GPU/Triton 环境里运行 `make patch-test M=l04_gpu_kernel`。
3. 若只能在 CPU 环境里学习，运行 `python labs/l04_gpu_kernel/scripts/bench_softmax.py --seq 4096 --block 1024 --batch 8`，并把它标注为教学模拟。
4. 出现数值错误、illegal memory access、资源不足或 skip 时，按 `debug_checklist.md` 分层排查。
5. 用 `performance_metrics_template.md` 写下本次证据和边界。

## 证据分级

| 证据 | 能说明什么 | 不能说明什么 |
|---|---|---|
| GPU patch-test pass | starter Triton kernel 在测试 shape/dtype 上与 PyTorch softmax 对齐 | backward、超长行分块、生产性能 |
| GPU patch-test skip | 当前环境缺少 CUDA 验证条件 | kernel 正确 |
| `bench_softmax.py` 输出 | stable/online softmax 数学和 roofline 预算 | starter kernel 通过 |
| `run_smoke.py` 输出 | 课程通用 artifact 链路可用 | Triton kernel 功能或性能通过 |
