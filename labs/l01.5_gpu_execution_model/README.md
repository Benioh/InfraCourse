# L01.5 · GPU 执行模型与硬件基础

这一讲在环境探针（L01）与 PyTorch 显存分析（L02）之间插入一层：理解代码如何从 Python 到达 GPU，以及 GPU 硬件的执行层级。后面做 kernel 优化（L04）、profiler 分析和性能瓶颈定位时，如果对 dispatch 链路和硬件模型没有直觉，你会无法判断瓶颈在 CPU 还是 GPU、在 launch 还是 compute、在 bandwidth 还是 compute throughput。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认本讲在全课程中的位置。
2. 读 [lecture.md](lecture.md)：从 PyTorch op dispatch 讲到 GPU 硬件执行层级。
3. 读 [source_walkthrough.md](source_walkthrough.md)：跟读 CUDA stream 与 sync 实验代码。
4. 做 quiz：确认你对 dispatch 链路、异步执行、硬件层级的理解。
5. 做 patch：实现 GPU 执行模型的探测与分析函数。
6. 跑 smoke：生成一次 kernel launch + sync 的证据。
7. 填写 [outputs/gpu_model_report_template.md](outputs/gpu_model_report_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | GPU execution model & hardware fundamentals |
| 它解决什么问题 | 理解 Python 代码如何经过 dispatch 到达 GPU 执行，以及 GPU 硬件如何组织计算 |
| 它连接哪些证据 | kernel launch timeline、sync 前后时间差、SM occupancy、memory hierarchy 层级图 |
| 它连接哪些源码 | `patch/reference/gpu_exec_probe.py`、`scripts/stream_overlap_demo.py`、`scripts/sync_overhead_demo.py` |
| lab 检验什么 | dispatch 链路理解、async 执行证明、硬件层级计算、Stream 并发 |

## 你会学到什么

### PyTorch Op Dispatch 链路
- 一个 `torch.matmul(A, B)` 从 Python 到 GPU 执行经历的完整路径：Python → torch._C → ATen dispatcher → CUDA kernel launch → GPU 异步执行。
- 为什么 Python 端调用返回了不代表 GPU 算完了（异步执行模型）。
- `torch.cuda.synchronize()` 的作用：强制 CPU 等待 GPU 所有已提交 kernel 完成。
- CPU overhead 的含义：即使 GPU 很快，如果 Python/C++ 端 dispatch 慢，GPU 会 idle。

### GPU 硬件执行层级
- **Kernel**：GPU 上一次函数调用的单位。一个 PyTorch op 可能对应一个或多个 kernel。
- **Thread → Warp → Block → Grid**：GPU 并行执行的层级结构。32 个 thread 组成一个 warp（SIMT 执行单位），多个 warp 组成一个 block，block 被调度到 SM 上。
- **SM (Streaming Multiprocessor)**：GPU 的基本计算单元。每个 SM 有自己的 register file、shared memory 和 warp scheduler。
- **内存层级**：Register（最快，per-thread）→ Shared Memory / L1 Cache / SRAM（per-SM，~100TB/s）→ L2 Cache → HBM / Global Memory（全局，~2-3TB/s on H100）。
- **Tensor Core**：专用矩阵乘法单元，做 mixed-precision matmul（如 FP16×FP16→FP32），吞吐远高于 CUDA Core。

### CUDA Stream
- Stream 是 GPU 上的有序命令队列。同一 stream 内 kernel 按提交顺序执行；不同 stream 可以并发。
- 默认所有 PyTorch op 在 default stream 上执行。
- `torch.cuda.Stream()` 创建新 stream，`with torch.cuda.stream(s)` 切换。
- stream 之间的依赖通过 Event 建立。

### 性能直觉
- **Compute bound**：kernel 的算力需求超过 GPU 算力上限（如大矩阵乘法）。
- **Memory bound**：kernel 花大部分时间在读写 HBM（如 element-wise 操作、layer norm）。
- **CPU overhead / Launch bound**：GPU 算得比 CPU 提交还快，GPU 在等 CPU 发新 kernel。
- 为什么 `torch.cuda.synchronize()` 用来正确测时间，但过度 sync 会杀性能。

## Patch 闭环

```bash
cat labs/l01.5_gpu_execution_model/patch/task.md
$EDITOR labs/l01.5_gpu_execution_model/patch/starter/gpu_exec_probe.py
make patch-test M=l01.5_gpu_execution_model
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_dispatch_chain_stages` | 能正确列出 PyTorch op 从 Python 到 GPU 的 dispatch 阶段 |
| `test_async_timing_without_sync` | 不 sync 时 CPU 端测量时间远小于实际 GPU 执行时间 |
| `test_async_timing_with_sync` | sync 后 CPU 端时间包含 GPU 执行时间 |
| `test_memory_hierarchy_bandwidth` | 能正确排序 GPU 内存层级的带宽 |
| `test_thread_hierarchy_sizes` | 正确计算 warp/block/grid 中的 thread 数量关系 |
| `test_stream_independence` | 两个 stream 上的 kernel 可以 overlap |
| `test_compute_vs_memory_bound` | 能根据 op 特征判断 compute bound 还是 memory bound |

## Smoke 闭环

```bash
python labs/l01.5_gpu_execution_model/scripts/run_smoke.py \
  --config configs/4090_debug.yaml \
  --mode smoke
```

smoke 会写出：

```text
runs/mini_infra/l01.5_gpu_execution_model/<run-id>/
├── command.sh
├── config.resolved.yaml
├── metrics.jsonl
├── report.md
└── artifacts/
    ├── dispatch_chain.json
    ├── async_timing.json
    ├── stream_overlap.json
    └── hardware_spec.json
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/gpu_model_report_template.md](outputs/gpu_model_report_template.md) | 记录本机 GPU 硬件参数和 dispatch 观察 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速复习 dispatch 链路和硬件层级 |

## 进入下一讲

`make patch-test M=l01.5_gpu_execution_model` 通过后，进入 [L02 PyTorch 显存账本](../l02_pytorch_systems/README.md)。下一讲会基于这里的硬件直觉，开始分析 PyTorch 训练循环的显存组成。
