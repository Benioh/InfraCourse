# L01.5 System Map · GPU 执行模型与硬件基础

## 在全课程中的位置

```
L01 环境探针（Python/CUDA/NCCL 可用性）
    ↓
→ L01.5 GPU 执行模型与硬件基础 ←（你在这里）
    ↓
L02 PyTorch 显存账本（params/grads/optimizer/activation）
    ↓
L03 Manual DDP（all-reduce gradient sync）
    ↓
L04 GPU Kernel / Triton（实际写 kernel）
```

## 本讲证据边界

| 证据类型 | 本讲产出 | 下游依赖 |
|---|---|---|
| dispatch 链路理解 | `dispatch_chain.json` | L04 写 Triton kernel 时理解 launch 开销 |
| 异步执行证明 | `async_timing.json` | L02 profiler trace 解读、后续所有 timing 测量 |
| 硬件层级模型 | `hardware_spec.json` | L04 memory-bound 分析、L22 FlashAttention 理解 |
| Stream overlap | `stream_overlap.json` | L03 DDP 计算通信 overlap、L11 bucketed overlap |

## 本讲不做什么

- 不写真正的 CUDA kernel（那是 L04）
- 不用 profiler 工具（那是 Profiler 专题）
- 不做分布式通信（那是 L03）
- 不分析完整训练循环的显存（那是 L02）

本讲建立心智模型，后续 lab 在此基础上做实操。
