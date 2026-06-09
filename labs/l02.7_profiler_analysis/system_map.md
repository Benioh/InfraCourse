# L02.7 System Map · Profiler 与性能分析

## 在全课程中的位置

```
L01.5 GPU 执行模型与硬件基础（dispatch、async、内存层级）
    ↓
L02 PyTorch 显存账本（params/grads/optimizer/activation）
    ↓
L02.5 Memory Snapshot（分配追踪）
    ↓
→ L02.7 Profiler 与性能分析 ←（你在这里）
    ↓
L03 Manual DDP（梯度同步）→ 后续用 profiler 验证 overlap
    ↓
L04 GPU Kernel（Triton）→ 用 roofline 判断 kernel 瓶颈
```

## 本讲证据边界

| 证据类型 | 本讲产出 | 下游依赖 |
|---|---|---|
| Op 级别时间分析 | profiler_summary.json | 所有后续 Lab 的性能验证 |
| 瓶颈分类结论 | bottleneck_analysis.json | 优化方向选择 |
| GPU idle 段识别 | gpu_idle_segments.json | DDP overlap 验证、data loading 诊断 |
| 内存分类 | memory_breakdown.json | OOM 诊断、activation checkpoint 决策 |

## 本讲不做什么

- 不实际跑 nsys（需要 GPU + NVIDIA 工具链）——但教你怎么解读 nsys 输出
- 不优化具体的 kernel（那是 L04）
- 不做分布式通信优化（那是 L11、L12）
- 不做 activation checkpoint（那是 L06）

本讲建立分析能力，后续 lab 是应用这些分析做优化。
