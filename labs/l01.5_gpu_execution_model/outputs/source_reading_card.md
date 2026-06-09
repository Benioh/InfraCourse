# L01.5 源码阅读卡

## 快速复习路径

1. **Dispatch 链路**：Python → torch._C → ATen Dispatcher → CUDA kernel launch → GPU async execute
2. **异步模型**：CPU 提交 kernel 立即返回；sync = CPU 等 GPU 的 barrier
3. **内存层级**：Register(30×) > Shared Memory(30×) > L2(4×) > HBM(1×)
4. **Bottleneck 判断**：AI = FLOPs / Bytes；AI > ridge point → compute bound

## 关键文件

| 文件 | 主路径 |
|---|---|
| `scripts/sync_overhead_demo.py` | 对比有无 sync 的时间差 |
| `scripts/stream_overlap_demo.py` | 两个 stream 的 kernel overlap |
| `patch/reference/gpu_exec_probe.py` | dispatch 链路 + 内存层级 + bottleneck 分类 |
| `lecture.md §2` | Dispatch 全链路详解 |
| `lecture.md §3` | GPU 硬件层级 |
| `lecture.md §5` | Compute vs Memory bound + Roofline |

## 一句话速查

- **为什么不 sync 测时间不准** → GPU 异步执行，CPU 端 time.time() 只测了 launch 时间
- **为什么小 kernel 慢** → CPU dispatch overhead 占比大，GPU 大部分时间在等
- **为什么 kernel fusion 有效** → 减少 HBM 往返，中间结果留在 register/shared memory
- **为什么混合精度快 2-3×** → Tensor Core 算力比 CUDA Core 高 10-30×
- **为什么 DDP overlap 有效** → 通信走 NVLink（不抢 SM），计算走 SM，互不干扰
