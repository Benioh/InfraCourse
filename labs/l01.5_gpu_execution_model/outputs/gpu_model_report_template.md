# L01.5 GPU 执行模型报告

## 硬件信息

- GPU 型号：___
- SM 数量：___
- HBM 容量：___
- HBM 带宽：___
- Tensor Core 型号/代：___

## Dispatch 链路观察

| 阶段 | 观察到的现象 |
|---|---|
| python_call | ___ |
| cpp_dispatch | ___ |
| kernel_launch | ___ |
| gpu_execute | ___ |

## 异步执行证据

- 不 sync 时 CPU 端测量时间：___ ms
- sync 后完整时间：___ ms
- 时间差倍数：___×
- 结论：（是否确认 GPU 异步执行）___

## Sync 开销观察

| 矩阵大小 | 每次 sync 耗时 | 无 sync 耗时 | Overhead |
|---|---|---|---|
| 256×256 | ___ ms | ___ ms | ___% |
| 1024×1024 | ___ ms | ___ ms | ___% |
| 4096×4096 | ___ ms | ___ ms | ___% |

## Stream Overlap 观察

| 矩阵大小 | Sequential | Two streams | Speedup |
|---|---|---|---|
| 256×256 | ___ ms | ___ ms | ___× |
| 1024×1024 | ___ ms | ___ ms | ___× |
| 4096×4096 | ___ ms | ___ ms | ___× |

大矩阵 overlap 效果减弱的原因：___

## Bottleneck 分类练习

| Op | FLOPs | Bytes | Arithmetic Intensity | 分类 |
|---|---|---|---|---|
| MatMul(4096,4096,4096) | ___ | ___ | ___ | ___ |
| Elementwise(4096,4096) | ___ | ___ | ___ | ___ |
| LayerNorm(4096) | ___ | ___ | ___ | ___ |

## 反思

- 这次实验中最意外的发现：___
- 对后续 profiler 分析的启发：___
