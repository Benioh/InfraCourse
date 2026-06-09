# L02.7 讲义：Profiler 与性能分析

当一个训练循环跑得比预期慢，你面对的第一个问题是：时间花在哪里？是 GPU 在做有用计算？还是 GPU 在等 CPU 发任务？还是 GPU 在等通信完成？还是 GPU 在等数据？

不同原因对应不同的优化方向。Profiler 就是帮你回答"时间花在哪里"的工具。

## 1. 学完要能回答什么

1. torch.profiler 输出中 CPU Time 和 CUDA Time 分别代表什么？为什么 CPU Time 大不一定代表慢？
2. 如何从 profiler 输出中快速定位最耗时的 op？
3. 如何从 nsys timeline 中判断 GPU 是否有大量 idle time？
4. 如何判断 NCCL 通信和 GPU 计算是否 overlap？
5. torch memory profiler 如何帮你找到 OOM 的根因？
6. 五种性能瓶颈（compute/memory/launch/comm/data）各自的 profiler 特征是什么？
7. 什么时候该用 torch.profiler？什么时候该用 nsys？什么时候该用 memory profiler？
8. profiling 本身有多少开销？如何最小化对结果的干扰？

## 2. torch.profiler：Op 级别分析

### 2.1 基本用法

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("train_step"):
        output = model(input_batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

# 打印 top-10 耗时 op
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 2.2 输出解读

```
Name                    CPU total    CUDA total    # Calls    Input Shapes
----------------------  ----------   -----------   --------   ------------
aten::mm                 120ms         850ms         48       [[4096,4096],[4096,4096]]
aten::addmm               80ms         420ms         24       [[4096],[4096,4096],[4096,4096]]
aten::layer_norm           40ms         180ms         24       [[1,2048,4096]]
aten::softmax              15ms          95ms         24       [[1,32,2048,2048]]
```

**关键点**：
- `CPU total`：CPU 侧花的时间（dispatch + launch + 可能的 sync 等待）。
- `CUDA total`：GPU 侧实际执行时间。
- CPU Time > CUDA Time 很正常——因为 CPU 可能在做 Python 逻辑、dispatcher 开销等。
- **真正有意义的是 CUDA total**：这是 GPU 上真正的计算/带宽时间。
- `Input Shapes` 帮你判断 op 是否在做合理大小的工作。

### 2.3 识别"大量小 op"模式

如果 profiler 输出中有大量 `# Calls` 很高、每次 `CUDA total` 很小（<10μs）的 op，这暗示：
- GPU 的时间被 kernel launch overhead 吃掉了。
- 适合用 torch.compile 或 CUDA Graph 优化。

### 2.4 Schedule 模式（避免 warmup 干扰）

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,      # 前 2 步不采集
        warmup=2,    # 接下来 2 步采集但不计入统计（warmup）
        active=3,    # 真正统计的 3 步
        repeat=1,
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(model, batch)
        prof.step()
```

## 3. nsys / Nsight Systems：系统级 Timeline

### 3.1 为什么需要 nsys

torch.profiler 告诉你每个 op 花了多少时间，但它**不能**直观展示：
- GPU 什么时候在 idle（kernel 之间的空白）。
- NCCL 通信和计算是否真的在时间上重叠。
- 不同 rank 之间的时间差异（straggler）。
- CPU 端 Python/C++ 执行和 GPU kernel 的时间对应关系。

nsys 是 NVIDIA 的系统级 profiler，它直接从 GPU driver 采集 kernel 执行事件，生成一个精确的 timeline。

### 3.2 基本用法

```bash
# 采集一次训练的 timeline
nsys profile --trace=cuda,nvtx,osrt \
  --output=training_profile \
  --force-overwrite=true \
  python train.py --steps=10

# 生成报告
nsys stats training_profile.nsys-rep

# 用 Nsight Systems GUI 打开 .nsys-rep 文件查看 timeline
```

### 3.3 Timeline 解读

在 nsys timeline 中你会看到多行：

```
CPU Thread (Python) |████░░░░████░░░░████░░░░|  (Python/C++ 执行)
CUDA Stream 0      |  ████████  ████████  ████|  (GPU compute kernel)
CUDA Stream 1      |     ████      ████      |  (NCCL comm kernel)
```

- **GPU idle**：CUDA Stream 行中的空白段。如果 Stream 0 有大量空白，说明 GPU 在等。
- **Overlap**：如果 Stream 0 和 Stream 1 的色块在时间上有重叠，说明计算和通信在 overlap。
- **Launch gap**：连续 kernel 之间如果有 5-20μs 的小间隙，那是正常的 launch overhead。如果间隙 >100μs，可能是 CPU 侧有阻塞。

### 3.4 常见 Pattern

| Timeline Pattern | 含义 | 优化方向 |
|---|---|---|
| GPU 全满，无空白 | Compute/memory bound | 换 dtype、fusion |
| GPU 大段空白在 step 开头 | Data loading bottleneck | pin_memory、num_workers |
| GPU 小段空白在每个 kernel 之间 | Launch bound | compile / CUDA Graph |
| NCCL 和 compute 不重叠 | 通信未 overlap | 检查 DDP bucket 配置 |
| 某些 rank 的 compute 比别人长 | 负载不均 | 检查 batch 分配 |

### 3.5 nsys 与多卡分析

```bash
# 多卡场景：对每个 rank 分别采集
torchrun --nproc_per_node=4 train.py

# 或者直接 nsys 包裹 torchrun
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
  torchrun --nproc_per_node=4 train.py --steps=5
```

对比不同 rank 的 timeline 可以发现：
- 某个 rank 的 backward 比别人长（数据不均、模型参数分布不均）。
- NCCL all-reduce 时间在所有 rank 上一致（正常），还是某些 rank 先到先等（straggler）。

## 4. torch memory profiler：显存分析

### 4.1 为什么 nvidia-smi 不够

`nvidia-smi` 只告诉你"进程总共用了多少显存"，但不告诉你：
- 这些显存是谁分配的？是 activation？optimizer state？gradient？还是泄漏的 tensor？
- OOM 发生前的分配历史是什么？哪个 op 导致了内存峰值？
- PyTorch caching allocator 持有但未使用的内存有多少？

### 4.2 Memory Snapshot

```python
# 开始记录内存历史
torch.cuda.memory._record_memory_history(max_entries=100000)

# 执行训练逻辑
output = model(input_batch)
loss = criterion(output, targets)
loss.backward()
optimizer.step()

# 导出 snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)  # 停止记录
```

### 4.3 分析 snapshot

Memory snapshot 记录了每次 `cudaMalloc` 和 `cudaFree` 的：
- 分配大小
- 调用栈（哪行代码触发的）
- 时间戳
- 当前是否还 live

从中可以计算：
- **live allocations**：当前仍在使用的分配。
- **peak memory**：最高水位线。
- **分类占比**：按调用栈聚合，判断是 optimizer state、activation 还是 gradient 占大头。

### 4.4 常见内存问题模式

| 模式 | Snapshot 特征 | 根因 |
|---|---|---|
| 稳定高内存 | optimizer state 占 2×参数大小 | Adam/AdamW 正常行为 |
| 每 step 内存递增 | live allocations 持续增长 | tensor 引用未释放 / 计算图泄漏 |
| 突然 OOM | 某个 op 分配了巨大 activation | batch size 太大或 sequence 太长 |
| caching allocator 碎片 | `reserved` 远大于 `allocated` | 大小不一的 tensor 交替分配/释放 |

## 5. 性能瓶颈分类：完整方法论

### 5.1 五种瓶颈

| 瓶颈类型 | 含义 | Profiler 特征 | 优化方向 |
|---|---|---|---|
| **Compute bound** | GPU 算力是上限 | GPU 利用率高、CUDA time 接近理论最优 | 换更快 dtype（FP16/FP8）、Tensor Core |
| **Memory bound** | HBM 带宽是上限 | 大量 element-wise op、bandwidth utilization 高 | Kernel fusion、FlashAttention |
| **Launch bound / CPU overhead** | CPU 发 kernel 太慢 | 大量小 kernel、kernel 间 gap 大、GPU idle | torch.compile、CUDA Graph |
| **Communication bound** | 集合通信是上限 | NCCL kernel 占时间比例大、计算通信不 overlap | overlap、减少通信量（gradient compression）|
| **Data loading bottleneck** | 数据准备是上限 | step 开头 GPU idle、dataloader 时间长 | pin_memory、num_workers、prefetch |

### 5.2 分类流程

```
1. 先看 GPU utilization：
   - 高 (>80%) → 进入 compute/memory bound 分析
   - 低 (<50%) → 进入 idle 原因分析

2. GPU 利用率高时：
   - 查 top op：如果主要是 matmul → compute bound
   - 如果主要是 element-wise/norm/softmax → memory bound

3. GPU 利用率低时：
   - 查 idle 段出现在哪里：
     - step 开头 → data loading
     - kernel 之间均匀分布 → launch bound
     - 和 NCCL 同步相关 → communication bound
```

### 5.3 Roofline 判断

```python
# 判断一个 op 的瓶颈
def classify_from_roofline(flops, bytes_accessed, peak_flops, peak_bandwidth):
    """
    peak_flops: GPU 算力上限 (如 H100 TF32: 989 TFLOPS)
    peak_bandwidth: HBM 带宽上限 (如 H100: 3.35 TB/s)
    """
    arithmetic_intensity = flops / bytes_accessed  # FLOPs/Byte
    ridge_point = peak_flops / peak_bandwidth      # 分界点

    if arithmetic_intensity > ridge_point:
        return "compute_bound"
    else:
        return "memory_bound"
```

## 6. 什么时候用哪个工具

| 你的问题 | 用哪个工具 |
|---|---|
| 哪个 op 最耗时？ | torch.profiler |
| GPU 是否在 idle？idle 在哪里？ | nsys |
| 通信和计算是否 overlap？ | nsys |
| 不同 rank 是否负载均衡？ | nsys（多 rank timeline 对比） |
| OOM 的根因是什么？ | torch memory profiler |
| activation vs optimizer state 谁占内存多？ | torch memory profiler |
| 某个 op 的 shape 是否合理？ | torch.profiler (record_shapes=True) |
| 整体 GPU 利用率？ | nsys 或 nvidia-smi dmon |

## 7. Profiling 的开销和最佳实践

### 7.1 开销

- torch.profiler：开启后训练速度降低 10-30%（record_shapes 和 with_stack 更重）。
- nsys：开销较小（<5%），但生成的文件可能很大。
- memory profiler：开销中等，主要增加每次分配的记录成本。

### 7.2 最佳实践

1. **先用 torch.profiler 做粗筛**：找到 top op 和整体时间分布。
2. **怀疑 GPU idle 时切 nsys**：确认 kernel 间隔和通信 overlap。
3. **遇到 OOM 时用 memory profiler**：定位分配大户。
4. **采集时用 warmup**：跳过前几步（编译、JIT、CUDA context 初始化）。
5. **控制采集步数**：3-5 步足够，太多步会生成巨大的 trace 文件。
6. **线上不用 profiler**：profiler 有开销，只在调优时使用。

## 8. 小结

| 工具 | 粒度 | 看什么 | 不能看什么 |
|---|---|---|---|
| torch.profiler | Op 级别 | op 耗时、shape、memory | 真实 GPU timeline、通信 overlap |
| nsys | 系统级 | GPU/CPU timeline、kernel gap、NCCL | 具体 op shape、Python 代码位置 |
| memory profiler | 分配级别 | 谁分配了显存、何时释放 | 运行时间、计算瓶颈 |

三者互补，完整的性能分析通常需要结合使用。
