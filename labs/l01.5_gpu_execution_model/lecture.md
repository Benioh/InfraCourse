# L01.5 讲义：GPU 执行模型与硬件基础

这一讲解决一个根本问题：当你写 `y = torch.matmul(A, B)` 时，从 Python 解释器到 GPU 上的晶体管，中间到底发生了什么？不理解这条链路，后面看 profiler 结果时你分不清时间花在了 CPU dispatch 还是 GPU compute；写 kernel 时你不知道为什么有些写法快有些慢；调优时你判断不了瓶颈在哪一层。

## 1. 学完要能回答什么

1. PyTorch 一个 op（如 `torch.matmul`）从 Python 调用到 GPU 执行完，经过哪些阶段？
2. 为什么 `torch.matmul` 返回了不代表 GPU 已经算完？
3. `torch.cuda.synchronize()` 做了什么？什么时候该用、什么时候不该用？
4. GPU 的 Thread、Warp、Block、Grid 是什么关系？
5. SM 是什么？一张 GPU 有多少个 SM？
6. HBM、SRAM（Shared Memory）、Register 的带宽和容量差多少个数量级？
7. 什么是 Tensor Core？它和 CUDA Core 的区别？
8. CUDA Stream 是什么？为什么默认 stream 可能成为瓶颈？
9. 如何判断一个 kernel 是 compute bound 还是 memory bound？
10. 为什么减少小 kernel 和减少不必要的 sync 都能提升性能？

## 2. PyTorch Op Dispatch 全链路

### 2.1 从 Python 到 C++

当你写：

```python
C = torch.matmul(A, B)
```

Python 侧发生的事：

1. **Python 函数调用**：`torch.matmul` 是一个 Python 函数，它调用 `torch._C._VariableFunctions.matmul`。
2. **Dispatcher**：PyTorch 的 C++ dispatcher 根据 tensor 的 device（cpu/cuda）、dtype、是否需要 autograd 等信息，找到正确的 kernel 实现。
3. **ATen operator**：dispatcher 把调用路由到 ATen 层的 `at::matmul`，最终调用 CUBLAS（对于 CUDA tensor 的 matmul）。

整个链路：

```
Python torch.matmul(A, B)
    → torch._C (pybind11 bridge)
        → ATen Dispatcher (根据 device/dtype 选 kernel)
            → CUDA implementation (cublas, cutlass, 或自定义 kernel)
                → cudaLaunchKernel (CUDA Runtime API)
                    → GPU 硬件执行
```

### 2.2 GPU 异步执行模型

这是理解 GPU 编程最关键的一点：**CPU 和 GPU 是异步执行的。**

```python
import torch, time

A = torch.randn(4096, 4096, device='cuda')
B = torch.randn(4096, 4096, device='cuda')

start = time.time()
C = torch.matmul(A, B)  # ← CPU 这一行几乎瞬间返回！
cpu_time = time.time() - start
# cpu_time ≈ 0.0001s，但 GPU 可能还在算

torch.cuda.synchronize()  # ← 等 GPU 真正算完
total_time = time.time() - start
# total_time ≈ 0.003s，这才是真实耗时
```

**为什么是异步的？**

CPU 只负责把 kernel 的"指令"（launch 参数）提交到 GPU 的命令队列。提交完 CPU 就继续执行下一行 Python。GPU 在自己的节奏上执行这些 kernel。

这意味着：
- 不加 `synchronize()` 测的时间是 **CPU dispatch 时间**，不是 GPU 计算时间。
- 连续提交多个 op，它们在 GPU 上可以"流水线"执行，CPU 不用等。
- `synchronize()` 是一个 barrier：CPU 停下来，等 GPU 完成所有已提交的 kernel。

### 2.3 CPU Overhead 与 GPU Idle

两种性能杀手：

**CPU Overhead（Launch Bound）**：
```python
# 坏例子：1000 个小 op，每个只算几微秒
for i in range(1000):
    x = x + 1  # 每次都要走一遍 Python → dispatch → launch
```

每次 launch 一个 kernel，CPU 侧要花 ~5-20μs 做 dispatch + launch。如果 kernel 本身只需要 1μs，GPU 大部分时间在等 CPU 发新任务。

**解决办法**：
- Kernel fusion：把多个小 op 合成一个大 kernel。
- `torch.compile`：自动做 fusion。
- CUDA Graph：把一连串 kernel 录制下来，replay 时一次性提交。

**过度 Sync 导致 GPU Idle**：
```python
# 坏例子：每个 op 后面都 sync
for layer in model.layers:
    x = layer(x)
    torch.cuda.synchronize()  # ← 强制等 GPU 算完，破坏流水线
```

正确做法是只在需要拿结果的时候才 sync（比如打 loss log、做 checkpoint）。

### 2.4 正确测量 GPU 时间

```python
# 方法 1：synchronize + time.time()
torch.cuda.synchronize()
start = time.time()
output = model(input)
torch.cuda.synchronize()
elapsed = time.time() - start

# 方法 2：CUDA Event（更精确，不阻塞 CPU pipeline）
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
output = model(input)
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

方法 2 更好：Event 记录的是 GPU 时间线上的时间戳，不受 CPU 排队影响。

## 3. GPU 硬件执行层级

### 3.1 CUDA 编程模型：Thread → Warp → Block → Grid

当一个 CUDA kernel launch 时，程序员指定 Grid 和 Block 的大小：

```
kernel<<<grid_size, block_size>>>(args...)
```

层级关系：

```
Grid（一次 kernel launch 的所有线程）
├── Block 0
│   ├── Warp 0 (thread 0-31)
│   ├── Warp 1 (thread 32-63)
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── ...
```

**关键数字**：
- 1 Warp = 32 Threads（这是硬件定义的，不可配置）
- 1 Block 最多 1024 Threads
- Block 之间**完全独立**，不能通信（除非通过 global memory）
- Warp 内的 32 个 thread 执行**相同指令**（SIMT = Single Instruction, Multiple Threads）

### 3.2 SM (Streaming Multiprocessor)

SM 是 GPU 的基本计算单元。每个 SM 包含：
- 多个 CUDA Core（做 FP32/INT32 运算）
- Tensor Core（做矩阵乘法）
- Warp Scheduler（调度 warp 执行）
- Register File（寄存器，per-thread）
- Shared Memory / L1 Cache（per-SM，block 内共享）

**Block 被调度到 SM 上执行。** 一个 SM 可以同时跑多个 Block（如果资源够）。

各代 GPU 的 SM 数量：
| GPU | SM 数量 | CUDA Cores | Tensor Cores |
|---|---|---|---|
| RTX 4090 | 128 | 16384 | 512 |
| A100 | 108 | 6912 | 432 |
| H100 | 132 | 16896 | 528 |
| H200 | 132 | 16896 | 528 |

### 3.3 GPU 内存层级

这是理解 memory bound 的关键：

```
Register          ← per-thread, ~20TB/s 等效, 容量最小 (256KB/SM)
    ↓
Shared Memory     ← per-SM (block 内共享), ~100TB/s 等效, 容量小 (164-228KB/SM on H100)
/ L1 Cache
    ↓
L2 Cache          ← 全局共享, ~12TB/s (H100), 50MB
    ↓
HBM (Global Mem)  ← 全局, 3.35TB/s (H100 80GB), 容量最大
```

**数量级感受**：
- HBM 带宽 ≈ 3TB/s（H100）
- Shared Memory 带宽 ≈ 100TB/s（约 HBM 的 30 倍）
- Register 访问无延迟

**工程意义**：
- 如果一个 kernel 需要反复读写 global memory，它就是 **memory bound**。
- FlashAttention 的核心优化就是把中间结果放在 SRAM（Shared Memory），避免写回 HBM 再读回来。
- Kernel fusion 的本质：多个 op 共享中间结果在 register/shared memory 中，减少 HBM 往返。

### 3.4 Tensor Core

Tensor Core 是专门做矩阵乘法的硬件单元。

**普通 CUDA Core**：一个 cycle 做一次 FMA（fused multiply-add），即 a×b+c。
**Tensor Core**：一个 cycle 做一个小矩阵乘法（如 4×4×4 或 16×8×16），吞吐高 10-20 倍。

使用 Tensor Core 的条件：
- 数据类型：FP16、BF16、TF32、FP8、INT8 等（不支持纯 FP32）
- 矩阵维度通常需要是 8 或 16 的倍数
- PyTorch 中通过 `torch.matmul` + FP16/BF16 tensor 自动使用（CUBLAS 路径）
- AMP（`torch.cuda.amp.autocast`）就是为了让 matmul 走 Tensor Core

**性能数字（H100 SXM）**：
| 精度 | CUDA Core 算力 | Tensor Core 算力 |
|---|---|---|
| FP32 | 67 TFLOPS | - |
| TF32 | - | 989 TFLOPS |
| FP16/BF16 | - | 1979 TFLOPS |
| FP8 | - | 3958 TFLOPS |

这就是为什么混合精度训练（AMP）能让训练快 2-3 倍——不只是因为节省了显存，更是因为 Tensor Core 的算力远超 CUDA Core。

## 4. CUDA Stream

### 4.1 Stream 是什么

Stream 是 GPU 上的**有序命令队列**。同一 stream 内的操作保证按提交顺序执行；不同 stream 内的操作可以并发执行。

```python
import torch

# 默认 stream：所有 PyTorch op 默认在这里执行
A = torch.randn(1024, 1024, device='cuda')
B = torch.matmul(A, A)  # 在 default stream

# 创建新 stream
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

with torch.cuda.stream(s1):
    C = torch.matmul(A, A)  # 在 s1 上执行

with torch.cuda.stream(s2):
    D = torch.matmul(A, A)  # 在 s2 上执行，可以和 s1 并发
```

### 4.2 Stream 的用途

1. **计算与通信 overlap**：DDP 的 gradient all-reduce 就是在一个 stream 做通信，同时另一个 stream 继续做后面层的 backward。
2. **数据传输与计算 overlap**：用一个 stream 做 H2D（CPU→GPU）copy，另一个 stream 做当前 batch 的 compute。
3. **多任务并发**：serving 场景中不同请求可以放在不同 stream 上。

### 4.3 Stream 之间的同步

```python
# Event-based synchronization
event = torch.cuda.Event()

with torch.cuda.stream(s1):
    result = torch.matmul(A, A)
    event.record()  # 在 s1 时间线上标记一个点

with torch.cuda.stream(s2):
    event.wait()  # s2 等 s1 的 event 完成
    # 现在可以安全使用 result
    output = result + 1
```

## 5. Compute Bound vs Memory Bound

判断一个 kernel 性能瓶颈在哪里，是优化的第一步。

### 5.1 Arithmetic Intensity（计算密度）

```
Arithmetic Intensity = FLOPs / Bytes accessed
```

- **高计算密度** → Compute bound（如 matmul：O(N³) 计算，O(N²) 内存访问）
- **低计算密度** → Memory bound（如 element-wise add：O(N) 计算，O(N) 内存访问）

### 5.2 Roofline Model

GPU 有两个"天花板"：
- 算力天花板：如 H100 的 989 TFLOPS (TF32)
- 带宽天花板：如 H100 的 3.35 TB/s

一个 kernel 的实际性能被哪个天花板限制，取决于它的 arithmetic intensity。

**分界点**（ridge point）= 算力 / 带宽 = 989 TFLOPS / 3.35 TB/s ≈ 295 FLOPs/Byte

- 低于 295 FLOPs/Byte → Memory bound
- 高于 295 FLOPs/Byte → Compute bound

### 5.3 常见 Op 的瓶颈分类

| Op | Arithmetic Intensity | 瓶颈 |
|---|---|---|
| MatMul (大矩阵) | O(N) FLOPs/Byte | Compute bound |
| Element-wise (add, relu) | ~0.25 FLOPs/Byte | Memory bound |
| LayerNorm | ~5-10 FLOPs/Byte | Memory bound |
| Softmax | ~5-10 FLOPs/Byte | Memory bound |
| Attention (naive) | 取决于序列长度 | Memory bound (长序列) |
| Reduction (sum, mean) | ~0.5 FLOPs/Byte | Memory bound |

**工程启示**：
- 大 matmul 优化方向：用 Tensor Core（换 dtype）、增大问题规模。
- Memory bound op 优化方向：kernel fusion（减少 HBM 访问次数）、用 shared memory 做中间缓存。
- 这就是为什么 FlashAttention 有效：它把 attention 从 memory bound 变成了计算由 SRAM 服务的形态。

## 6. 减少 CPU-GPU 同步的工程实践

### 6.1 什么操作会触发隐式同步

- `tensor.item()`、`tensor.cpu()`：需要等 GPU 算完才能把值搬到 CPU。
- `print(tensor)`：内部调用 `.item()` 或 `.cpu()`。
- `tensor.numpy()`：必须先 `.cpu()`。
- `if tensor > threshold`：需要把 GPU 上的值拿到 CPU 做判断。

### 6.2 常见陷阱

```python
# 坏：每个 step 都隐式 sync
for batch in dataloader:
    loss = model(batch)
    print(f"loss = {loss.item()}")  # ← 每次都 sync！

# 好：积攒几步再打日志
for i, batch in enumerate(dataloader):
    loss = model(batch)
    if i % 100 == 0:
        print(f"loss = {loss.item()}")
```

### 6.3 和 Profiler 的关系

当你在 torch profiler 或 nsys 中看到 GPU timeline 上有大段空白（GPU idle），首先检查：
1. 是不是有过多的 sync 点？
2. 是不是有 `.item()` / `.cpu()` 在 hot path 上？
3. CPU 端的 dispatch 是否太慢（大量小 kernel）？

这些直觉在 L02（profiler trace）和后面的 Profiler 专题中会反复用到。

## 7. 小结

| 概念 | 一句话 |
|---|---|
| Dispatch 链路 | Python → C++ ATen → Dispatcher → CUDA kernel launch → GPU async execute |
| 异步执行 | CPU 提交 kernel 后立即返回，GPU 自行执行；sync 是 CPU 等 GPU 的 barrier |
| Thread/Warp/Block/SM | 32 threads = 1 warp (SIMT)；多个 warp = 1 block；block 调度到 SM |
| 内存层级 | Register > Shared Memory > L2 > HBM；差 10-30 倍带宽 |
| Tensor Core | 专用矩阵乘单元，FP16/BF16 吞吐比 CUDA Core 高 10-20× |
| CUDA Stream | 有序命令队列，跨 stream 可并发，用 Event 同步 |
| Compute bound | 瓶颈在算力（大 matmul）；优化：用 Tensor Core |
| Memory bound | 瓶颈在带宽（element-wise, norm）；优化：kernel fusion |
| CPU overhead | 瓶颈在 launch（大量小 kernel）；优化：fusion / compile / CUDA Graph |
