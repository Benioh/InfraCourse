# L01.5 源码带读：CUDA Stream 与 Sync 实验

本讲的源码带读聚焦于两个实验脚本，帮你建立对 GPU 异步执行的直觉。

## 1. `scripts/sync_overhead_demo.py`

这个脚本演示 sync 对性能的影响。

### 核心逻辑

```python
def benchmark_with_sync(model, input_batch, num_iters=100):
    """每次 forward 后都 sync — 模拟过度同步"""
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        output = model(input_batch)
        torch.cuda.synchronize()  # 每次都等 GPU 完成
    elapsed = time.time() - start
    return elapsed / num_iters

def benchmark_without_sync(model, input_batch, num_iters=100):
    """只在最后 sync 一次 — 正常流水线"""
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        output = model(input_batch)
    torch.cuda.synchronize()  # 只在最后等一次
    elapsed = time.time() - start
    return elapsed / num_iters
```

### 你应该观察到什么

- `benchmark_with_sync` 的平均时间会明显高于 `benchmark_without_sync`。
- 差异来源：每次 sync 都强制 CPU 停下来等 GPU，破坏了 CPU-GPU 流水线。
- 在小模型上差异更明显（因为 GPU 计算本身很快，sync 的等待占比大）。

### 什么时候该 sync

- 测量 GPU 时间时（benchmark）
- 读取 GPU 结果到 CPU 时（`.item()`, `.cpu()` 会隐式 sync）
- checkpoint 之前（确保所有计算完成再保存）
- 打 loss log 时（但应该攒几步再打）

## 2. `scripts/stream_overlap_demo.py`

这个脚本演示两个 stream 上的 kernel 如何 overlap。

### 核心逻辑

```python
def sequential_execution(A, B, num_ops=10):
    """所有 op 在 default stream 上顺序执行"""
    results = []
    for _ in range(num_ops):
        results.append(torch.matmul(A, B))
    torch.cuda.synchronize()
    return results

def overlapped_execution(A, B, num_ops=10):
    """op 分布在两个 stream 上，可能 overlap"""
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    results = []
    for i in range(num_ops):
        stream = s1 if i % 2 == 0 else s2
        with torch.cuda.stream(stream):
            results.append(torch.matmul(A, B))
    torch.cuda.synchronize()
    return results
```

### 你应该观察到什么

- 如果矩阵足够小（单个 matmul 不占满所有 SM），`overlapped_execution` 会更快。
- 如果矩阵很大（单个 matmul 已经占满 GPU），两种方式时间差不多。
- 这验证了 Stream 提供的是并发**可能性**，不是保证。

### 实际工程中的 stream overlap

DDP 的 gradient all-reduce 就是最常见的 stream overlap 应用：
- Backward pass 在 compute stream 上执行
- 当某一层的 gradient 算完后，立即在 comm stream 上启动 all-reduce
- 两个 stream 并发：后面层的 gradient 计算 和 前面层的 all-reduce 同时进行

这就是为什么 DDP 的 bucket 机制要和 stream 配合——bucket 满了就立即发起 all-reduce，不等所有 gradient 都算完。

## 3. 阅读顺序

1. 先运行 `sync_overhead_demo.py`，观察 sync 开销。
2. 再运行 `stream_overlap_demo.py`，观察 overlap 效果。
3. 带着这两个观察去做 quiz 和 patch。

## 4. 和后续 Lab 的联系

| 本讲观察 | 后续 Lab 中的应用 |
|---|---|
| sync 开销 | L02 profiler trace 中 GPU idle 段的解释 |
| stream overlap | L03 DDP backward+allreduce overlap |
| CPU dispatch overhead | L04 为什么 kernel fusion 有效 |
| memory hierarchy | L04 Triton BLOCK_SIZE 选择、L22 FlashAttention 原理 |
