# L01.5 Patch · GPU 执行模型探测

> 实现 4 个函数，建立从 Python 到 GPU 的执行链路心智模型，并能用代码证明异步执行和硬件层级的关键性质。

## 你要改的文件

`labs/l01.5_gpu_execution_model/patch/starter/gpu_exec_probe.py`

只改这一个文件。其它文件（reference/、tests/）不要动。

## 任务背景

后面做 profiler 分析、kernel 优化、分布式训练调优时，你需要能准确判断：时间花在了 CPU dispatch 还是 GPU compute？瓶颈在算力还是带宽？两个操作能不能 overlap？

本关 patch 让你把这些判断变成可验证的代码。

## 四个要实现的函数

### 1. `describe_dispatch_chain(op_name: str) -> list[str]`

给定一个 PyTorch op 名称（如 `"matmul"`），返回该 op 从 Python 到 GPU 执行的 dispatch 阶段列表。

**返回值**：一个 `list[str]`，按执行顺序包含以下阶段：
- `"python_call"` — Python 函数调用
- `"cpp_dispatch"` — C++/ATen dispatcher 路由
- `"kernel_launch"` — CUDA runtime 提交 kernel 到 GPU
- `"gpu_execute"` — GPU 异步执行 kernel

对于所有 CUDA tensor op，返回这 4 个阶段（顺序固定）。

### 2. `measure_async_gap(matrix_size: int, device: str) -> dict`

测量 GPU 异步执行导致的 CPU/GPU 时间差。

**输入**：
- `matrix_size`: 方阵边长（如 4096）
- `device`: `"cuda"` 或 `"cpu"`

**返回 dict 的 key**：
- `cpu_time_ms`: `float` — 不加 sync 的 CPU 端测量时间（毫秒）
- `sync_time_ms`: `float` — 加 sync 后的完整时间（毫秒）
- `is_async`: `bool` — `True` 如果 `sync_time_ms > cpu_time_ms * 2`（证明 GPU 是异步的）

**逻辑**：
1. 创建两个 `matrix_size × matrix_size` 的随机 tensor（在 device 上）。
2. 做一次 warmup matmul（丢弃结果）。
3. 如果 device 是 `"cuda"`，先 `torch.cuda.synchronize()`。
4. 记录 start time，执行 `torch.matmul`，记录 end time → `cpu_time_ms`。
5. 如果 device 是 `"cuda"`，`torch.cuda.synchronize()`，再记录 end time → `sync_time_ms`。
6. 如果 device 是 `"cpu"`，`sync_time_ms = cpu_time_ms`，`is_async = False`。

### 3. `classify_memory_hierarchy() -> list[dict]`

返回 GPU 内存层级信息，从最快到最慢排列。

**返回值**：`list[dict]`，每个 dict 包含：
- `name`: `str` — 层级名称
- `scope`: `str` — 作用域
- `relative_bandwidth`: `int` — 相对带宽（以 HBM 为 1）

必须包含这 4 个层级（按 bandwidth 从高到低）：

| name | scope | relative_bandwidth |
|---|---|---|
| `"register"` | `"per_thread"` | 30 |
| `"shared_memory"` | `"per_sm"` | 30 |
| `"l2_cache"` | `"global"` | 4 |
| `"hbm"` | `"global"` | 1 |

注意：register 和 shared memory 的 relative_bandwidth 都是 30（相对于 HBM），因为 shared memory ~100TB/s vs HBM ~3TB/s。实际 register 更快，但在这个粗粒度模型中我们给相同值。

### 4. `classify_op_bottleneck(op_type: str, m: int, n: int, k: int) -> dict`

根据 op 类型和问题规模判断是 compute bound 还是 memory bound。

**输入**：
- `op_type`: `"matmul"` 或 `"elementwise"`
- `m, n, k`: 问题规模参数
  - 对于 `"matmul"`：A 是 m×k，B 是 k×n → C 是 m×n
  - 对于 `"elementwise"`：操作在一个 m×n 的 tensor 上（k 忽略）

**返回 dict 的 key**：
- `op_type`: 透传输入
- `flops`: `int` — 计算量
  - matmul: `2 * m * n * k`
  - elementwise: `m * n`
- `bytes_accessed`: `int` — 内存访问量（假设 float32 = 4 bytes）
  - matmul: `(m*k + k*n + m*n) * 4`
  - elementwise: `(m*n + m*n) * 4`（读一个 tensor + 写一个 tensor）
- `arithmetic_intensity`: `float` — `flops / bytes_accessed`
- `bottleneck`: `str` — `"compute_bound"` 如果 `arithmetic_intensity > 100`，否则 `"memory_bound"`

**异常**：
- `op_type` 不是 `"matmul"` 或 `"elementwise"` → `raise ValueError`，message 包含 `"op_type"`

## 测试覆盖

`tests/test_patch.py` 里 7 个测试：

| 类别 | 测试 | 通过条件 |
|---|---|---|
| Dispatch | `test_dispatch_chain_stages` | 返回 4 个阶段，顺序正确 |
| Async | `test_async_timing_without_sync` | device=cpu 时 is_async 为 False |
| Async | `test_async_timing_with_sync` | device=cuda 时函数不 crash（GPU 可选） |
| Memory | `test_memory_hierarchy_bandwidth` | 4 个层级，bandwidth 从高到低排列 |
| Memory | `test_memory_hierarchy_scope` | scope 值正确 |
| Bottleneck | `test_compute_vs_memory_bound` | matmul(4096,4096,4096) → compute_bound；elementwise(4096,4096,0) → memory_bound |
| Bottleneck | `test_bottleneck_invalid_op` | 非法 op_type 抛 ValueError |

跑测试：

```bash
make patch-test M=l01.5_gpu_execution_model
```

7 个全绿就过关。

## 卡住怎么办

1. 重读 lecture.md 第 2 节（dispatch 链路）和第 5 节（compute vs memory bound）。
2. `make patch-hint M=l01.5_gpu_execution_model` 列出 TODO + 关键提示。
3. `make patch-show-solution M=l01.5_gpu_execution_model` 打开参考解。

## 配套源码研读

- `scripts/sync_overhead_demo.py` — 对比有无 sync 的时间差异。
- `scripts/stream_overlap_demo.py` — 两个 stream 上的 kernel overlap 实验。

## 进入下一关的前置

`make patch-test M=l01.5_gpu_execution_model` 全绿后，进入 [L02 PyTorch 显存账本](../l02_pytorch_systems/README.md)。
