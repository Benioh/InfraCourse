# L02.7 Patch · Profiler 数据解析与瓶颈分类

> 实现 4 个函数，把 profiler 原始数据变成可操作的性能结论。

## 你要改的文件

`labs/l02.7_profiler_analysis/patch/starter/profiler_analyzer.py`

只改这一个文件。

## 任务背景

真实场景中 profiler 会产出大量数据（几百个 op、几千个 kernel event）。你不能每次都手动看 trace 文件——需要程序化地提取关键信息并自动判断瓶颈类型。本关 patch 让你实现这种自动分析能力。

## 四个要实现的函数

### 1. `parse_op_summary(events: list[dict], top_k: int = 5) -> list[dict]`

从 profiler event 列表中提取 top-K 耗时 op。

**输入**：
- `events`: profiler 数据列表，每个 dict 包含：
  - `name`: `str` — op 名称（如 `"aten::mm"`）
  - `cpu_time_us`: `float` — CPU 侧时间（微秒）
  - `cuda_time_us`: `float` — GPU 侧时间（微秒）
  - `calls`: `int` — 调用次数
  - `input_shapes`: `list` — 输入 shape
- `top_k`: 返回前 K 个

**返回值**：`list[dict]`，按 `cuda_time_us` 从大到小排列，每个 dict 包含：
- `name`: op 名称
- `cuda_time_us`: GPU 总时间
- `calls`: 调用次数
- `avg_cuda_time_us`: `cuda_time_us / calls`
- `pct`: 该 op 的 cuda_time 占所有 op 总 cuda_time 的百分比（0-100 的 float）

### 2. `classify_bottleneck(profile_stats: dict) -> dict`

根据整体 profiling 统计数据判断瓶颈类型。

**输入 `profile_stats`**：
- `gpu_utilization`: `float` — GPU 利用率（0-1）
- `top_op_type`: `str` — 最耗时 op 的类型：`"matmul"`, `"elementwise"`, `"comm"`, `"other"`
- `avg_kernel_gap_us`: `float` — kernel 之间的平均间隔（微秒）
- `comm_pct`: `float` — 通信时间占比（0-100）
- `data_wait_pct`: `float` — step 开头 GPU idle 占比（0-100）

**返回 dict**：
- `bottleneck`: `str` — 五选一：`"compute_bound"`, `"memory_bound"`, `"launch_bound"`, `"communication_bound"`, `"data_loading"`
- `confidence`: `str` — `"high"` 或 `"medium"`
- `evidence`: `str` — 一句话解释判断依据

**分类逻辑**（按优先级检查）：
1. 如果 `data_wait_pct > 20` → `"data_loading"`（confidence: high）
2. 如果 `comm_pct > 30` 且 `gpu_utilization < 0.7` → `"communication_bound"`（high）
3. 如果 `avg_kernel_gap_us > 50` 且 `gpu_utilization < 0.6` → `"launch_bound"`（high）
4. 如果 `gpu_utilization > 0.8` 且 `top_op_type == "matmul"` → `"compute_bound"`（high）
5. 如果 `gpu_utilization > 0.8` 且 `top_op_type == "elementwise"` → `"memory_bound"`（high）
6. 否则 → `"compute_bound"`（confidence: medium）

### 3. `detect_gpu_idle_segments(timeline: list[dict], threshold_us: float = 100.0) -> list[dict]`

从 timeline 数据中识别 GPU idle 段（kernel 之间的间隙大于阈值）。

**输入**：
- `timeline`: 按时间排序的 kernel event 列表，每个 dict 包含：
  - `start_us`: `float` — kernel 开始时间（微秒）
  - `end_us`: `float` — kernel 结束时间（微秒）
  - `name`: `str` — kernel 名称
- `threshold_us`: 只报告大于此阈值的 idle 段

**返回值**：`list[dict]`，每个 idle 段包含：
- `start_us`: idle 段开始时间
- `end_us`: idle 段结束时间
- `duration_us`: `end_us - start_us`
- `before_kernel`: 前一个 kernel 的 name
- `after_kernel`: 后一个 kernel 的 name

### 4. `memory_breakdown(allocations: list[dict]) -> dict`

从 memory allocation 记录中分类各组件的内存占用。

**输入**：
- `allocations`: 分配记录列表，每个 dict 包含：
  - `size_bytes`: `int` — 分配大小
  - `category`: `str` — 分类：`"parameter"`, `"gradient"`, `"optimizer_state"`, `"activation"`, `"other"`
  - `is_live`: `bool` — 是否仍在使用

**返回 dict**：
- `total_live_bytes`: `int` — 所有 live 分配的总大小
- `breakdown`: `dict[str, int]` — 每个 category 的 live 字节数
- `breakdown_pct`: `dict[str, float]` — 每个 category 占 total 的百分比（0-100）
- `largest_category`: `str` — 占用最多的 category

**注意**：只统计 `is_live == True` 的分配。如果 `total_live_bytes == 0`，所有百分比为 0，`largest_category` 为 `"none"`。

## 测试覆盖

| 类别 | 测试 | 通过条件 |
|---|---|---|
| OpSummary | `test_parse_op_summary` | top-K 正确、排序正确、pct 之和 ≤ 100 |
| Bottleneck | `test_classify_bottleneck_compute` | matmul + high util → compute_bound |
| Bottleneck | `test_classify_bottleneck_memory` | elementwise + high util → memory_bound |
| Bottleneck | `test_classify_bottleneck_launch` | large gap + low util → launch_bound |
| GpuIdle | `test_detect_gpu_idle` | 正确识别超过阈值的 idle 段 |
| Overlap | `test_check_comm_overlap` | 此测试通过 timeline 数据判断重叠（集成在 idle 检测中） |
| Memory | `test_memory_breakdown` | 正确分类、百分比之和约等于 100 |

跑测试：

```bash
make patch-test M=l02.7_profiler_analysis
```

## 卡住怎么办

1. 重读 lecture.md 第 5 节（瓶颈分类方法论）。
2. `make patch-hint M=l02.7_profiler_analysis`
3. `make patch-show-solution M=l02.7_profiler_analysis`

## 进入下一关

`make patch-test M=l02.7_profiler_analysis` 全绿后，进入 [L03 Manual DDP](../l03_nccl_ddp_smoke/README.md)。
