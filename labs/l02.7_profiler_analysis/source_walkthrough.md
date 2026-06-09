# L02.7 源码带读：Profiler 分析实验

## 1. `scripts/profile_training_step.py`

### 核心逻辑

```python
def profile_step(model, dataloader, device):
    """用 torch.profiler 采集一个 training step 的 op 信息"""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        batch = next(iter(dataloader))
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        loss = output.loss
        loss.backward()

    return prof.key_averages()
```

### 你应该观察到什么

- `key_averages()` 按 op 名称聚合，展示每种 op 的总时间和平均时间。
- 大模型中 `aten::mm` 和 `aten::addmm` 通常占 CUDA time 的 60-80%。
- `aten::copy_` 如果排名靠前，说明有大量数据搬运（device 间或 dtype 转换）。
- `record_shapes=True` 让你看到每个 op 的输入 shape——小 shape 暗示低效。

## 2. `scripts/analyze_nsys_timeline.py`

### 核心逻辑

这个脚本模拟对 nsys 输出的分析（不依赖真实 nsys，用模拟数据）：

```python
def analyze_timeline(kernel_events):
    """从 kernel event 列表中提取性能指标"""
    total_time = kernel_events[-1]["end_us"] - kernel_events[0]["start_us"]
    compute_time = sum(e["end_us"] - e["start_us"] for e in kernel_events)
    idle_time = total_time - compute_time
    gpu_utilization = compute_time / total_time

    gaps = []
    for i in range(len(kernel_events) - 1):
        gap = kernel_events[i+1]["start_us"] - kernel_events[i]["end_us"]
        gaps.append(gap)

    return {
        "total_time_us": total_time,
        "compute_time_us": compute_time,
        "idle_time_us": idle_time,
        "gpu_utilization": gpu_utilization,
        "avg_kernel_gap_us": sum(gaps) / len(gaps) if gaps else 0,
        "max_kernel_gap_us": max(gaps) if gaps else 0,
    }
```

### 你应该观察到什么

- `gpu_utilization` < 0.6 → GPU 有大量 idle，需要找原因。
- `max_kernel_gap_us` > 1000 → 某处有大段 idle，可能是 sync point 或 data loading。
- `avg_kernel_gap_us` > 50 → kernel launch overhead 累积，考虑 fusion。

## 3. 阅读顺序

1. 先看 `profile_training_step.py`：理解如何采集和解读 op 级数据。
2. 再看 `analyze_nsys_timeline.py`：理解如何从 timeline 数据中提取指标。
3. 带着这些理解去做 patch：你的 `profiler_analyzer.py` 就是把这些分析程序化。

## 4. 和后续 Lab 的联系

| 本讲能力 | 后续 Lab 中的应用 |
|---|---|
| Op 耗时 top-K | L04 确认 Triton kernel 是否比 eager 快 |
| 瓶颈分类 | L11 确认 DDP bucket 大小是否合适 |
| GPU idle 检测 | L03/L11 验证 backward-allreduce overlap |
| Memory breakdown | L06 验证 AC 是否真的省了 activation 内存 |
| nsys 解读 | L05 验证 TP 的 all-reduce 是否 overlap |
