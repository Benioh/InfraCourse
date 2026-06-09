# L02.7 · Profiler 与性能分析

这一讲解决一个工程核心问题：当训练或推理慢了，你怎么知道时间花在哪里？仅靠 `time.time()` 和直觉是不够的——你需要 profiler 工具来回答：是哪个 op 慢了？是 GPU 在算还是在等？是 compute bound 还是 memory bound？通信和计算有没有 overlap？

本讲涵盖三个层次的 profiler：torch.profiler（op 级别）、nsys/Nsight Systems（系统级 timeline）和 torch memory profiler（显存分配追踪）。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认本讲在课程中的位置。
2. 读 [lecture.md](lecture.md)：三种 profiler 的使用场景、输出解读和实战技巧。
3. 读 [source_walkthrough.md](source_walkthrough.md)：跟读 profiler 分析实验代码。
4. 做 quiz：确认你能看懂 profiler 输出、判断瓶颈类型。
5. 做 patch：实现 profiler 数据的解析和瓶颈分类函数。
6. 跑 smoke：生成一次完整的 profiler 分析证据。
7. 填写 [outputs/profiler_report_template.md](outputs/profiler_report_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Performance analysis & optimization |
| 它解决什么问题 | 系统地定位性能瓶颈：op 耗时、GPU idle、内存分配、通信 overlap |
| 它连接哪些证据 | torch profiler trace（JSON/Chrome trace）、nsys timeline（.nsys-rep）、memory snapshot |
| 它连接哪些源码 | `patch/reference/profiler_analyzer.py`、`scripts/profile_training_step.py`、`scripts/analyze_nsys_timeline.py` |
| lab 检验什么 | profiler 数据解析、瓶颈分类、GPU idle 识别、通信 overlap 判断 |

## 你会学到什么

### torch.profiler
- 如何用 `torch.profiler.profile()` 收集训练循环的 op 级别时间线。
- 如何区分 CPU Time 和 CUDA Time（为什么 CPU Time 大不代表慢）。
- 如何从 profiler 输出中找到最耗时的 op（`key_averages()`）。
- 如何判断 op 的 shape 是否合理（小 shape 暗示 padding/不必要的分拆）。
- 如何识别"大量小 op"模式（CPU overhead / launch bound）。
- 如何看 memory 使用峰值和趋势。

### nsys / Nsight Systems
- nsys 和 torch.profiler 的区别：nsys 是系统级工具，看的是 CPU-GPU 真实 timeline。
- 如何从 nsys timeline 中识别 GPU idle（kernel 之间的空白段）。
- 如何从 nsys 中看到 kernel launch 间隔（判断是否 launch bound）。
- 如何在 nsys 中识别 NCCL 通信 kernel 及其耗时。
- 如何判断计算和通信是否 overlap（两条 stream 上的 kernel 是否时间重叠）。
- 如何发现不同 rank 之间的负载不均（straggler effect）。

### torch memory profiler
- 如何用 `torch.cuda.memory._record_memory_history()` 追踪每次分配。
- 如何从 memory snapshot 中判断：activation、optimizer state、gradient 各占多少。
- 如何识别 tensor 未释放（内存泄漏）。
- 如何识别保存了计算图导致的内存膨胀。
- 如何区分 PyTorch caching allocator 持有的内存和真正"在用"的内存。

### 性能瓶颈分类方法论
- **Compute bound**：roofline model，Tensor Core 利用率。
- **Memory bound**：HBM bandwidth utilization，L2 cache hit rate。
- **CPU overhead / Launch bound**：kernel launch 间隔大、GPU idle。
- **Communication bound**：NCCL kernel 占总时间比例高。
- **Data loading bottleneck**：GPU idle 在 step 开始处（等数据）。

## Patch 闭环

```bash
cat labs/l02.7_profiler_analysis/patch/task.md
$EDITOR labs/l02.7_profiler_analysis/patch/starter/profiler_analyzer.py
make patch-test M=l02.7_profiler_analysis
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_parse_op_summary` | 能从 profiler 数据中提取 top-K 耗时 op |
| `test_classify_bottleneck_compute` | GPU 利用率高 + 大 matmul → compute bound |
| `test_classify_bottleneck_memory` | 带宽利用率高 + element-wise → memory bound |
| `test_classify_bottleneck_launch` | 大量小 kernel + GPU idle → launch bound |
| `test_detect_gpu_idle` | 能从 timeline 数据中识别 GPU idle 段 |
| `test_check_comm_overlap` | 能判断计算和通信是否 overlap |
| `test_memory_breakdown` | 能从 memory snapshot 数据中分类各组件占用 |

## Smoke 闭环

```bash
python labs/l02.7_profiler_analysis/scripts/run_smoke.py \
  --config configs/4090_debug.yaml \
  --mode smoke
```

smoke 会写出：

```text
runs/mini_infra/l02.7_profiler_analysis/<run-id>/
├── command.sh
├── config.resolved.yaml
├── metrics.jsonl
├── report.md
└── artifacts/
    ├── profiler_summary.json
    ├── bottleneck_analysis.json
    ├── gpu_idle_segments.json
    └── memory_breakdown.json
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/profiler_report_template.md](outputs/profiler_report_template.md) | 记录一次真实 profiling 的分析结论 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速复习 profiler 使用流程 |

## 进入下一讲

`make patch-test M=l02.7_profiler_analysis` 通过后，进入 [L03 Manual DDP](../l03_nccl_ddp_smoke/README.md)。后面的 DDP、TP、FSDP 等 lab 中你都会用到本讲的 profiler 技巧来验证 overlap 和瓶颈。
