"""Profiler 数据解析与瓶颈分类 — starter 文件

实现 4 个函数，把 profiler 原始数据变成可操作的性能结论。
参考 patch/task.md 了解每个函数的详细规格。
"""


def parse_op_summary(events: list[dict], top_k: int = 5) -> list[dict]:
    """从 profiler event 列表中提取 top-K 耗时 op。

    按 cuda_time_us 从大到小排列，返回 name/cuda_time_us/calls/avg_cuda_time_us/pct。
    """
    # TODO: 实现此函数
    raise NotImplementedError


def classify_bottleneck(profile_stats: dict) -> dict:
    """根据整体 profiling 统计数据判断瓶颈类型。

    返回 bottleneck/confidence/evidence。
    """
    # TODO: 实现此函数
    raise NotImplementedError


def detect_gpu_idle_segments(timeline: list[dict], threshold_us: float = 100.0) -> list[dict]:
    """从 timeline 数据中识别 GPU idle 段。

    返回超过 threshold_us 的 idle 段信息。
    """
    # TODO: 实现此函数
    raise NotImplementedError


def memory_breakdown(allocations: list[dict]) -> dict:
    """从 memory allocation 记录中分类各组件的内存占用。

    返回 total_live_bytes/breakdown/breakdown_pct/largest_category。
    """
    # TODO: 实现此函数
    raise NotImplementedError
