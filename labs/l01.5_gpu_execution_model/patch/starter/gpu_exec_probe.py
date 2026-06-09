"""GPU 执行模型探测 — starter 文件

实现 4 个函数，建立从 Python 到 GPU 的执行链路心智模型。
参考 patch/task.md 了解每个函数的详细规格。
"""

import time
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


def describe_dispatch_chain(op_name: str) -> list[str]:
    """返回 PyTorch op 从 Python 到 GPU 执行的 dispatch 阶段列表。

    对于所有 CUDA tensor op，返回 4 个阶段（按执行顺序）：
    python_call → cpp_dispatch → kernel_launch → gpu_execute
    """
    # TODO: 实现此函数
    raise NotImplementedError


def measure_async_gap(matrix_size: int, device: str) -> dict:
    """测量 GPU 异步执行导致的 CPU/GPU 时间差。

    返回 dict 包含：cpu_time_ms, sync_time_ms, is_async
    """
    # TODO: 实现此函数
    raise NotImplementedError


def classify_memory_hierarchy() -> list[dict]:
    """返回 GPU 内存层级信息，从最快到最慢排列。

    每个 dict 包含：name, scope, relative_bandwidth
    4 个层级：register > shared_memory > l2_cache > hbm
    """
    # TODO: 实现此函数
    raise NotImplementedError


def classify_op_bottleneck(op_type: str, m: int, n: int, k: int) -> dict:
    """根据 op 类型和问题规模判断 compute bound 还是 memory bound。

    返回 dict 包含：op_type, flops, bytes_accessed, arithmetic_intensity, bottleneck
    """
    # TODO: 实现此函数
    raise NotImplementedError
