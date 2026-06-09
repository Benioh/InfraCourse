"""GPU 执行模型探测 — 参考实现"""

import time
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


def describe_dispatch_chain(op_name: str) -> list[str]:
    return ["python_call", "cpp_dispatch", "kernel_launch", "gpu_execute"]


def measure_async_gap(matrix_size: int, device: str) -> dict:
    if torch is None:
        raise RuntimeError("torch is not installed")

    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)

    _ = torch.matmul(A, B)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    _ = torch.matmul(A, B)
    cpu_time_ms = (time.time() - start) * 1000

    if device == "cuda":
        torch.cuda.synchronize()
        sync_time_ms = (time.time() - start) * 1000
        is_async = sync_time_ms > cpu_time_ms * 2
    else:
        sync_time_ms = cpu_time_ms
        is_async = False

    return {
        "cpu_time_ms": cpu_time_ms,
        "sync_time_ms": sync_time_ms,
        "is_async": is_async,
    }


def classify_memory_hierarchy() -> list[dict]:
    return [
        {"name": "register", "scope": "per_thread", "relative_bandwidth": 30},
        {"name": "shared_memory", "scope": "per_sm", "relative_bandwidth": 30},
        {"name": "l2_cache", "scope": "global", "relative_bandwidth": 4},
        {"name": "hbm", "scope": "global", "relative_bandwidth": 1},
    ]


def classify_op_bottleneck(op_type: str, m: int, n: int, k: int) -> dict:
    if op_type == "matmul":
        flops = 2 * m * n * k
        bytes_accessed = (m * k + k * n + m * n) * 4
    elif op_type == "elementwise":
        flops = m * n
        bytes_accessed = (m * n + m * n) * 4
    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    arithmetic_intensity = flops / bytes_accessed
    bottleneck = "compute_bound" if arithmetic_intensity > 100 else "memory_bound"

    return {
        "op_type": op_type,
        "flops": flops,
        "bytes_accessed": bytes_accessed,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
    }
