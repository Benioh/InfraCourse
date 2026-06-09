"""Sync overhead demo: 演示 torch.cuda.synchronize() 对性能的影响。

用法：
    python labs/l01.5_gpu_execution_model/scripts/sync_overhead_demo.py

需要 CUDA GPU 可用。CPU-only 环境会跳过。
"""

import time
import torch


def benchmark_with_sync(A, B, num_iters=100):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = torch.matmul(A, B)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed / num_iters * 1000


def benchmark_without_sync(A, B, num_iters=100):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed / num_iters * 1000


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    print("=== Sync Overhead Demo ===\n")

    for size in [256, 1024, 4096]:
        A = torch.randn(size, size, device="cuda")
        B = torch.randn(size, size, device="cuda")

        _ = torch.matmul(A, B)
        torch.cuda.synchronize()

        t_sync = benchmark_with_sync(A, B)
        t_nosync = benchmark_without_sync(A, B)
        overhead_pct = (t_sync - t_nosync) / t_nosync * 100

        print(f"Matrix {size}x{size}:")
        print(f"  With sync every iter:  {t_sync:.3f} ms/iter")
        print(f"  Sync only at end:      {t_nosync:.3f} ms/iter")
        print(f"  Overhead from sync:    {overhead_pct:.1f}%")
        print()

    print("Takeaway: 对于小 kernel，过度 sync 的开销可能比计算本身还大。")
    print("工程实践：只在需要结果时才 sync（log、checkpoint、benchmark）。")


if __name__ == "__main__":
    main()
