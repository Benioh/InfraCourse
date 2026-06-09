"""Stream overlap demo: 演示不同 CUDA stream 上的 kernel 并发执行。

用法：
    python labs/l01.5_gpu_execution_model/scripts/stream_overlap_demo.py

需要 CUDA GPU 可用。
"""

import time
import torch


def sequential_execution(A, B, num_ops=20):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_ops):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000


def two_stream_execution(A, B, num_ops=20):
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_ops):
        stream = s1 if i % 2 == 0 else s2
        with torch.cuda.stream(stream):
            _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    print("=== Stream Overlap Demo ===\n")
    print("如果单个 matmul 不占满 GPU，两个 stream 可以并发执行。\n")

    for size in [256, 512, 1024, 2048, 4096]:
        A = torch.randn(size, size, device="cuda")
        B = torch.randn(size, size, device="cuda")

        _ = torch.matmul(A, B)
        torch.cuda.synchronize()

        t_seq = sequential_execution(A, B)
        t_overlap = two_stream_execution(A, B)
        speedup = t_seq / t_overlap if t_overlap > 0 else 1.0

        print(f"Matrix {size}x{size}:")
        print(f"  Sequential (1 stream):  {t_seq:.2f} ms")
        print(f"  Two streams:            {t_overlap:.2f} ms")
        print(f"  Speedup:                {speedup:.2f}x")
        print()

    print("Takeaway:")
    print("  - 小矩阵：GPU 有余量，两个 stream 能真正 overlap → speedup > 1")
    print("  - 大矩阵：单个 matmul 已占满 SM，stream 并发无法提速")
    print("  - 这就是为什么 DDP overlap 对通信有效（通信用的是 NVLink/网络，不抢 SM）")


if __name__ == "__main__":
    main()
