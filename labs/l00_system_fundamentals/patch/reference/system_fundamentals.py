"""系统基础验证工具 — 参考实现"""


def estimate_h2d_transfer_time(size_bytes: int, pinned: bool = False) -> dict:
    if pinned:
        bandwidth_gbps = 12.0
    else:
        bandwidth_gbps = 6.0

    transfer_time_ms = (size_bytes / (bandwidth_gbps * 1e9)) * 1000

    return {
        "size_bytes": size_bytes,
        "pinned": pinned,
        "bandwidth_gbps": bandwidth_gbps,
        "transfer_time_ms": transfer_time_ms,
    }


def identify_comm_bottleneck(src: str, dst: str, data_bytes: int) -> dict:
    if src == "cpu" and "gpu" in dst:
        bandwidth = 32.0
        path = "PCIe"
    elif src == "gpu_same_node" and dst == "gpu_same_node":
        bandwidth = 450.0
        path = "NVLink"
    elif "diff_node" in src or "diff_node" in dst:
        bandwidth = 100.0
        path = "InfiniBand_NDR"
    else:
        bandwidth = 32.0
        path = "PCIe"

    transfer_time_ms = (data_bytes / (bandwidth * 1e9)) * 1000

    return {
        "path": path,
        "bandwidth_gbps": bandwidth,
        "transfer_time_ms": transfer_time_ms,
        "bottleneck": path,
    }


def dataloader_throughput(batch_size_bytes: int, num_workers: int,
                          preprocessing_time_ms: float,
                          h2d_time_ms: float) -> dict:
    prep_per_batch = preprocessing_time_ms / max(num_workers, 1)
    bottleneck_time = max(prep_per_batch, h2d_time_ms)
    throughput = 1000.0 / bottleneck_time if bottleneck_time > 0 else float('inf')

    bottleneck = "preprocessing" if prep_per_batch >= h2d_time_ms else "h2d_transfer"

    return {
        "preprocessing_per_batch_ms": prep_per_batch,
        "h2d_per_batch_ms": h2d_time_ms,
        "bottleneck": bottleneck,
        "throughput_batches_per_sec": throughput,
    }


def explain_pinned_memory() -> dict:
    return {
        "what": "Page-locked memory that cannot be swapped out by the OS",
        "why_faster": "Enables DMA transfer directly from physical RAM to GPU without CPU involvement",
        "tradeoff": "Consumes physical RAM, slow to allocate, excessive pinning can cause system OOM",
        "pytorch_api": "tensor.pin_memory() or DataLoader(pin_memory=True)",
    }
