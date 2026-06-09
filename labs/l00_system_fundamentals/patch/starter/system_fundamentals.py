"""系统基础验证工具 — starter"""


def estimate_h2d_transfer_time(size_bytes: int, pinned: bool = False) -> dict:
    """估算 CPU→GPU 数据传输时间。

    pinned=True: 使用 DMA，带宽约 12 GB/s (PCIe Gen4 x16 有效带宽)
    pinned=False: 需要先 copy 到 pinned buffer，有效带宽约 6 GB/s

    返回 dict: size_bytes, pinned, bandwidth_gbps, transfer_time_ms
    """
    # TODO: 实现此函数
    raise NotImplementedError


def identify_comm_bottleneck(src: str, dst: str, data_bytes: int) -> dict:
    """根据通信路径判断瓶颈带宽。

    src/dst: "cpu", "gpu_same_node", "gpu_diff_node"
    带宽假设: PCIe=32GB/s, NVLink=450GB/s, IB_NDR=100GB/s

    返回 dict: path, bandwidth_gbps, transfer_time_ms, bottleneck
    """
    # TODO: 实现此函数
    raise NotImplementedError


def dataloader_throughput(batch_size_bytes: int, num_workers: int,
                          preprocessing_time_ms: float,
                          h2d_time_ms: float) -> dict:
    """计算 DataLoader 的理论吞吐上限。

    假设 workers 完全并行，主进程只做 H2D。
    吞吐受限于 max(preprocessing_time / num_workers, h2d_time)。

    返回 dict: preprocessing_per_batch_ms, h2d_per_batch_ms,
               bottleneck, throughput_batches_per_sec
    """
    # TODO: 实现此函数
    raise NotImplementedError


def explain_pinned_memory() -> dict:
    """返回 pinned memory 的关键知识点。

    返回 dict:
    - what: str — 一句话解释
    - why_faster: str — 为什么比 pageable 快
    - tradeoff: str — 代价
    - pytorch_api: str — PyTorch 中如何使用
    """
    # TODO: 实现此函数
    raise NotImplementedError
