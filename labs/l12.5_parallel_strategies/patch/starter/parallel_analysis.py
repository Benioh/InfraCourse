"""并行策略分析工具 — starter 文件"""


def estimate_training_memory(num_params_billion: float, precision: str = "fp16",
                             optimizer: str = "adam", batch_size: int = 1,
                             seq_len: int = 2048, hidden_size: int = 4096,
                             num_layers: int = 32) -> dict:
    """估算模型训练显存需求（GB）。

    precision: "fp16" 或 "fp32"
    optimizer: "adam" 或 "sgd"

    返回 dict: params_gb, grads_gb, optimizer_gb, activation_gb, total_gb
    """
    # TODO: 实现此函数
    raise NotImplementedError


def compute_comm_volume(operation: str, tensor_size_bytes: int,
                        world_size: int) -> dict:
    """计算集合通信的通信量。

    operation: "all_reduce", "all_gather", "reduce_scatter", "broadcast"

    返回 dict: operation, tensor_size_bytes, world_size, comm_bytes, formula
    """
    # TODO: 实现此函数
    raise NotImplementedError


def select_parallel_strategy(num_params_billion: float, num_gpus: int,
                             gpu_memory_gb: float, inter_node: bool = False,
                             seq_len: int = 2048) -> dict:
    """根据模型规模和硬件条件推荐并行策略。

    返回 dict: strategy, reason, memory_per_gpu_gb
    """
    # TODO: 实现此函数
    raise NotImplementedError


def zero_memory_breakdown(num_params_billion: float, world_size: int,
                          zero_stage: int) -> dict:
    """计算 ZeRO-1/2/3 各阶段的显存分布。

    返回 dict: stage, params_per_gpu_gb, grads_per_gpu_gb, optimizer_per_gpu_gb,
               total_per_gpu_gb, saving_vs_ddp_pct
    """
    # TODO: 实现此函数
    raise NotImplementedError
