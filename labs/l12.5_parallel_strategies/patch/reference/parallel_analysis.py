"""并行策略分析工具 — 参考实现"""


def estimate_training_memory(num_params_billion: float, precision: str = "fp16",
                             optimizer: str = "adam", batch_size: int = 1,
                             seq_len: int = 2048, hidden_size: int = 4096,
                             num_layers: int = 32) -> dict:
    num_params = num_params_billion * 1e9
    bytes_per_param = 2 if precision == "fp16" else 4

    params_bytes = num_params * bytes_per_param
    grads_bytes = num_params * bytes_per_param

    if optimizer == "adam":
        optimizer_bytes = num_params * 12 if precision == "fp16" else num_params * 8
    else:
        optimizer_bytes = num_params * 4

    activation_bytes = batch_size * seq_len * hidden_size * num_layers * 2 * bytes_per_param

    to_gb = lambda b: b / (1024**3)

    return {
        "params_gb": to_gb(params_bytes),
        "grads_gb": to_gb(grads_bytes),
        "optimizer_gb": to_gb(optimizer_bytes),
        "activation_gb": to_gb(activation_bytes),
        "total_gb": to_gb(params_bytes + grads_bytes + optimizer_bytes + activation_bytes),
    }


def compute_comm_volume(operation: str, tensor_size_bytes: int,
                        world_size: int) -> dict:
    if operation == "all_reduce":
        comm_bytes = 2 * tensor_size_bytes * (world_size - 1) // world_size
        formula = "2 * N * (ws-1) / ws"
    elif operation == "all_gather":
        comm_bytes = tensor_size_bytes * (world_size - 1) // world_size
        formula = "N * (ws-1) / ws"
    elif operation == "reduce_scatter":
        comm_bytes = tensor_size_bytes * (world_size - 1) // world_size
        formula = "N * (ws-1) / ws"
    elif operation == "broadcast":
        comm_bytes = tensor_size_bytes
        formula = "N"
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return {
        "operation": operation,
        "tensor_size_bytes": tensor_size_bytes,
        "world_size": world_size,
        "comm_bytes": comm_bytes,
        "formula": formula,
    }


def select_parallel_strategy(num_params_billion: float, num_gpus: int,
                             gpu_memory_gb: float, inter_node: bool = False,
                             seq_len: int = 2048) -> dict:
    mem = estimate_training_memory(num_params_billion, seq_len=seq_len)
    single_gpu_needed = mem["total_gb"]

    if single_gpu_needed < gpu_memory_gb * 0.7:
        return {
            "strategy": "DDP",
            "reason": f"Model fits in single GPU ({single_gpu_needed:.1f}GB < {gpu_memory_gb*0.7:.1f}GB)",
            "memory_per_gpu_gb": single_gpu_needed,
        }

    if single_gpu_needed < gpu_memory_gb * 1.5:
        return {
            "strategy": "ZeRO-2 / FSDP",
            "reason": f"Model barely fits with optimizer sharding ({single_gpu_needed:.1f}GB)",
            "memory_per_gpu_gb": single_gpu_needed / 2,
        }

    if inter_node:
        return {
            "strategy": "PP + DP",
            "reason": f"Model too large ({single_gpu_needed:.1f}GB), inter-node favors PP over TP",
            "memory_per_gpu_gb": single_gpu_needed / num_gpus,
        }

    return {
        "strategy": "TP + FSDP",
        "reason": f"Model too large ({single_gpu_needed:.1f}GB), intra-node NVLink supports TP",
        "memory_per_gpu_gb": single_gpu_needed / num_gpus,
    }


def zero_memory_breakdown(num_params_billion: float, world_size: int,
                          zero_stage: int) -> dict:
    num_params = num_params_billion * 1e9
    to_gb = lambda b: b / (1024**3)

    params_bytes = num_params * 2
    grads_bytes = num_params * 2
    optimizer_bytes = num_params * 12

    ddp_total = params_bytes + grads_bytes + optimizer_bytes

    if zero_stage == 1:
        opt_per_gpu = optimizer_bytes / world_size
        total_per_gpu = params_bytes + grads_bytes + opt_per_gpu
    elif zero_stage == 2:
        grads_per_gpu = grads_bytes / world_size
        opt_per_gpu = optimizer_bytes / world_size
        total_per_gpu = params_bytes + grads_per_gpu + opt_per_gpu
    elif zero_stage == 3:
        params_per_gpu = params_bytes / world_size
        grads_per_gpu = grads_bytes / world_size
        opt_per_gpu = optimizer_bytes / world_size
        total_per_gpu = params_per_gpu + grads_per_gpu + opt_per_gpu
    else:
        raise ValueError(f"Invalid zero_stage: {zero_stage}")

    if zero_stage == 1:
        params_per_gpu_val = params_bytes
        grads_per_gpu_val = grads_bytes
    elif zero_stage == 2:
        params_per_gpu_val = params_bytes
        grads_per_gpu_val = grads_bytes / world_size
    else:
        params_per_gpu_val = params_bytes / world_size
        grads_per_gpu_val = grads_bytes / world_size

    saving = (1 - total_per_gpu / ddp_total) * 100

    return {
        "stage": zero_stage,
        "params_per_gpu_gb": to_gb(params_per_gpu_val),
        "grads_per_gpu_gb": to_gb(grads_per_gpu_val),
        "optimizer_per_gpu_gb": to_gb(opt_per_gpu),
        "total_per_gpu_gb": to_gb(total_per_gpu),
        "saving_vs_ddp_pct": saving,
    }
