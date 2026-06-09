# Source Reading Card：L01 环境探针

## 主路径

1. `labs/l01_env_conda_cuda/patch/reference/env_probe.py`
   - `collect_python_env()`：只读 `sys` 和 `os.environ`，生成纯 Python 环境快照。
   - `parse_local_rank_mapping()`：把 CVD 和 local rank 映射成物理 GPU。
   - `summarize_drift()`：比较两份环境快照，输出稳定 diff。

2. `scripts/env/check_cuda.py`
   - 记录 torch import、torch version、CUDA runtime、CUDA 可见性、device count 和设备属性。
   - 只在 CUDA 可用时枚举设备。

3. `scripts/env/check_nccl.py`
   - 记录 distributed、NCCL、Gloo 后端是否可用。
   - 不启动多进程，不证明真实通信通过。

4. `scripts/env/collect_env.py`
   - 记录解释器、平台、工作目录、关键环境变量、`nvidia-smi`、`nvcc` 和 `conda` 输出。

5. `labs/l01_env_conda_cuda/scripts/torchrun_hello.py`
   - 读取 `WORLD_SIZE`、`LOCAL_RANK`。
   - GPU 足够时选 NCCL，否则选 Gloo。
   - 输出每个 rank 的身份和 device。

6. `labs/l01_env_conda_cuda/scripts/run_smoke.py`
   - 创建 run 目录。
   - 写 `command.sh`、`config.resolved.yaml`、`prediction.yaml`。
   - 运行所有探针。
   - 写 `metrics.jsonl` 和 `report.md`。

7. `mini_infra/observability/evidence.py`
   - 汇总 run 是否有 command/config/report/metrics/artifacts。
   - 用于后续交付审计。

## 关键结论

- patch 函数必须在没有 torch 的环境里也能运行。
- `nvidia-smi`、`torch.version.cuda`、`torch.cuda.is_available()` 回答的问题不同。
- `CUDA_VISIBLE_DEVICES` 改变进程内逻辑 device 编号。
- `nccl_available=true` 还需要真实 `torchrun` 日志配合判断。
- L01 的 metrics 主要看 `torchrun_ok` 和 `cuda_available`，训练吞吐字段是统一 schema 占位。
- 环境结论必须能回到具体 artifact 字段。

## 自检

- 我能否解释 `collect_python_env()` 为什么不 import torch？
- 我能否手算 CVD 和 `LOCAL_RANK` 的映射？
- 我能否区分 CPU-only validation、本地 GPU smoke 和 H200 cluster smoke？
- 我能否指出后续 DDP hang 时应该回看哪些 L01 artifact？
