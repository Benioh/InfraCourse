# Debug Checklist：环境、CUDA 与 torchrun

## 1. 先固定现场

- 保存完整命令：`command.sh`。
- 保存解析后的配置：`config.resolved.yaml`。
- 保存当前 git commit、Python 解释器、Conda 环境、容器镜像和节点信息。
- 保留 `artifacts/check_cuda.json`、`check_nccl.json`、`collect_env.json`、`torchrun_hello.log`。
- 不要只保存终端截图；后续比较需要机器可读字段。

## 2. import 失败

按顺序看：

1. `collect_env.json` 里的 `executable`。
2. `collect_env.json` 里的 `env.PYTHONPATH`。
3. `collect_env.json` 里的 `env.CONDA_PREFIX`。
4. 当前工作目录 `cwd`。
5. 具体 import error。

判断：

- `executable` 不在预期环境：先修启动脚本或 Conda activation。
- `PYTHONPATH` 指向项目内同名目录：先确认 `python -c "import xxx; print(xxx.__file__)"`。
- 依赖缺失：按课程环境文件补依赖，再重新跑 smoke。

## 3. CUDA 不可用

按顺序看：

1. `check_cuda.json.torch_importable`。
2. `check_cuda.json.torch_version`。
3. `check_cuda.json.cuda_version`。
4. `check_cuda.json.cuda_available`。
5. `check_cuda.json.cuda_device_count`。
6. `collect_env.json.commands.nvidia_smi`。
7. `collect_env.json.env.LD_LIBRARY_PATH`。
8. `collect_env.json.env.CUDA_HOME`。
9. `collect_env.json.env.CUDA_VISIBLE_DEVICES` 或 shell 里的 CVD。

判断：

- `torch_importable=false`：先修 Python 包和解释器。
- `nvidia_smi` 有 GPU，但 `cuda_available=false`：重点查 PyTorch wheel、容器 GPU 可见性、driver/runtime 组合。
- `cuda_device_count` 小于预期：重点查 `CUDA_VISIBLE_DEVICES`、容器参数和调度系统资源分配。

## 4. NCCL 或 Gloo 后端异常

按顺序看：

1. `check_nccl.json.distributed_available`。
2. `check_nccl.json.nccl_available`。
3. `check_nccl.json.gloo_available`。
4. `torchrun_hello.log` 是否每个 rank 都有一行 JSON。
5. 每行 JSON 的 `rank`、`local_rank`、`world_size`、`backend`、`device`。

判断：

- `distributed_available=false`：当前 torch 不支持 distributed。
- `nccl_available=false`：GPU 多进程不能按 NCCL 通过，需要换 wheel 或环境。
- `torchrun_hello.log` 缺 rank：启动器或进程初始化失败。
- backend 走 Gloo：确认这是 CPU-only validation，还是 GPU 数不足导致 fallback。

## 5. rank/device 错位

按顺序看：

1. `CUDA_VISIBLE_DEVICES`。
2. `LOCAL_RANK`。
3. `WORLD_SIZE`。
4. `torchrun_hello.log` 中每个 rank 的 `device`。
5. patch 中 `parse_local_rank_mapping()` 的输出。

例子：

```text
CUDA_VISIBLE_DEVICES=2,3
LOCAL_RANK=1
visible_devices=[2, 3]
physical_device=3
process device=cuda:1
```

如果 `LOCAL_RANK` 超过 `visible_devices` 长度，应该先修 `--nproc_per_node` 或 CVD，而不是改训练模型。

## 6. 环境漂移

拿最近一次通过的 run 作为 baseline，对比：

- `command.sh`
- `config.resolved.yaml`
- `collect_env.json`
- `check_cuda.json`
- `check_nccl.json`
- `torchrun_hello.log`

优先看这些字段：

| 字段 | 漂移含义 |
|---|---|
| `python_executable` / `executable` | 进入了不同 Python |
| `PYTHONPATH` | import 搜索路径变化 |
| `CONDA_PREFIX` | Conda 环境变化 |
| `torch_version` | PyTorch wheel 变化 |
| `cuda_version` | wheel CUDA runtime 变化 |
| `CUDA_VISIBLE_DEVICES` | GPU 可见性变化 |
| `world_size` | 进程数变化 |
| `backend` | NCCL/Gloo 路径变化 |

## 7. 结束条件

排查结束时，报告中必须写清：

- 失败发生在哪一层：解释器、import、CUDA、distributed、torchrun、device mapping、artifact。
- 用哪个字段证明这个判断。
- 修改了哪个变量或配置。
- 修复后重新跑了哪个命令。
- 新旧 run artifact 的路径。
