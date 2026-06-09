# 环境证据复盘模板

## Run 信息

- 日期：
- 机器 / 节点：
- GPU 型号和数量：
- 容器或 Conda 环境：
- git commit：
- 命令：
- 配置文件：
- run 目录：

## 运行模式

- [ ] CPU-only validation
- [ ] 本地单机 GPU smoke
- [ ] H200 单节点 smoke
- [ ] 集群启动器验证

本次模式能证明：

本次模式不能证明：

## 关键 artifact

| artifact | 路径 | 是否存在 | 备注 |
|---|---|---|---|
| `command.sh` |  |  |  |
| `config.resolved.yaml` |  |  |  |
| `artifacts/check_cuda.json` |  |  |  |
| `artifacts/check_nccl.json` |  |  |  |
| `artifacts/collect_env.json` |  |  |  |
| `artifacts/torchrun_hello.log` |  |  |  |
| `metrics.jsonl` |  |  |  |
| `report.md` |  |  |  |

## Python 和 import

| 字段 | 数值 | 判断 |
|---|---|---|
| `python_executable` / `executable` |  |  |
| `python_version` |  |  |
| `CONDA_PREFIX` |  |  |
| `PYTHONPATH` |  |  |
| `torch_importable` |  |  |
| `torch_version` |  |  |

## CUDA

| 字段 | 数值 | 判断 |
|---|---|---|
| `nvidia_smi` |  |  |
| `cuda_version` |  |  |
| `cuda_available` |  |  |
| `cuda_device_count` |  |  |
| `CUDA_VISIBLE_DEVICES` |  |  |
| `devices` |  |  |

## Distributed / torchrun

| 字段 | 数值 | 判断 |
|---|---|---|
| `distributed_available` |  |  |
| `nccl_available` |  |  |
| `gloo_available` |  |  |
| `WORLD_SIZE` |  |  |
| `torchrun_ok` |  |  |
| `backend` per rank |  |  |
| `device` per rank |  |  |

## rank/device 映射

记录 `CUDA_VISIBLE_DEVICES` 和每个 `LOCAL_RANK` 的映射：

| local rank | process device | physical device | 证据行 |
|---|---|---|---|
|  |  |  |  |

## Drift 对比

baseline run：

current run：

| 字段 | baseline | current | 判断 |
|---|---|---|---|
|  |  |  |  |

## 结论

- 本次能证明什么：
- 不能证明什么：
- 后续课程需要注意的环境边界：
- 下一步动作：
