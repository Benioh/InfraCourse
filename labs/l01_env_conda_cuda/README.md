# L01 · 环境探针：把环境变成可复查证据

这一讲解决开工前的第一个工程问题：拿到一台陌生 4090、H200 或 CPU 开发机时，怎样证明后面的 PyTorch、DDP、Megatron、Serving 和 RL 实验具备基本运行条件。

这里的“证明”要落成机器可读证据：Python 从哪里启动，Conda 环境和 `PYTHONPATH` 是否干净，PyTorch wheel 是否能 import，当前进程能看到几张 CUDA 设备，NCCL/Gloo 是否可用，`torchrun` 的每个 rank 绑定到了哪个 device。后面所有课程都会复用这套 command/config/metrics/report/artifacts 证据链。

## 学习路线

建议按下面顺序走，先把环境层问题讲清，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L01 在全课程证据链里的位置。
2. 读 [lecture.md](lecture.md)：理解 Python、Conda、PyTorch wheel、CUDA 可见性、NCCL/Gloo 和 `torchrun` 的边界。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按顺序阅读环境探针、smoke runner 和 artifact 汇总代码。
4. 跑 notebook：[n00_env_warmup.ipynb](../../notebooks/n00_env_warmup.ipynb)。
5. 做 quiz：确认能区分 import、CUDA、NCCL、rank/device 映射和环境漂移。
6. 做 patch：实现 3 个环境探针函数。
7. 跑 smoke：生成一次完整环境证据。
8. 填写 [outputs/env_report_template.md](outputs/env_report_template.md)，沉淀本机环境结论。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Environment and evidence chain |
| 它解决什么问题 | 把“这台机器能不能跑后续 lab”拆成可复查字段和 artifact |
| 它连接哪些证据 | `command.sh`、`config.resolved.yaml`、`check_cuda.json`、`check_nccl.json`、`collect_env.json`、`torchrun_hello.log`、`metrics.jsonl`、`report.md` |
| 它连接哪些源码 | `patch/reference/env_probe.py`、`scripts/env/check_cuda.py`、`scripts/env/check_nccl.py`、`scripts/collect_env.py`、`scripts/run_smoke.py`、`mini_infra/observability/evidence.py` |
| lab 检验什么 | 环境快照、LOCAL_RANK 到物理 GPU 的映射、两次环境快照的 drift 摘要 |

## 你会学到什么

- 用 `sys.executable`、`PYTHONPATH`、`CONDA_PREFIX` 判断脚本是否进入预期环境。
- 区分 `nvidia-smi`、`torch.version.cuda`、`torch.cuda.is_available()` 和 `torch.cuda.device_count()` 各自证明的层面。
- 解释 `CUDA_VISIBLE_DEVICES=2,3` 如何把物理 GPU 重映射成进程内的 `cuda:0/1`。
- 读懂 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 和 `torchrun` 多进程身份。
- 把环境检查落成 JSON、日志、metrics 和报告，而不是只保存终端截图。
- 面对 import 失败、CUDA 不可用、NCCL 不可用、rank/device 错位时，给出有顺序的排查路径。

## Patch 闭环

```bash
cat labs/l01_env_conda_cuda/patch/task.md
$EDITOR labs/l01_env_conda_cuda/patch/starter/env_probe.py
make patch-test M=l01_env_conda_cuda
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_collect_has_required_keys` | 环境快照包含 6 个必填字段 |
| `test_collect_pythonpath_is_list` | `PYTHONPATH` 被拆成 list，空值过滤 |
| `test_collect_cvd_unset_vs_empty` | `CUDA_VISIBLE_DEVICES` 的 unset 和空串语义不同 |
| `test_parse_no_cvd_returns_local_rank` | 没有 CVD 重映射时 physical device 等于 local rank |
| `test_parse_remapped_cards` | CVD 重映射后 local rank 能映射到物理 GPU |
| `test_parse_out_of_bounds_raises` | rank 越界时错误信息可定位 |
| `test_drift_pythonpath_change` | 环境漂移按字段输出可读 diff |

## Smoke 闭环

本地 4090 或 CPU smoke：

```bash
python labs/l01_env_conda_cuda/scripts/run_smoke.py \
  --config configs/4090_debug.yaml \
  --mode smoke
```

H200 单节点模板：

```bash
python labs/l01_env_conda_cuda/scripts/run_smoke.py \
  --config configs/h200_node.yaml \
  --mode cluster-smoke
```

smoke 会写出：

```text
runs/mini_infra/l01_env_conda_cuda/<run-id>/
├── command.sh
├── config.resolved.yaml
├── prediction.yaml
├── metrics.jsonl
├── report.md
└── artifacts/
    ├── check_cuda.json
    ├── check_torch.json
    ├── check_nccl.json
    ├── collect_env.json
    └── torchrun_hello.log
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | import、CUDA、NCCL、torchrun 和环境漂移的排查顺序 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速复习本讲源码主路径 |
| [outputs/env_report_template.md](outputs/env_report_template.md) | 记录一次真实环境检查的证据和结论 |

## 进入下一讲

`make patch-test M=l01_env_conda_cuda` 通过，并完成一次 smoke 复盘后，进入 [L02 PyTorch 系统课](../l02_pytorch_systems/README.md)。下一讲会开始看真实训练循环的显存和 profiler 证据。
