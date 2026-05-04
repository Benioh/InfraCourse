# L00 · 环境、Conda、CUDA 与 Git 生存课

> 先把环境变成可复现证据链，再谈训练、推理和 RL。

## 课程定位

- 角色：Infra 新兵
- 阶段：第 0 章：入门与环境
- 优先级：核心
- 框架：PyTorch, CUDA/NCCL
- 本地模式：1×RTX 4090 调试
- 集群模式：8×H200 单节点验证

## 时间与依赖

- 预计耗时：源码 30min / smoke 20min / 对照实验 45min / 报告 45min
- 前置依赖：`docs/前置体检清单.md`；Conda/Git/YAML/Make 最小能力
- 最小硬件：0 GPU 可完成 80%；1×4090 可完整验证；8×H200 可补集群 NCCL 映射
- No-GPU 可完成度：80%（环境、Python、Conda、Git、报告；CUDA/NCCL 标注缺失）
- 术语预习：`docs/术语表.md#rank--local-rank--world-size`、`docs/术语表.md#validation-only`

## MiniInfra 主线增量

- 主线定位：本关先在 MiniInfra 真实项目最小同构骨架上完成增量，再回到 notebook、真实源码和 lab smoke 验证。
- 主线模块：`mini_infra/observability/evidence.py`
- 本关增量：建立 command/config/metrics/report 的证据链规范，让后续每个真实同构栈 run 都可审计、可复现。
- 主线命令：`make mini-infra M=l01_env_conda_cuda RUN_ID=<run_id>`
- Notebook 联动：`notebooks/n00_env_warmup.ipynb`
- 真实源码对照：`scripts/env/collect_env.py`, `scripts/env/check_cuda.py`, `scripts/env/create_env.sh`
- 辅助 toy / validation-only：环境脚本和 torchrun hello 是证据采集器；它们服务于 MiniInfra evidence 规范，不是单独主线。
- 报告要求：必须同时写清 MiniInfra 同构文件、对应真实源码、lab artifact/metrics 三者如何互相验证；不能只写 toy demo 结论。

## 本关总问题

同一段 PyTorch 代码为什么会在 import、CUDA、NCCL、rank/device 映射四个层面分别失败？

## 学习闭环

本关不再是“运行一条命令看结果”，而是按高级 infra 工程师的工作方式推进：

1. **读源码**：先读本关 MiniInfra 主线模块，再从真实框架或课程脚本入口画出调用链，标出不变量和失败边界。
2. **建直觉**：运行 notebook，把公式、显存、调度或 RL 概念变成可解释预测。
3. **做项目**：围绕真实场景产出可复现 artifacts，而不是一次性 demo。
4. **做对照**：每次只改一个核心变量，用 metrics/logs/trace 支撑结论。
5. **做排障**：至少选择一个 debug ticket，按最小复现路径定位。
6. **写报告**：报告必须回答“我读懂了哪段源码、验证了哪个假设、产物在哪里”。

## 源码研读路径

### 1. Conda 环境创建入口

- 源码路径：`scripts/env/create_env.sh`
- 读代码重点：梳理 env 名称、yaml 文件、conda/mamba fallback、失败退出路径。
- 必答问题：
  - ENV 参数如何映射到 envs/*.yaml？
  - 脚本如何避免静默创建错误环境？
  - 如果 learner 在已有环境中运行，应该记录哪些漂移？

### 2. CUDA 与 PyTorch wheel 体检

- 源码路径：`scripts/env/check_cuda.py`
- 读代码重点：区分 driver、runtime、device_count、torch wheel CUDA ABI。
- 必答问题：
  - torch.cuda.is_available 为 false 时有哪些可定位字段？
  - 为什么 driver 正常不代表 PyTorch wheel 可用？
  - 哪些字段必须写入 artifacts 才能复现？

### 3. NCCL 与 torchrun rank 语义

- 源码路径：`labs/l01_env_conda_cuda/scripts/torchrun_hello.py`
- 读代码重点：读取 RANK、LOCAL_RANK、WORLD_SIZE，并验证每个进程绑定的 device。
- 必答问题：
  - LOCAL_RANK 与 CUDA_VISIBLE_DEVICES 的关系是什么？
  - 单机多进程和多机多进程的最小差异是什么？
  - 哪一类 rank/device mismatch 不能靠重启解决？

## Notebook 桥接

- `notebooks/n00_env_warmup.ipynb`
  - 运行前：先预测 `LOCAL_RANK` 与 `CUDA_VISIBLE_DEVICES` 的映射。
  - 运行后：读取 `collect_env.json`，确认哪些字段能进入环境证据链。
  - 无 GPU：可完整运行，用 sample payload 理解字段结构。

## 真实项目：Environment Flight Recorder：一键环境审计器

**项目场景**：你接手一台陌生 4090/H200 机器，必须在 10 分钟内判断它能否安全运行后续训练/推理/RL lab，并留下别人可复现的环境证据。

### 里程碑

- 生成 driver / CUDA runtime / PyTorch wheel / NCCL / torchrun 五段式报告。
- 构造一个故意错误的 PYTHONPATH 或 CUDA_VISIBLE_DEVICES 场景，并用最小证据定位。
- 把 4090 本地模式与 H200 单节点模式的差异写入 config.resolved.yaml。
- 产出一份给后来者运行的 env_report.md 和 command.sh。

### 必交付物

- artifacts/check_cuda.json、check_torch.json、check_nccl.json、collect_env.json。
- artifacts/torchrun_hello.log，含每个 rank 的 hostname、rank、local_rank、device。
- report.md 中必须包含一次环境漂移诊断。

### 进阶挑战

- 把 collect_env 的 JSON 转成 Markdown 表格。
- 为 ticket/env_rank_device_mismatch_003 写一个最小复现命令。

## 实验矩阵

| 控制变量 | 观察指标/现象 | 工程目的 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | device_count、LOCAL_RANK 到 device 的映射 | 学习 rank/device mismatch 的最小排查路径。 |
| `PYTHONPATH` | import 路径、包版本、脚本实际加载文件 | 避免同名源码 checkout 与 pip 包互相污染。 |
| `torchrun nproc_per_node` | 每个 rank 的日志完整性 | 为 L02 DDP hang 建立前置直觉。 |

## 推荐命令

先查看任务简报：

```bash
make mission M=l01_env_conda_cuda
```

### 环境准备

```bash
make env ENV=base
```

### 本地 smoke

```bash
make smoke M=l01_env_conda_cuda
```

### 4090 运行

```bash
make run-4090 M=l01_env_conda_cuda
```

### H200 运行

```bash
make run-h200 M=l01_env_conda_cuda
```

### 评分

```bash
make grade M=l01_env_conda_cuda
```

## 数据与输入

- 无外部数据集。

## 必看指标

- `cuda_available`
- `cuda_device_count`
- `torchrun_ok`
- `nccl_available`
- `collect_env_complete`

## Debug Tickets

- `env_import_error_001`
- `env_cuda_unavailable_002`
- `env_rank_device_mismatch_003`

## 成功标准

- 能画出至少一条从入口命令到核心指标或 artifact 写入的源码调用链。
- 能解释本关关键不变量：每次 run 必须写 command.sh、config.resolved.yaml 和 artifacts/collect_env.json。; CUDA 可用、NCCL 可用、torchrun 可用是三个不同结论，不能互相替代。
- 能完成真实项目的必交付物，并在 `runs/l01_env_conda_cuda/<run_id>/` 留下命令、配置、日志、指标和报告。
- 能说明本地 4090、H200 集群与 validation-only 三种边界，不把验证当成真实训练/服务/RL 成功。

## 交付物

- `prediction.yaml`：实验前预测，必须包含源码热点和预期瓶颈。
- `command.sh` / `config.resolved.yaml`：可复现命令与解析后的配置。
- `metrics.jsonl`：机器可读指标，至少覆盖本 README 的必看指标。
- `artifacts/`：环境、数据、trace、checkpoint、benchmark 或验证产物。
- `report.md`：中文实验报告，包含源码研读、实验矩阵、debug ticket 与迁移建议。
- `grade.json`：autograder 输出，只作为完成度检查，不替代工程判断。

## 常见误区

- 只跑命令不读源码，不知道指标从哪里来。
- 同时改多个配置，导致无法解释性能/显存变化。
- 把 validation-only 当作真实集群运行结果。
- 报告只贴日志，不说明源码不变量、失败边界和下一步验证。

## Stuck Checklist

- `torch.cuda.is_available=false` → 先记录 driver/runtime/wheel，再确认 PyTorch CUDA wheel 是否匹配。
- `ModuleNotFoundError` → 检查 `which python`、`PYTHONPATH` 和本地同名目录。
- `LOCAL_RANK` 与 device 不一致 → 检查 `CUDA_VISIBLE_DEVICES` 顺序和 torchrun `nproc_per_node`。
---
← 上一关 [课程首页](../../README.md) · 你会建立环境证据链
→ 下一关 [L01 PyTorch Systems](../l02_pytorch_systems/README.md) · 你将学习单卡训练与 profiler
