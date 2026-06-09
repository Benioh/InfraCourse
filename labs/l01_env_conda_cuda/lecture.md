# L01 讲义：环境、CUDA 与证据链

这一讲讲环境层的工程证据。

一台机器能跑 `python`，不代表它能跑 PyTorch；`nvidia-smi` 能看到 GPU，也不代表当前 Python 进程能用 CUDA；`torchrun` 能拉起多个进程，也不代表每个 rank 都绑到了预期的 device。后面做 DDP、Megatron、vLLM、SGLang、verl 和 SLiME 时，很多故障会伪装成框架 bug，实际根因在解释器、wheel、动态库、容器设备可见性或启动参数。

L01 的目标是把这些基础事实变成可复查证据。学生要能看懂每个字段回答的问题，也要知道它不能证明什么。

## 1. 这节课学完要能回答什么

学完这一讲，你应该能自然回答下面这些问题。

1. `sys.executable`、`PYTHONPATH` 和 `CONDA_PREFIX` 分别能排查哪类 import 问题？
2. `nvidia-smi`、`torch.version.cuda`、`torch.cuda.is_available()` 和 `torch.cuda.device_count()` 分别证明哪一层？
3. `CUDA_VISIBLE_DEVICES=2,3` 时，进程内 `cuda:0` 和 `cuda:1` 对应哪些物理 GPU？
4. `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE` 在 `torchrun` 里分别表示什么？
5. `distributed_available`、`nccl_available` 和一次真实 `torchrun` 日志有什么区别？
6. patch 中的 `collect_python_env()` 为什么不能 import torch？
7. 为什么环境报告必须保存命令、配置、JSON、日志、metrics 和 report？
8. 当后续 lab 失败时，如何判断要先查环境层还是模型代码层？

本讲定位很明确：它属于 environment and evidence chain。我们关心的是环境事实、启动身份和证据落盘，不训练模型，也不评估吞吐。

## 2. 从一个常见故障讲起

假设你接手一台 H200 节点。`nvidia-smi` 能看到 8 张 GPU，但运行训练脚本时出现：

```text
torch.cuda.is_available() == False
```

另一个同学在同一台机器上能跑通。这个时候直接改训练 batch size、换模型配置或重装全部依赖都太早。你要先回答几个更小的问题：

- 当前脚本从哪个 Python 启动？
- 这个 Python import 到的 torch 是哪一个 wheel？
- wheel 绑定的 CUDA runtime 是什么版本？
- 容器或调度系统是否限制了可见 GPU？
- `LD_LIBRARY_PATH`、`CUDA_HOME`、`PYTHONPATH` 是否和预期一致？
- 多进程启动时，每个 rank 的 `LOCAL_RANK` 是否映射到正确 device？

L01 就是把这些问题变成字段和 artifact。字段足够细，排查时才能从“环境有问题”推进到“当前解释器 import 到了错误 torch”或“CVD 重映射导致 local rank 选错卡”。

## 3. 环境证据分成哪几层

环境检查不能只看一个命令。每个命令只回答一层问题。

| 层 | 代表证据 | 回答的问题 | 不能证明什么 |
|---|---|---|---|
| 解释器 | `sys.executable` | 当前脚本由哪个 Python 运行 | torch 是否可用 |
| import 路径 | `PYTHONPATH`、`CONDA_PREFIX` | 是否可能 import 到项目内同名目录或错误环境 | CUDA 是否可用 |
| PyTorch wheel | `torch.__version__`、`torch.version.cuda` | wheel 是否存在，绑定的 CUDA runtime 是什么 | driver 是否满足、GPU 是否可见 |
| CUDA 可见性 | `torch.cuda.is_available()`、`device_count` | 当前 Python 进程能否使用 CUDA | NCCL 通信是否成功 |
| 分布式后端 | `nccl_available`、`gloo_available` | PyTorch 是否编译或安装了后端 | 多进程是否真的跑通 |
| 启动身份 | `RANK`、`LOCAL_RANK`、`WORLD_SIZE` | 每个进程是谁、在本机排第几 | device 选择是否正确，仍要看日志 |
| 证据落盘 | `command.sh`、`config.resolved.yaml`、`artifacts/*.json` | 这次运行能否被别人复查 | 结论是否正确，仍要读字段 |

这张表是本讲的核心。后面所有环境排查都按这几层拆。

## 4. Python、Conda 和 PYTHONPATH

`collect_python_env()` 的第一类字段来自 `sys` 和 `os.environ`：

```python
{
    "python_version": sys.version.split()[0],
    "python_executable": sys.executable,
    "pythonpath": [...],
    "cuda_visible_devices": ...,
    "local_rank": ...,
    "world_size": ...,
}
```

**定义：**
Python executable 是实际运行脚本的解释器路径。Conda 环境是一组隔离包和部分动态库依赖。`PYTHONPATH` 是 Python 额外搜索模块的路径列表。

**直观理解：**
同一台机器上可能有系统 Python、Conda Python、容器内 Python 和项目虚拟环境。你在 shell 里看到的 `pip show torch`，不一定对应脚本真正使用的 `sys.executable`。

**输入：**
当前进程的 `sys.version`、`sys.executable` 和环境变量。

**中间状态：**
`PYTHONPATH` 要用 `os.pathsep` 拆开，并过滤空字符串。`CUDA_VISIBLE_DEVICES` 要保留 unset 和空串的区别。

**输出：**
一个可以 JSON 序列化的 dict。

**代价和边界：**
这个函数不能 import torch。它要在 torch 没装、torch import 崩溃或 wheel 错误时仍然能运行。它只能证明解释器和环境变量事实，不能证明 CUDA 或 NCCL 可用。

常见误解：

- 看到 Conda 环境名正确，就认为脚本一定用了正确 Python。实际要看 `sys.executable`。
- 把 `PYTHONPATH` 当成无害字段。它可能让 Python 优先 import 项目里的同名目录。
- 把空 `CUDA_VISIBLE_DEVICES` 和未设置混在一起。本讲要求字段能区分。

## 5. CUDA 可见性不是一个布尔值

CUDA 排查要分清四个证据：

| 证据 | 含义 |
|---|---|
| `nvidia-smi` | driver 层能看到硬件 |
| `torch.version.cuda` | PyTorch wheel 绑定的 CUDA runtime |
| `torch.cuda.is_available()` | 当前 Python 进程能通过 torch 使用 CUDA |
| `torch.cuda.device_count()` | 当前进程能看到几个逻辑 CUDA device |

`scripts/env/check_cuda.py` 把这些字段写成 `check_cuda.json`。它还记录设备 name、capability 和显存大小。这个脚本用于 smoke，不属于 patch 的纯 Python 函数。

一个典型判断：

```text
nvidia-smi 能看到 GPU
torch_importable = true
torch.version.cuda = null
torch.cuda.is_available() = false
```

这通常指向 CPU-only wheel 或 wheel 与 CUDA 运行环境不匹配。你还需要结合 `collect_env.json` 里的 `LD_LIBRARY_PATH`、`CUDA_HOME` 和容器设备可见性判断。

另一个典型判断：

```text
torch.version.cuda = 12.1
torch.cuda.is_available() = true
cuda_device_count = 2
CUDA_VISIBLE_DEVICES = 2,3
```

这说明当前进程只看到两张逻辑卡。进程内的 `cuda:0` 对应物理 GPU 2，`cuda:1` 对应物理 GPU 3。

## 6. LOCAL_RANK 和 CUDA_VISIBLE_DEVICES 的映射

`torchrun` 会给每个进程注入 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`。

| 字段 | 含义 |
|---|---|
| `RANK` | 全局进程编号，多节点时跨节点唯一 |
| `LOCAL_RANK` | 本机进程编号，通常用于选择本机 GPU |
| `WORLD_SIZE` | 总进程数 |

`CUDA_VISIBLE_DEVICES` 会改变进程内看到的 device 编号。设：

```bash
CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node 2 ...
```

两个进程看到的是：

| LOCAL_RANK | 进程内 device | 物理 GPU |
|---|---|---|
| 0 | `cuda:0` | 2 |
| 1 | `cuda:1` | 3 |

patch 中的 `parse_local_rank_mapping()` 就是把这张表写进代码。它的输入是 `collect_python_env()` 的输出和一个 `local_rank`，中间状态是解析后的 `visible_devices`，输出是 `physical_device`。

边界条件很重要：

- `local_rank < 0` 是调用错误，应该抛 `ValueError`。
- CVD 含非整数 token，例如 `0,abc`，应该抛 `ValueError`，错误信息要包含 `CUDA_VISIBLE_DEVICES`。
- CVD 非空且 `local_rank >= len(visible_devices)`，应该抛 `IndexError`，错误信息要包含 `local_rank=`、`visible_devices=` 和 `out of range`。

这些错误信息不是装饰。后续训练 hang 或多进程启动失败时，清楚的异常文本能直接指向启动参数问题。

## 7. NCCL、Gloo 和 torchrun smoke

`scripts/env/check_nccl.py` 检查三件事：

- `torch.distributed.is_available()`
- `torch.distributed.is_nccl_available()`
- `torch.distributed.is_gloo_available()`

这些字段说明当前 PyTorch 是否带有对应后端。它们不能替代一次真实多进程启动。

`labs/l01_env_conda_cuda/scripts/torchrun_hello.py` 才会在 `torchrun` 下运行。它做的事情很少：

1. 读取 `WORLD_SIZE`。
2. 判断 GPU 数是否足够覆盖 world size。
3. GPU 足够时用 `nccl`，否则用 `gloo`。
4. 读取 `LOCAL_RANK`。
5. GPU 模式下调用 `torch.cuda.set_device(local_rank)`。
6. 输出 rank、local_rank、world_size、backend 和 device 的 JSON 行。

这里要看清一个边界：CPU-only validation 可以验证脚本路径、环境变量和 Gloo 分支，但不能写成 NCCL/GPU 已通过。H200 smoke 要看每个 rank 的 device 和 backend 是否符合预期。

## 8. run_smoke.py 如何组织证据链

`run_smoke.py` 是本讲的系统闭环。它不承担 patch 测试职责；它把环境探针组织成一次可复查运行。

主路径是：

```text
read config
  -> prepare_run_dir
  -> write command.sh
  -> write prediction.yaml
  -> write config.resolved.yaml
  -> run check_cuda.py
  -> run check_torch.py
  -> run check_nccl.py
  -> run collect_env.py
  -> run torchrun_hello.py
  -> write train.log
  -> append metrics.jsonl
  -> write report.md
```

这些文件各自承担不同角色：

| 文件 | 用途 |
|---|---|
| `command.sh` | 复现这次运行的命令入口 |
| `config.resolved.yaml` | 固定本次 smoke 的配置事实 |
| `prediction.yaml` | 记录运行前对风险和瓶颈的判断 |
| `artifacts/check_cuda.json` | 记录 torch import、CUDA runtime、CUDA 可见性和设备属性 |
| `artifacts/check_nccl.json` | 记录 distributed、NCCL 和 Gloo 后端可用性 |
| `artifacts/collect_env.json` | 记录解释器、平台、环境变量和系统命令输出 |
| `artifacts/torchrun_hello.log` | 记录每个 rank 的身份、backend 和 device |
| `metrics.jsonl` | 按课程统一 schema 记录本次 smoke 结果 |
| `report.md` | 把证据转成人能读的结论 |

本讲的 metrics 中 `loss`、`tokens_per_sec`、`peak_memory_gb` 是占位字段，因为没有训练 workload。真正要看的字段是 `torchrun_ok`、`cuda_available` 和 artifact 内容。

## 9. Lab 只验收哪一个最小合同

patch 只实现三个函数：

```python
collect_python_env()
parse_local_rank_mapping(env, local_rank)
summarize_drift(baseline, current)
```

它不检查真实 CUDA，不启动 `torchrun`，也不证明 NCCL 性能。它验收的是字段语义：

- 环境快照字段是否完整。
- `PYTHONPATH` 是否返回 list。
- CVD unset 和空串是否区分。
- local rank 是否能映射到物理 device。
- 越界错误是否能定位问题。
- 两次环境快照是否能输出稳定 drift。

这些小函数是后续真实 smoke 的基础。patch 通过后还要跑 smoke，才能留下 `artifacts/` 证据。

## 10. 生产排查顺序

遇到环境相关问题时，按下面顺序排查。

### 10.1 import 失败

先看：

- `python_executable`
- `pythonpath`
- `CONDA_PREFIX`
- `collect_env.json` 里的 `cwd`
- 具体 import error

判断目标是：脚本是否进入预期 Python，是否被项目路径污染，是否缺依赖。

### 10.2 CUDA 不可用

先看：

- `torch_importable`
- `torch_version`
- `cuda_version`
- `cuda_available`
- `cuda_device_count`
- `nvidia_smi`
- `LD_LIBRARY_PATH`
- `CUDA_HOME`

判断目标是：问题在 wheel、driver、容器可见性、动态库路径，还是 GPU 被 CVD 屏蔽。

### 10.3 NCCL 或多进程异常

先看：

- `distributed_available`
- `nccl_available`
- `gloo_available`
- `torchrun_hello.log`
- `RANK`、`LOCAL_RANK`、`WORLD_SIZE`
- `CUDA_VISIBLE_DEVICES`

判断目标是：后端是否存在，进程是否拉起，rank 是否到齐，device 是否绑定正确。

### 10.4 环境漂移

先找一份最近通过的 run，把它的 `collect_env.json` 和当前 run 对比。`summarize_drift()` 给的是最小示例：字段并集、按 key 排序、值不同就输出。真实项目可以把这个思路扩展到更多字段。

## 11. 小结

L01 建立的是整门课的证据习惯。你不需要在这里记住所有 CUDA 版本组合，但要知道每个字段回答哪一层问题。后面任何训练或服务实验出错，都先确认本讲的基础事实：解释器、import path、torch wheel、CUDA 可见性、分布式后端、rank/device 映射和 artifact 是否完整。

通过这一讲后，学生应该能把“机器能不能跑”改写成一组可复查问题，并用 JSON、日志和报告支撑结论。
