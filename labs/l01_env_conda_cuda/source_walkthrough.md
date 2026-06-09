# 源码带读：L01 环境探针

这份带读按“纯 Python 探针 -> CUDA/NCCL 探针 -> torchrun 身份 -> smoke 证据落盘 -> MiniInfra 汇总”的顺序走。读源码时只抓主路径，先不要追 argparse、日志格式和测试夹具细节。

## 0. 源码地图

```text
labs/l01_env_conda_cuda/patch/reference/env_probe.py
  -> collect_python_env
  -> parse_local_rank_mapping
  -> summarize_drift

scripts/env/check_cuda.py
scripts/env/check_nccl.py
scripts/env/collect_env.py

labs/l01_env_conda_cuda/scripts/torchrun_hello.py
labs/l01_env_conda_cuda/scripts/run_smoke.py

mini_infra/observability/evidence.py
```

MiniInfra 和 lab 脚本共同保留一个不变量：每次运行都要能回到命令、配置、指标、报告和 artifact。

## 1. Patch reference：纯 Python 环境快照

文件：`labs/l01_env_conda_cuda/patch/reference/env_probe.py`

先看第 9-26 行：

```python
def collect_python_env() -> dict[str, Any]:
    raw_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = [seg for seg in raw_pythonpath.split(os.pathsep) if seg]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
```

读完这里要得到三个结论。

第一，`collect_python_env()` 只读 `sys` 和 `os.environ`，它不 import torch。这样 torch 没装或 import 失败时，环境快照仍能生成。

第二，`PYTHONPATH` 被拆成 list，并过滤空字符串。这让后续 drift 比较能按字段比较，而不是比较一整段路径字符串。

第三，`CUDA_VISIBLE_DEVICES` 用 `os.environ.get` 原样保留。未设置是 `None`，设置为空串是 `""`，两者语义不同。

再看第 15-25 行：

```python
def _maybe_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw != "" else None
```

这里把 `LOCAL_RANK` 和 `WORLD_SIZE` 转成整数。空串不转，返回 `None`。读完后要能解释：这个函数只记录启动器注入的身份，不创建进程，也不判断身份是否正确。

可以先跳过：文件顶部 import 和 `_MISSING` 常量，等读 drift 时再回来看。

## 2. Patch reference：rank 到物理 GPU 的映射

仍然在 `env_probe.py`。

重点看第 29-57 行：

```python
def parse_local_rank_mapping(env: dict[str, Any], local_rank: int) -> dict[str, Any]:
    if local_rank < 0:
        raise ValueError(...)

    cvd = env.get("cuda_visible_devices")
    if cvd is None or cvd == "":
        visible: list[int] = []
    else:
        visible = [int(seg.strip()) for seg in cvd.split(",") if seg.strip() != ""]
```

这里的输入是环境快照和本地 rank。中间状态是 `visible`。

继续看第 44-56 行：

```python
if visible:
    if local_rank >= len(visible):
        raise IndexError(...)
    physical = visible[local_rank]
else:
    physical = local_rank
```

读完要得到两个结论。

第一，CVD 为空时，教学版约定“没有显式重映射”，所以物理 device 等于 local rank。真实机器上还要结合可见 GPU 数和调度系统限制判断。

第二，CVD 非空时，`LOCAL_RANK` 是 `visible_devices` 的索引，不是物理 GPU 编号。

可以先跳过：异常包装的细节。读完主路径后再回来看错误信息必须包含哪些字段。

## 3. Patch reference：环境漂移摘要

重点看第 63-82 行：

```python
keys = sorted(set(baseline) | set(current))
for key in keys:
    a = baseline.get(key, _MISSING)
    b = current.get(key, _MISSING)
```

这段代码的输入是两份快照。中间状态是两边 key 的并集。输出是按 key 排序的字符串列表。

读完要形成一个判断：环境 drift 的第一版不需要复杂算法。先保证字段稳定、顺序稳定、缺失值可见。后续项目可以把这个模式扩展到 CUDA、NCCL、package version 和 config。

## 4. CUDA 探针：当前 Python 进程能否用 CUDA

文件：`scripts/env/check_cuda.py`

重点看第 16-27 行：

```python
payload = {
    "torch_importable": torch is not None,
    "torch_version": getattr(torch, "__version__", None) if torch else None,
    "cuda_available": bool(torch and torch.cuda.is_available()),
    "cuda_device_count": (
        torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
    ),
    "cuda_version": getattr(torch.version, "cuda", None) if torch else None,
    "devices": [],
    "import_error": IMPORT_ERROR,
}
```

每个字段回答的问题不同：

- `torch_importable`：当前 Python 能否 import torch。
- `torch_version`：import 到的 torch 版本。
- `cuda_version`：wheel 绑定的 CUDA runtime。
- `cuda_available`：当前进程能否通过 torch 使用 CUDA。
- `cuda_device_count`：当前进程能看到几个逻辑 CUDA device。
- `import_error`：import torch 失败时保留错误文本。

再看第 28-42 行。只有 CUDA 可用时才枚举 device name、capability 和 total memory。CPU-only 机器上 `devices` 为空不是脚本错误。

可以先跳过：`main()` 的 `--pretty` 和 `--output` 分支，只要知道它会把 payload 写成 JSON。

## 5. NCCL 探针：后端存在不等于真实通信通过

文件：`scripts/env/check_nccl.py`

重点看第 16-34 行：

```python
backend = getattr(torch.distributed, "is_available", lambda: False)()
nccl_available = getattr(torch.distributed, "is_nccl_available", lambda: False)()
gloo_available = getattr(torch.distributed, "is_gloo_available", lambda: False)()
```

这里要得到三个结论。

第一，`distributed_available` 表示 PyTorch distributed 模块可用。

第二，`nccl_available` 和 `gloo_available` 表示对应后端存在。

第三，这个脚本不启动多个进程，也不执行 all-reduce。真实通信还要看 `torchrun_hello.log` 或后续 DDP smoke。

可以先跳过：argparse 和输出写文件逻辑。

## 6. collect_env：把系统上下文放进 artifact

文件：`scripts/env/collect_env.py`

重点看第 29-54 行：

```python
return {
    "python": sys.version,
    "platform": platform.platform(),
    "executable": sys.executable,
    "cwd": os.getcwd(),
    "env": {
        "PATH": os.environ.get("PATH"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
    },
    ...
}
```

这份 artifact 比 patch 的环境快照更宽。它记录平台、工作目录、关键环境变量、torch 版本和系统命令输出。排查真实问题时，`collect_env.json` 常用来解释为什么同一份代码在两台机器上行为不同。

可以先跳过：`maybe_run()` 的异常处理。先知道它会尝试运行 `nvidia-smi`、`nvcc` 和 `conda`。

## 7. torchrun_hello：验证多进程身份和 device 绑定

文件：`labs/l01_env_conda_cuda/scripts/torchrun_hello.py`

重点看第 17-29 行：

```python
world_size = int(os.environ.get("WORLD_SIZE", "1"))
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
can_use_cuda = gpu_count >= world_size and gpu_count > 0
backend = "nccl" if can_use_cuda else "gloo"
...
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if can_use_cuda:
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
else:
    device = "cpu"
```

这段代码把环境变量和 PyTorch 运行时连起来。

读完要得到三个结论：

1. GPU 数量覆盖 world size 时才走 NCCL 分支。
2. GPU 分支用 `LOCAL_RANK` 选择进程内逻辑 device。
3. CPU 或 GPU 不足时走 Gloo，用于 validation-only smoke。

再看第 30-38 行。payload 输出 `rank`、`local_rank`、`world_size`、`backend` 和 `device`。真实排查时要逐行检查每个 rank 是否符合预期。

可以先跳过：`dist.barrier()` 和 `destroy_process_group()` 的细节。后续 DDP 课程会展开 collective。

## 8. run_smoke：把探针组织成一次可复查运行

文件：`labs/l01_env_conda_cuda/scripts/run_smoke.py`

先看第 60-82 行：

```python
config = yaml.safe_load(...)
run_dir = prepare_run_dir(...)
write_command_snapshot(run_dir)
ensure_prediction(...)
write_yaml(run_dir / "config.resolved.yaml", ...)
run_python("check_cuda.py", ...)
run_python("check_torch.py", ...)
run_python("check_nccl.py", ...)
run_python("collect_env.py", ...)
torchrun_ok = run_torchrun(...)
```

这是 smoke 主路径。它先固定命令和配置，再运行各类探针，最后跑 `torchrun_hello.py`。

再看第 84-109 行：

```python
check_cuda = json.loads(...)
write_text(run_dir / "train.log", ...)
append_jsonl(run_dir / "metrics.jsonl", ...)
```

这里把检查结果转成课程统一的日志和 metrics。`loss`、`tokens_per_sec`、`peak_memory_gb` 为占位字段，环境课真正关注 `torchrun_ok` 和 `cuda_available`。

最后看第 110-148 行。报告里写目标、环境与配置、预测、结果、诊断和 debug ticket。读完要能说清：报告结论必须能回到 artifact 字段。

可以先跳过：`run_torchrun()` 对 `torchrun` 命令存在性的兼容分支。主结论是日志必须落到 `artifacts/torchrun_hello.log`。

## 9. MiniInfra evidence：检查 run 目录是否可审计

文件：`mini_infra/observability/evidence.py`

重点看第 11-29 行：

```python
def summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    return {
        "run_dir": ...,
        "has_command": (run_dir / "command.sh").exists(),
        "has_config": (run_dir / "config.resolved.yaml").exists(),
        "has_report": (run_dir / "report.md").exists(),
        "metric_count": len(metrics),
        "latest_metric": metrics[-1] if metrics else None,
        "artifact_count": ...,
    }
```

这个文件不判断 CUDA 对不对。它判断一次 run 是否有基本审计材料。后续 capstone 会把多次 run 的证据聚合成交付包，L01 的证据规范会一直被复用。

## 10. 读完后的自检问题

1. 为什么 `collect_python_env()` 不 import torch，而 `check_cuda.py` 可以 import torch？
2. `CUDA_VISIBLE_DEVICES=2,3` 且 `LOCAL_RANK=1` 时，进程内 device 和物理 GPU 分别是什么？
3. `check_nccl.py` 里的 `nccl_available=true` 还缺哪类证据才能说明多进程 smoke 通过？
4. `run_smoke.py` 生成的哪些文件能证明命令和配置可复现？
5. `metrics.jsonl` 里的哪些字段对 L01 有解释价值，哪些只是统一 schema 的占位字段？
6. 如果后续 L03 DDP hang，你会回到 L01 的哪些 artifact 先看？
