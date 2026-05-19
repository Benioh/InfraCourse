# L00 Patch · 环境探针：把环境变成机器可读的证据

> 实现 3 个函数，让你能在 10 分钟内判断"这台机器能不能跑后续 lab"，并留下别人可以复现的环境证据。

## 你要改的文件

`labs/l01_env_conda_cuda/patch/starter/env_probe.py`

只改这一个文件。其它文件（reference/、tests/）不要动。

## 任务背景

接手一台陌生 4090/H200 时，最先暴雷的不是模型代码，而是环境本身：

- `torch.cuda.is_available() == False` —— driver 在、wheel 不对。
- `LOCAL_RANK=1` 但跑出来在 `cuda:0` —— `CUDA_VISIBLE_DEVICES` 把物理卡重映射了。
- `import xxx` 拿到的不是 pip 装的版本 —— `PYTHONPATH` 里有同名目录。

后面所有 lab（DDP、Megatron、vLLM、RL）都假设这三类问题已经解决。本关 patch 的目标就是把"我看了一眼觉得没问题"升级为"我有 JSON 证据可以复现"。

## 三个要实现的函数

### 1. `collect_python_env() -> dict`

返回当前 Python 进程的环境快照。**必须包含**这些 key：

| key | 类型 | 含义 |
|---|---|---|
| `python_version` | `str` | `sys.version.split()[0]`（如 `"3.11.9"`） |
| `python_executable` | `str` | `sys.executable` |
| `pythonpath` | `list[str]` | 把 `os.environ.get("PYTHONPATH", "")` 用 `os.pathsep` 切开；空字符串过滤掉 |
| `cuda_visible_devices` | `str \| None` | `os.environ.get("CUDA_VISIBLE_DEVICES")`；未设置返回 `None`（**不是空串**） |
| `local_rank` | `int \| None` | `int(os.environ["LOCAL_RANK"])` 如果存在；否则 `None` |
| `world_size` | `int \| None` | 同上但用 `WORLD_SIZE` |

要点：
- `pythonpath` 必须是 `list`，不是 `str`。空 PYTHONPATH 返回空 list `[]`。
- `cuda_visible_devices` 区分"未设置"和"设置成空串"——前者 `None`，后者 `""`。
- 这些字段全部从 `os.environ` 和 `sys` 读，**不要 import torch**（让函数在没装 torch 的纯环境里也能跑）。

### 2. `parse_local_rank_mapping(env: dict, local_rank: int) -> dict`

给定 `collect_python_env()` 的输出和一个 LOCAL_RANK，返回这个 rank 实际会落到哪张物理卡。

**输入**：
- `env`: 至少包含 key `cuda_visible_devices`。
- `local_rank`: 非负整数。

**输出 dict 的 key**：
- `local_rank`: 透传 `local_rank` 参数。
- `visible_devices`: `list[int]`。
  - `cuda_visible_devices` 是 `None` 或空串 → 返回 `[]`（约定：表示"看见所有卡"）。
  - 否则按逗号切，把每段 `int(x.strip())` 装成 list。
- `physical_device`: `int`。
  - `visible_devices` 为空 → `physical_device = local_rank`。
  - 否则 `physical_device = visible_devices[local_rank]`。

**异常**：
- 如果 `local_rank < 0` → `raise ValueError("local_rank must be >= 0, got {local_rank}")`。
- 如果 `visible_devices` 非空且 `local_rank >= len(visible_devices)` → `raise IndexError`，message **必须**包含三个字符串："local_rank=", "visible_devices=", "out of range"。这样工程师看 traceback 就能立刻定位 mismatch。
- `visible_devices` 里某段不能转 int（比如 `"abc"`）→ `raise ValueError`，message 必须包含 `"CUDA_VISIBLE_DEVICES"`。

### 3. `summarize_drift(baseline: dict, current: dict) -> list[str]`

对比两次 `collect_python_env()` 的结果，返回人类可读的漂移描述。

**约定**：
- 返回 `list[str]`，每条形如 `"<key>: <baseline> → <current>"`。
- 只比较两边都有的 key，且值不同的。**新增的 key 也算漂移**（描述为 `"<key>: <missing> → <value>"`）。
- 如果 baseline 和 current 完全相等 → 返回 `[]`。
- 输出按 key 字母序排列，方便 diff。
- 字段值是 list 时（比如 `pythonpath`），用 `repr` 打印整个 list；不要逐元素 diff。

## 测试覆盖

`tests/test_patch.py` 里 7 个测试：

| 类别 | 测试 | 通过条件 |
|---|---|---|
| 基础 | `test_collect_has_required_keys` | 6 个必填 key 都存在 |
| 基础 | `test_collect_pythonpath_is_list` | 即使 PYTHONPATH 未设置，返回 `[]` 而不是 `[""]` |
| 基础 | `test_collect_cvd_unset_vs_empty` | unset → None；空串 → `""` |
| 映射 | `test_parse_no_cvd_returns_local_rank` | CVD 未设置时 physical == local_rank |
| 映射 | `test_parse_remapped_cards` | CVD="2,3" + LOCAL_RANK=1 → physical=3 |
| 映射 | `test_parse_out_of_bounds_raises` | LOCAL_RANK=5 + CVD="0,1" → IndexError 含 "out of range" |
| 漂移 | `test_drift_pythonpath_change` | 改 PYTHONPATH 后产出明确的 diff 描述 |

跑测试：

```bash
make patch-test M=l01_env_conda_cuda
```

7 个全绿就过关。

## 卡住怎么办

1. 先看 `notebooks/n00_env_warmup.ipynb`（如果存在）把字段画一遍。
2. `make patch-hint M=l01_env_conda_cuda` 列出 TODO + 关键提示。
3. `make patch-show-solution M=l01_env_conda_cuda` 打开参考解。

## 配套源码研读（必看）

- `scripts/env/check_cuda.py` —— 真实项目里的 CUDA 探针写法。
- `labs/l01_env_conda_cuda/scripts/torchrun_hello.py` —— `LOCAL_RANK` 与 `CUDA_VISIBLE_DEVICES` 的实战互动。
- `mini_infra/observability/evidence.py` —— 把环境快照写成可审计 JSON 的最小骨架。

## 进入下一关的前置

`make patch-test M=l01_env_conda_cuda` 全绿后，再去看下一关 [L01 显存账本](../l02_pytorch_systems/README.md)。
