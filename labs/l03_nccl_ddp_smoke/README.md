# L04 · ManualDDP：把梯度同步写清楚

这一讲解决分布式训练的第一个核心问题：多个 rank 各自算出本地梯度后，怎样把梯度变成所有 rank 一致的平均值。PyTorch DDP 会自动做这件事，但本讲先手写一个最小 `ManualDDP`，让学生看清 `all_reduce(SUM)`、除以 `world_size`、跳过 `grad=None` 和 `requires_grad=False` 的参数这些基本合同。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L04 在训练系统从单进程到多进程的过渡位置。
2. 读 [lecture.md](lecture.md)：理解 rank、world size、process group、all-reduce、barrier 和 DDP 梯度平均。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch reference、worker tests、smoke 和 MiniInfra collective 路径读源码。
4. 跑 notebook：[n03_ddp_collectives.ipynb](../../notebooks/n03_ddp_collectives.ipynb)。
5. 做 quiz：确认 collective 顺序、平均梯度和 fallback 边界。
6. 做 patch：实现 `ManualDDP.synchronize_grads()`。
7. 跑 smoke：生成 2-rank process group 的 `ddp_hello.json` 和 metrics。
8. 填写 [outputs/distributed_debug_template.md](outputs/distributed_debug_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / distributed basics |
| 它解决什么问题 | backward 后把多个 rank 的本地 grad 同步成平均 grad |
| 它连接哪些指标 | world_size、backend、all_reduce_sum、barrier_ok、fallback_used、grad 是否与 PyTorch DDP 对齐 |
| 它连接哪些源码 | `patch/reference/manual_ddp.py`、`patch/tests/worker_cases.py`、`scripts/ddp_hello.py`、`scripts/run_smoke.py`、`mini_infra/distributed/collectives.py` |
| lab 检验什么 | forward 不通信；backward 后 all-reduce grad；求和后除以 world size；跳过冻结参数和 `grad=None` |

## 你会学到什么

- `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 和 process group 的关系。
- all-reduce sum 和 averaged gradient 的区别。
- DDP 为什么要让每个 rank 的 gradient 一致。
- `requires_grad=False` 和 `grad=None` 参数为什么不能参与通信。
- world size 为 1 或 distributed 未初始化时，`synchronize_grads()` 为什么应是 no-op。
- smoke 的 `fallback_used=True` 为什么只能算 validation-only，不能写成真实 NCCL/DDP 通过。

## Patch 闭环

```bash
cat labs/l03_nccl_ddp_smoke/patch/task.md
$EDITOR labs/l03_nccl_ddp_smoke/patch/starter/manual_ddp.py
make patch-test M=l03_nccl_ddp_smoke
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_grads_match_pytorch_ddp` | 与 PyTorch DDP 在相同输入下 grad 对齐 |
| `test_grads_are_averaged_not_summed` | grad 是平均值，不能差一个 world size 因子 |
| `test_world_size_1_is_noop` | 单 rank 时同步不改 grad |
| `test_skips_no_grad_params` | 冻结参数不通信 |
| `test_handles_partial_grads` | 部分参数 `grad=None` 时不报错 |

## Smoke 闭环

```bash
python labs/l03_nccl_ddp_smoke/scripts/run_smoke.py --mode smoke
```

smoke 会尝试用 `torch.distributed.run` 拉起 2 个进程，运行 `scripts/ddp_hello.py`，并写出：

- `artifacts/ddp_hello.json`
- `metrics.jsonl`
- `train.log`
- `report.md`

关键字段：

| 字段 | 含义 |
|---|---|
| `backend` | 本次使用 `nccl`、`gloo` 或 fallback |
| `world_size` | 参与 process group 的进程数 |
| `all_reduce_sum` | rank+1 张量经过 all-reduce 后的和，2-rank 时应为 3 |
| `barrier_ok` | 是否到达 barrier |
| `fallback_used` | 是否走模拟结果 |

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | DDP hang、梯度大 world size 倍、backend/fallback 的排查顺序 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 ManualDDP 和 process group 源码路径 |
| [outputs/distributed_debug_template.md](outputs/distributed_debug_template.md) | 记录一次分布式 smoke 或 grad sync 复盘 |

## 进入下一讲

`make patch-test M=l03_nccl_ddp_smoke` 通过，并完成一次 smoke 复盘后，进入 [L05 GPU kernel](../l04_gpu_kernel/README.md)。下一讲会把单机训练证据推进到 kernel 级性能。
