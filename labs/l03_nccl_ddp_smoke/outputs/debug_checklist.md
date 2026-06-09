# L04 DDP Debug Checklist

这份清单用于排查 `ManualDDP` patch、2-rank smoke 和真实 DDP 入门问题。按顺序查，避免先改学习率或 batch size。

## 1. 先确认问题类型

| 现象 | 先查 |
|---|---|
| patch test 超时 | process group 初始化、collective 顺序、某个 rank 退出 |
| 梯度比 baseline 大 world size 倍 | 是否漏掉 `p.grad /= world_size` |
| 单 rank 测试失败 | no-op 分支是否正确 |
| frozen 参数报错 | 是否跳过 `requires_grad=False` |
| partial grad 报错 | 是否跳过 `p.grad is None` |
| smoke 生成 fallback | torchrun 是否完成、rank0 artifact 是否存在 |

## 2. Process Group 检查

- `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE` 是否在每个 worker 中设置。
- `WORLD_SIZE` 是否等于实际启动的进程数。
- 所有 rank 是否使用同一个 backend。
- `dist.init_process_group()` 是否只在需要分布式时调用。
- 结束时是否调用 `dist.destroy_process_group()`，避免污染后续测试。

对应源码：

- `labs/l03_nccl_ddp_smoke/patch/tests/conftest.py` L33-L53
- `labs/l03_nccl_ddp_smoke/scripts/ddp_hello.py` L31-L48

## 3. Gradient Sync 检查

- `ManualDDP.__init__` 是否保留原始 `model`，没有复制参数。
- `self.world_size` 是否来自 `dist.get_world_size(process_group)`。
- `world_size == 1` 或 distributed 未初始化时是否直接返回。
- 遍历参数时是否先跳过 frozen 参数。
- 是否跳过 `p.grad is None`。
- 是否使用 `dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.process_group)`。
- 是否在 all-reduce 后除以 `self.world_size`。

对应源码：

- `labs/l03_nccl_ddp_smoke/patch/reference/manual_ddp.py` L11-L24
- `labs/l03_nccl_ddp_smoke/patch/tests/worker_cases.py` L48-L81
- `labs/l03_nccl_ddp_smoke/patch/tests/worker_cases.py` L100-L131

## 4. Collective 顺序检查

- 所有 rank 是否遍历同一组需要通信的参数。
- 是否有某个 rank 对参数 A 通信，另一个 rank 跳过 A 后对参数 B 通信。
- 模型分支是否导致某些 rank 有 grad，另一些 rank 为 `None`。
- 如果加入了自定义跳过逻辑，是否保证所有 rank 的决策一致。

快速判断：

```text
每个 rank 的 all-reduce 次数必须一致。
每一次 all-reduce 的 tensor 顺序必须一致。
每一次 all-reduce 的 tensor shape/dtype 必须一致。
```

## 5. Smoke Artifact 检查

查看最近 run 的：

- `artifacts/ddp_hello.json`
- `metrics.jsonl`
- `train.log`
- `report.md`

关键字段：

| 字段 | 期望 |
|---|---|
| `world_size` | smoke 默认应为 2 |
| `all_reduce_sum` | 2-rank 时应为 3 |
| `barrier_ok` | 应为 true |
| `fallback_used` | false 才能作为真实 process group 证据 |

如果 `fallback_used=True`：

- 不要写成 NCCL 或真实 DDP 通过。
- 先看 `train.log` 和 `report.md` 的 fallback 原因。
- 检查 `torch.distributed.run` 是否可用。
- 检查 `artifacts/ddp_hello.json` 是否由 rank0 写出。

## 6. 最小复现命令

```bash
make patch-test M=l03_nccl_ddp_smoke
python labs/l03_nccl_ddp_smoke/scripts/run_smoke.py --mode smoke
python -m labs.l03_nccl_ddp_smoke.scripts.ddp_hello --output /tmp/ddp_hello.json
python mini_infra/distributed/collectives.py --json
```

解释命令输出时要分清：

- patch test：验证 `ManualDDP` 梯度同步语义。
- smoke：验证最小 process group、all-reduce 和 barrier。
- MiniInfra snapshot：验证环境变量和期望 all-reduce sum 的记录逻辑。
