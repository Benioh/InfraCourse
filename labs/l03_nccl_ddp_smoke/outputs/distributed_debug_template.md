# L04 Distributed Debug Template

## 1. Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- git commit：
- Python / PyTorch：
- backend：
- world size：
- 是否 fallback：

## 2. 本次要验证的问题

| 项 | 内容 |
|---|---|
| 目标 |  |
| 比较对象 |  |
| 成功标准 |  |
| 已知边界 |  |

示例：

```text
目标：验证 ManualDDP 的 grad 与 PyTorch DDP 对齐。
比较对象：patch tests 中的 PyTorch DDP reference。
成功标准：5 个 patch tests 全部通过。
已知边界：CPU/gloo 测试不代表 NCCL 性能。
```

## 3. 证据路径

| 证据 | 路径或命令 | 观察 |
|---|---|---|
| patch tests | `make patch-test M=l03_nccl_ddp_smoke` |  |
| smoke artifact | `artifacts/ddp_hello.json` |  |
| metrics | `metrics.jsonl` |  |
| log | `train.log` |  |
| report | `report.md` |  |

## 4. 指标记录

| 字段 | 值 | 判断 |
|---|---|---|
| `world_size` |  | 是否等于预期进程数 |
| `backend` |  | gloo / nccl / fallback |
| `all_reduce_sum` |  | 2-rank 时应为 3 |
| `barrier_ok` |  | 是否为 true |
| `fallback_used` |  | true 时不能写成真实通信通过 |
| patch test count |  | 是否全部通过 |

## 5. 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| no-op | `patch/reference/manual_ddp.py` L17-L19 |  |
| skip frozen / None grad | `patch/reference/manual_ddp.py` L20-L22 |  |
| all-reduce sum | `patch/reference/manual_ddp.py` L23 |  |
| average grad | `patch/reference/manual_ddp.py` L24 |  |
| smoke fallback | `scripts/run_smoke.py` L28-L39 |  |

## 6. 结论

- 本次能证明：
- 本次不能证明：
- 如果失败，最小复现命令：
- 下一步要查的文件或日志：
