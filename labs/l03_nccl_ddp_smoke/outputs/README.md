# L04 课后产物使用说明

本目录保存 ManualDDP 和分布式 smoke 的复用材料。它们面向真实排查，不只是课堂提交物。

## 文件说明

| 文件 | 用途 |
|---|---|
| `debug_checklist.md` | 排查 DDP hang、梯度放大、fallback 和 rank/world size 配置问题 |
| `source_reading_card.md` | 复习 ManualDDP、patch tests、smoke 和 MiniInfra collective 主路径 |
| `distributed_debug_template.md` | 记录一次分布式同步或 smoke 复盘，保留证据链和判断边界 |

## 建议使用顺序

1. 先用 `source_reading_card.md` 回忆本讲源码路径。
2. 运行 `make patch-test M=l03_nccl_ddp_smoke`，确认梯度同步语义。
3. 运行 `python labs/l03_nccl_ddp_smoke/scripts/run_smoke.py --mode smoke`，拿到 `ddp_hello.json`、`metrics.jsonl` 和 `report.md`。
4. 如果出现 hang、数值差异或 fallback，按 `debug_checklist.md` 从通信边界到梯度语义逐项排查。
5. 用 `distributed_debug_template.md` 写下本次运行的命令、artifact、指标和结论。

## 证据分级

| 证据 | 能说明什么 | 不能说明什么 |
|---|---|---|
| patch tests 全部通过 | `ManualDDP` 的平均梯度语义和 skip 边界成立 | NCCL 性能、多机稳定性 |
| `fallback_used=False` 的 smoke | 2-rank process group、all-reduce、barrier 跑通 | 梯度同步和训练收敛 |
| `fallback_used=True` 的 smoke | validation artifact 生成链路可用 | 真实 collective 通过 |
| `metrics.jsonl` | 本次运行的 world size、backend、all_reduce_sum、barrier 状态 | 端到端吞吐或 GPU 利用率 |
