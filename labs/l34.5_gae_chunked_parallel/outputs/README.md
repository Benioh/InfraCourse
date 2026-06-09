# L40 课后产物：GAE Chunked Parallel

本目录存放 L40 后续复盘可以复用的材料。它们用于记录数值等价、源码定位和真实性能证据，不是提交作业的格式。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 排查 GAE 方向、last_value、chunk boundary、余数 chunk 和性能结论 |
| `source_reading_card.md` | 快速回忆 patch 与 SLiME GAE 主路径 |
| `rl_rollout_template.md` | 记录数值测试、源码对应、GPU 条件和结论 |

每次跑完 patch 或真实 benchmark 后，至少记录命令、shape、T、chunk_size、last_value、最大误差、硬件和计时范围。没有这些证据时，不要把数值等价写成生产加速结论。
