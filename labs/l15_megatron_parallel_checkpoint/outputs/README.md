# L16 课后产物：Megatron Parallel Checkpoint

本目录存放 L16 之后可以继续复用的排查材料。它们用于记录 checkpoint save/load、parallel_state mismatch、latest marker、optimizer shard 和 scheduler step 的证据，适合在 CPU drill、Megatron dryrun 或真实训练 resume 后填写。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序排查 marker、format、parallel_state、optimizer/scheduler 和真实 shard load |
| `source_reading_card.md` | 快速回忆 patch、MiniInfra、Megatron checkpointing 和 distributed optimizer 主路径 |
| `checkpoint_debug_template.md` | 记录一次 checkpoint save/load 或 resume 事故的配置、指标、源码对应和结论 |

建议每次跑完 drill 或真实 resume 后，至少保存命令、配置、保存拓扑、当前拓扑、iteration、latest marker、warnings、strict 错误和 artifact 路径。没有这些证据时，不要把一次 load 成功当作训练恢复已经正确。
