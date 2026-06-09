# 课后产物：L36 SLiME Weight Sync Coordinator

本目录保存三份可复用材料，用来复盘 actor -> rollout 权重同步、shape/dtype mismatch 和同步耗时。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 排查 sync 慢、版本未推进、shape/dtype mismatch、engine lock 和 stale rollout |
| `source_reading_card.md` | 快速回忆 patch、MiniInfra、SLiME、Megatron updater 和 SGLang engine 主路径 |
| `rl_rollout_template.md` | 记录一次 actor/rollout 资源切分、sync 指标和版本判断 |

使用这些材料时，先保存命令、resolved config、metrics、validation artifact 和 report。没有 `bytes_synced`、`weight_sync_time_sec`、`weight_version` 或 mismatch 列表时，不要判断同步健康。
