# L17 课后产物：Crash Resume

本目录存放 L17 之后可以继续复用的排查材料。它们用于记录 crash-safe checkpoint 写入、dangling tmp、latest marker、恢复状态和 baseline/resume loss 对比，适合在 CPU drill、GPU smoke 或真实训练恢复事故后填写。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序排查 tmp 文件、marker、payload、RNG/optimizer 和 loss 分叉 |
| `source_reading_card.md` | 快速回忆 patch、drill、MiniInfra 和 Megatron checkpoint 主路径 |
| `checkpoint_debug_template.md` | 记录一次 crash-resume drill 或恢复事故的配置、指标、源码对应和结论 |

建议每次跑完 `run_crash_drill.sh` 或真实训练恢复后，至少保存命令、配置、seed、crash step、latest marker、tmp 文件状态、`max_rel_diff`、metrics 和 checkpoint artifact 路径。
