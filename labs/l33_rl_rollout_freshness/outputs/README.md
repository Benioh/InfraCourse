# L38 课后产物：Rollout Freshness

本目录存放 L38 结束后可以复用的排查材料。它们用于真实 RL rollout 问题复盘，不是提交作业的格式。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 排查 stale rollout、版本字段缺失、局部更新失衡和 no-fresh-server failure |
| `source_reading_card.md` | 快速回忆 MiniInfra、patch、SLiME rollout 和 SGLang 版本证据主路径 |
| `rl_rollout_template.md` | 记录一次 drill 或真实 SLiME run 的配置、指标、源码定位和结论 |

每次跑完 patch、drill 或真实任务后，至少记录命令、配置、server 列表、`max_staleness`、update 策略、failure、p50/p99 staleness、reward/KL 和产物路径。没有版本证据时，不要直接把 reward 变化归因到模型或 reward parser。
