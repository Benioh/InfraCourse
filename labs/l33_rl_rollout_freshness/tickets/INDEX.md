# Debug Tickets — L11.5

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `slime_stale_rollout_silent` | 所有 server stale 时 generate 默默使用旧权重 | RuntimeError 必须冒泡 |
| `slime_weight_sync_stall` | actor 持续 update 但 rollout 等不到 fresh | update 范围 / 频率 / staleness |
| `slime_subset_starvation` | 一直只更同一个 server，其他长期 stale | round-robin / 公平更新 |
| `slime_actor_version_overflow` | actor_version 不递增导致 staleness 计算错乱 | 单调性约束 |
| `slime_freshness_metric_missing` | 报告里没有 staleness 字段 | meta_info 必须包含 staleness |

工单 YAML 在 `InfraCourse/tickets/`。
