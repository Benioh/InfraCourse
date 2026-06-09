# L38 Debug Tickets

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `slime_stale_rollout_silent` | 所有 server stale 时 generate 仍使用旧权重 | no-fresh-server 必须抛错 |
| `slime_weight_sync_stall` | actor 持续 update，但 rollout 等不到 fresh server | update 范围、频率和 staleness 阈值 |
| `slime_subset_starvation` | 一直只更新同一批 server，其他 server 长期 stale | round-robin 观察和 per-server freshness |
| `slime_actor_version_stuck` | actor_version 不递增，staleness 无法反映训练推进 | actor 版本单调性 |
| `slime_freshness_metric_missing` | 报告里没有 actor/server 版本字段 | meta_info 和 metrics 证据完整性 |

工单 YAML 在 `InfraCourse/tickets/`。
