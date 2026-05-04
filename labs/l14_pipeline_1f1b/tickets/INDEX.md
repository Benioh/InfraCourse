# Debug Tickets — L05.7

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `pp_warmup_off_by_one` | stage 0 warmup 多/少一个 forward | num_stages-s-1 边界 |
| `pp_steady_double_F` | steady 阶段连续两个 F | 1F1B 严格交替 |
| `pp_micro_underrun` | num_microbatches < num_stages 不抛错 | 输入校验 |
| `pp_bubble_ratio_too_high` | 实测 bubble_ratio 远高于理论 | 检查 schedule 是否被错误填充 |
| `pp_interleaved_speedup_missing` | 切到 interleaved 没有获得 50% bubble 减少 | 后续可作为 stretch goal |

工单 YAML 在 `InfraCourse/tickets/`。
