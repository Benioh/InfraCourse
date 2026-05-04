# Debug Tickets — L05.8.5

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `crash_partial_corrupts_load` | save 中途 SIGKILL，下次 load 拿到半文件 | atomic rename + tmp 清理 |
| `crash_lr_scheduler_drift` | resume 后 loss 抖动，因为 scheduler 没存 | scheduler / global step 也要存 |
| `crash_rng_state_lost` | resume 后 dropout 模式不一致 | torch / numpy / cuda RNG 全部存 |
| `crash_optimizer_momentum_lost` | resume 后训练像热启动 | optimizer state 必存 |
| `crash_concurrent_two_writers` | 多进程同时 save 同 step 抢文件 | 必须 rank0 唯一写 |

工单 YAML 在 `InfraCourse/tickets/`。
