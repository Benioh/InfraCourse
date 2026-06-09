# Debug Tickets — L17

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `crash_tmp_loaded` | loader 读取了半写 `.tmp` | committed checkpoint 边界 |
| `crash_marker_ahead` | latest marker 指向不存在的 step | marker 更新顺序 |
| `crash_rng_missing` | 恢复后随机相关 loss 分叉 | RNG state 保存与恢复 |
| `crash_optimizer_reset` | resume 后 loss 从 crash step 后偏离 | optimizer state 恢复 |
| `crash_duplicate_step` | 同 step 重复 save 产生多个文件 | idempotent save |

工单 YAML 在 `InfraCourse/tickets/`。
