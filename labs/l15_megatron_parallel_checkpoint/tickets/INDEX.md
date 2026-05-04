# Debug Tickets — L05.8

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `ckpt_tp_mismatch` | resume 时 TP 改了但 strict load 居然不抛错 | parallel_state 校验、format 字段意义 |
| `ckpt_resume_lr_jump` | resume 后 loss 跳变 / LR 不连续 | scheduler_state、global_step 必须进 checkpoint |
| `ckpt_missing_marker` | save 之后没有 latest_checkpointed_iteration.txt | save 的"两件套"约定 |
| `ckpt_strict_silently_skips_unknown_key` | 校验放过了未知 ep 维度 | 校验项前向兼容性 |
| `ckpt_load_then_save_overwrites_iter` | resume 后又写到同一文件 | iteration 单调 / 文件命名规约 |

工单 YAML 统一放在 `InfraCourse/tickets/`。
