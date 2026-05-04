# Debug Tickets — L04.8

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `mgt_lifecycle_train_loss_nan` | loss 一开始就是 NaN | precision、grad clip、bad batch 排查顺序 |
| `mgt_lifecycle_skip_step_storm` | 连续 50 步 `skipped_iter=1` | grad scaler / overflow detector / fp16 vs bf16 |
| `mgt_lifecycle_lr_no_restart` | restart_step 到了 LR 没回弹 | scheduler 每 step 调用一次的契约 |
| `mgt_lifecycle_loss_plateau` | loss 平稳但不下降 | 数据 shuffle、tokenizer、batch_size、warmup |
| `mgt_lifecycle_resume_drift` | 中途崩溃后重启 loss 跳变 | optimizer/scheduler/global_step 一并恢复 |

工单内容统一放在 `InfraCourse/tickets/<id>.yaml`。建议至少完成前两张作为通过证据。
