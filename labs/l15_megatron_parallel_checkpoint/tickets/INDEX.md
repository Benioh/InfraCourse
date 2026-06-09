# Debug Tickets — L16

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `ckpt_missing_marker` | checkpoint 文件存在但 latest marker 缺失 | 恢复入口必须由 marker 决定 |
| `ckpt_tp_mismatch` | 保存 TP 与加载 TP 不一致 | strict parallel_state 校验 |
| `ckpt_ep_introduced` | 新拓扑引入 EP 后旧 checkpoint 被接受 | 新并行轴也要进入 metadata |
| `ckpt_resume_lr_jump` | resume 后学习率跳变 | scheduler_state / opt_param_scheduler |
| `ckpt_optimizer_shard_shape` | optimizer state shape mismatch | distributed optimizer sharding metadata |

工单 YAML 在 `InfraCourse/tickets/`。
