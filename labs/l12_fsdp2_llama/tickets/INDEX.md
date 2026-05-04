# Debug Tickets — L05.3

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `fsdp2_uneven_shard` | 模型某层 dim 不能被 world_size 整除导致 hang | even_dims_only / pad-then-shard |
| `fsdp2_mp_policy_dtype_mismatch` | reduce_dtype=fp16 训练后期 NaN | reduce 累加精度选 fp32 |
| `fsdp2_root_wrapped_first` | 学生先 wrap root 再 wrap blocks，反向时 reshard 顺序乱 | wrap 顺序 |
| `fsdp2_state_dict_full_oom` | save 时用 FULL_STATE_DICT 把所有 rank rank0 拉满 | SHARDED_STATE_DICT |
| `fsdp2_grad_none_after_step` | 用 `optimizer.zero_grad(set_to_none=True)` 但下一步 grad 为 None | DTensor + zero_grad 行为 |

工单 YAML 在 `InfraCourse/tickets/`。
