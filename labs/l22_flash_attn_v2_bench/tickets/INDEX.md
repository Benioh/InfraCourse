# Debug Tickets - L23

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `fa_dtype_falls_back_to_math` | 用 fp32 触发 math backend，speedup ≈ 1 | dtype 选 fp16/bf16 |
| `fa_install_glibc_mismatch` | `pip install flash-attn` 报 GLIBC 错 | 用 prebuilt wheel / 升 base image |
| `fa_head_dim_unsupported` | head_dim=96 触发 fall-back | FA2 支持的 head_dim 集合 |
| `fa_causal_attn_mask_double` | 同时传 attn_mask 与 is_causal 报错 | 二选一 |
| `fa_grad_NaN_with_dropout` | dropout > 0 时 NaN | training mode 必须 set seed |

工单 YAML 在 `InfraCourse/tickets/`。
