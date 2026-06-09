# L39 Debug Tickets

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `grpo_constant_reward_nan` | 同 group reward 全相等导致 std=0 | zero fallback |
| `grpo_clip_skipped_silently` | ratio 超过 clip 区间但 clipped fraction 为 0 | clip 分支和指标 |
| `grpo_kl_explodes_under_shift` | log_probs 与 reference 差值太大 | k3 KL 和输入 clamp |
| `grpo_advantage_shape_misuse` | 把 `[G]` advantage 当成 `[G,T]` 直接用 | unsqueeze / expand |
| `grpo_mask_average_off` | loss 直接 mean，没有除以 mask.sum | token-level normalization |

工单 YAML 在 `InfraCourse/tickets/`。
