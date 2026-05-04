# Debug Tickets — L11.8

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `grpo_constant_reward_NaN` | 同 group reward 全相等 → std=0 → div 0 | guard with eps + zero-fallback |
| `grpo_clip_skipped_silently` | clip_eps=10 但 ratio 仍被 clamp 报告为 clipped | 理解 min(unclipped, clipped) |
| `grpo_kl_explodes_under_distribution_shift` | log_probs - log_probs_ref 大时 exp() 爆 | clamp 输入 / k3 estimator |
| `grpo_advantage_per_token_misuse` | 把 [G] advantage 当 [G,T] 用 | unsqueeze 维度 |
| `grpo_mask_average_off` | 直接 mean 而不是除 mask.sum() | 显式 token-level normalization |

工单 YAML 在 `InfraCourse/tickets/`。
