# Debug Tickets — L10.3

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `dpo_logp_includes_prompt` | reward 一直不动 | mask labels==-100 |
| `dpo_log_sigmoid_overflow` | β=20 时 loss = inf | 用 logsigmoid 不要 log(sigmoid) |
| `dpo_ref_grad_leak` | reference 模型也被更新 | ref.requires_grad_(False) + no_grad |
| `dpo_reward_inversion` | 训练后 chosen reward < rejected | 数据 swap 或 sign 错误 |
| `dpo_kl_blowup` | 多轮 DPO 后 policy 偏离 ref 太远 | β 调高 / 多轮迭代 / 重置 ref |

工单 YAML 在 `InfraCourse/tickets/`。
