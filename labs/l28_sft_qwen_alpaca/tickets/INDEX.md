# Debug Tickets — L09.8

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `sft_eos_truncation` | 推理时模型不停说，根因是 SFT 数据末尾没 EOS | EOS 必须算入 assistant loss |
| `sft_loss_dropping_too_fast` | 100 步内 loss 几乎归零 → over-fit Alpaca 5k | lr / regularization / val split |
| `sft_user_loss_leak` | 模型学会复述 user prompt | labels 必须 -100 屏蔽 prompt |
| `sft_pad_token_collision` | pad_id == eos_id 导致 attention_mask 错 | pad 与 eos 必须在 mask 上区分 |
| `sft_role_drift` | 推理 chat template 与训练不一致 | tokenizer.apply_chat_template 一致性 |

工单 YAML 在 `InfraCourse/tickets/`。
