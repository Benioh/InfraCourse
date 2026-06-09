# L29 Debug Tickets

| Ticket | Stage | Symptom | First Check |
|---|---|---|---|
| `sft_user_loss_leak` | Tokenization | 模型学会复述 user prompt | 检查 user segment labels 是否全为 `-100` |
| `sft_assistant_mask_empty` | Tokenization | loss 很低但模型没有学到回答 | 检查 assistant token 是否被保留到 labels |
| `sft_eos_missing` | Generation | 推理时回答不停止 | 检查 assistant 段末尾 EOS 是否进入 labels |
| `sft_pad_loss_leak` | Padding | pad token 被模型学习 | 检查 pad 的 labels 和 attention mask |
| `sft_template_mismatch` | Serving / Train | 训练可用，推理格式漂移 | 对齐训练 tokenizer 和推理 chat template |
