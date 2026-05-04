# L09.8 Patch · SFT loss-mask & chat template

## 你要交付什么

```python
def tokenize_chat_with_loss_mask(
    messages: list[dict],   # [{"role": "system"|"user"|"assistant", "content": str}, ...]
    tokenizer: Any,         # 必须支持 .encode + .pad_token_id (HF tokenizer)
    max_length: int,
    pad_token_id: int | None = None,
) -> dict:                  # {"input_ids": list[int], "labels": list[int], "attention_mask": list[int]}
    ...

def sft_loss(
    logits: torch.Tensor,   # [B, T, V]
    labels: torch.Tensor,   # [B, T] (含 -100)
    ignore_index: int = -100,
) -> torch.Tensor:          # scalar mean CE
    ...
```

补丁规模目标：60–100 行。

## 不变量

1. messages 顺序：system → user → assistant 交替；不允许两个连续 user。
2. **prompt token labels = -100；assistant 回答 token labels = 对应 input_id。**
3. 末尾必须有 EOS，且 EOS 也算入 assistant loss。
4. pad 部分 labels = -100。
5. `sft_loss` 必须与 `F.cross_entropy(ignore_index=-100)` 数值一致。
6. `attention_mask` 在 pad 处为 0，其他为 1。

## 怎么验证

```bash
make patch-test M=l28_sft_qwen_alpaca
```

## 写完之后你能做什么

- 在 Qwen / Llama / Mistral 上跑标准 SFT
- 解释为什么 TRL `SFTTrainer` 的 `dataset_text_field` 之外还有 `formatting_func`
- 给后续 RM / DPO / PPO 提供"会说话的"基底模型
