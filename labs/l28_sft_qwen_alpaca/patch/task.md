# L29 Patch · SFT Loss Mask 与 Chat Tokenization

## 目标

补齐两个函数：

```python
def tokenize_chat_with_loss_mask(
    messages: list[dict],
    tokenizer: Any,
    max_length: int,
    pad_token_id: int | None = None,
) -> dict:
    ...

def sft_loss(logits, labels, ignore_index: int = -100):
    ...
```

## 行为合同

1. `messages` 不能为空，role 只允许 `system`、`user`、`assistant`。
2. 不允许两个连续 `user` 消息。
3. 最后一条消息必须是 `assistant`，因为 SFT 样本需要一个回答目标。
4. 每条消息按 `<|role|>{content}` 编码，使用 `tokenizer.encode(..., add_special_tokens=False)`。
5. `system` 和 `user` token 的 label 全部写 `-100`。
6. `assistant` token 的 label 等于同位置 `input_ids`。
7. assistant 段末尾追加 EOS，EOS 也进入 assistant loss。
8. 超出 `max_length` 时截断；不足时用 pad 补齐，pad 的 label 为 `-100`，attention mask 为 `0`。
9. `sft_loss` 必须等价于 `F.cross_entropy(..., ignore_index=-100, reduction="mean")`。

## 验证命令

```bash
make patch-test M=l28_sft_qwen_alpaca
```

参考实现：

```bash
IMPL=reference make patch-test M=l28_sft_qwen_alpaca
```

## 完成后应能解释

- 为什么 prompt token 不进入 loss。
- 为什么 assistant EOS 应进入 loss。
- pad token 和 EOS token 可以相同，但 attention mask 与 labels 仍要区分 pad 位置。
- `ignore_index=-100` 对 loss 分母有什么影响。
