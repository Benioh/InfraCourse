# L30 Patch · Multi-turn Tokenization with Loss Mask

## 目标

补齐：

```python
def tokenize_with_loss_mask(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int], list[int]]:
    ...
```

返回：

- `token_ids`：chat template delta 编码后的 token id。
- `loss_mask`：assistant 位置等于同位置 token id，其它 role 为 `-100`。
- `attention_mask`：本关不做 padding，全部为 `1`。

## 必须使用 Fixed Base Conversation + Delta

```python
BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]

base_str = tokenizer.apply_chat_template(BASE_CONVERSATION, tokenize=False)

for msg in messages:
    full_str = tokenizer.apply_chat_template(
        list(BASE_CONVERSATION) + [dict(msg)],
        tokenize=False,
    )
    delta_str = full_str[len(base_str):]
    delta_ids = tokenizer.encode(delta_str, add_special_tokens=False)
```

若 `msg["role"] == "assistant"`，当前 `delta_ids` 同时进入 `token_ids` 和 `loss_mask`。其它 role 只进入 `token_ids`，`loss_mask` 写等长 `-100`。

## 不变量

1. `len(token_ids) == len(loss_mask) == len(attention_mask)`。
2. `loss_mask[i]` 只允许是 `-100` 或同位置 `token_ids[i]`。
3. `attention_mask` 全部为 `1`。
4. 没有 assistant 消息时，`loss_mask` 只能包含 `-100`。
5. `system`、`user`、`tool`、`function` 等非 assistant 消息都不进入 loss。
6. default system injection 不应泄漏进输出。
7. QwQ 风格 `<think>...</think>` 在 assistant loss 中应保留。

## 禁止

不要直接调用：

```python
apply_chat_template(..., return_assistant_tokens_mask=True)
```

本关要你显式写出模板渲染、delta 截取、role mask 和 attention mask。

## 验证命令

```bash
make patch-test M=l28.5_multiturn_chat_mask
```

参考实现：

```bash
IMPL=reference make patch-test M=l28.5_multiturn_chat_mask
```
