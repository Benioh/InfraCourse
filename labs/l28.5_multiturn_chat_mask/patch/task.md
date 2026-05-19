# L28.5 Patch · Multi-turn Tokenization with Loss Mask

## 你要交付什么

```python
def tokenize_with_loss_mask(
    messages: list[dict],   # [{"role": ..., "content": ...}, ...]
    tokenizer,              # 任何提供 .apply_chat_template / .encode 的对象
) -> tuple[list[int], list[int], list[int]]:
    """返回 (token_ids, loss_mask, attention_mask)。

    loss_mask[i] = token_ids[i]   if 第 i 个 token 来自 assistant 消息
                = -100            否则
    attention_mask 全 1（没有 padding）。
    """
```

**禁止** 直接用 `apply_chat_template(..., return_assistant_tokens_mask=True)`——
那只在少数模型上工作，且 QwQ / Qwen3 的 chat template 不支持。

**允许** `apply_chat_template(messages, tokenize=False)` 和 `tokenizer.encode(text, add_special_tokens=False)`。

补丁规模目标：30–70 行 Python。

## 算法（必须用 Fixed Base 思路）

```
BASE = [
    {"role": "system",  "content": "You are a helpful assistant."},
    {"role": "user",    "content": "I am a user."},
]

base_str = tokenizer.apply_chat_template(BASE, tokenize=False)

for msg in messages:
    full = tokenizer.apply_chat_template(BASE + [msg], tokenize=False)
    delta_str = full[len(base_str):]
    delta_ids = tokenizer.encode(delta_str, add_special_tokens=False)
    
    if msg["role"] == "assistant":
        loss_chunk = list(delta_ids)
    else:
        loss_chunk = [-100] * len(delta_ids)
    
    extend out
```

**为什么是 Fixed Base？**
- 一定有 system + user 在前面，模板不会再悄悄注入默认 system；
- 当前 message 永远是 BASE+[msg] 的最后一条，QwQ/Qwen3 不会砍它的 `<think>`；
- 每次 delta 渲染独立，不会因为前文 message 改变而出现"prev 比 curr 还长"的错位。

## 接口契约

```python
messages = [
    {"role": "system",    "content": "S"},
    {"role": "user",      "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "user",      "content": "Bye"},
    {"role": "assistant", "content": "<think>parse</think>OK"},
]

ids, loss, attn = tokenize_with_loss_mask(messages, tokenizer)

assert len(ids) == len(loss) == len(attn)
assert all(a == 1 for a in attn)
# Assistant 消息位置 loss == ids，其他位置 loss == -100
```

## 不变量

1. `len(ids) == len(loss) == len(attn)`。
2. 每个位置 `loss[i] in (-100, ids[i])`，没有第三种值。
3. `attn[i] == 1` 恒成立。
4. `[m for m in messages if m["role"] == "assistant"]` 数量为 0 时，`set(loss) == {-100}`。
5. `tool` / `function` / `system` / `user` 全部归为 mask -100 类（即非 assistant 一律 mask）。
6. 当 tokenizer 在某些情境下条件渲染（默认 system 注入 / think 砍除）时，
   Fixed Base 算法应当不受影响——assistant 的内容（含 think）必然出现在 loss 位置。

## 怎么验证

```bash
make patch-test M=l28.5_multiturn_chat_mask
```

7 个测试，全 CPU。本关附带一个 `_mock_tokenizer.py` 模拟 Qwen / QwQ 的两种麻烦行为
（默认 system 注入、think 砍除）。

## 卡住怎么办

1. 跑 `notebooks/n23_chat_template_multiturn.ipynb` 看完默认 system / think drop 的现象示例。
2. `make patch-hint` 看 TODO；`make patch-show-solution` 看参考解。

## 写完之后你能做什么

- 给 multi-turn agentic RL/SFT pipeline 打一致的 loss mask。
- 解释为什么单独 tokenize 子串 / messages 滑窗 delta 这两种朴素方案都不行。
- 看懂 verl `BASE_CHATML_FORMAT` 与对应模型注册机制。
- 在 RL 调试报告里加上 "训练-rollout-推理 三阶段 chat template 一致性检查"。
