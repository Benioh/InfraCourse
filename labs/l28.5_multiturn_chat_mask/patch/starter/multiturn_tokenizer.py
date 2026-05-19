"""
L28.5 Patch · Multi-turn chat tokenization with loss mask.

填空规则：
- TODO(student) 必须自己写
- 不许用 apply_chat_template(..., return_assistant_tokens_mask=True)
- 允许 apply_chat_template(messages, tokenize=False) + tokenizer.encode

完成度自检：
    make patch-test M=l28.5_multiturn_chat_mask
"""

from __future__ import annotations

from typing import List, Mapping, Tuple

# 选这个 BASE 的两个原则：
# 1. 必须有一个 system 占位，避免 default_system_mode 注入额外的 system；
# 2. 必须以 user 结尾，让后续添加任何 role 时，前文渲染都是稳定的。
BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def tokenize_with_loss_mask(
    messages: List[Mapping[str, str]],
    tokenizer,
) -> Tuple[List[int], List[int], List[int]]:
    """返回 (token_ids, loss_mask, attention_mask)。

    使用 Fixed Base Conversation Delta 算法：每条 message 单独
    apply_chat_template(BASE + [msg]) 得到完整字符串，截掉 BASE 部分得到 delta，
    再 encode 得到 delta_ids。assistant 的 delta_ids 进 loss target，其他 -100。
    """
    # TODO(student):
    #   base_str = tokenizer.apply_chat_template(BASE_CONVERSATION, tokenize=False)
    #
    #   token_ids: List[int] = []
    #   loss_mask: List[int] = []
    #
    #   for msg in messages:
    #       full_str = tokenizer.apply_chat_template(
    #           list(BASE_CONVERSATION) + [dict(msg)], tokenize=False
    #       )
    #       delta_str = full_str[len(base_str):]
    #       delta_ids = tokenizer.encode(delta_str, add_special_tokens=False)
    #
    #       token_ids.extend(delta_ids)
    #       if msg["role"] == "assistant":
    #           loss_mask.extend(delta_ids)
    #       else:
    #           loss_mask.extend([-100] * len(delta_ids))
    #
    #   attention_mask = [1] * len(token_ids)
    #   return token_ids, loss_mask, attention_mask
    raise NotImplementedError("L28.5: implement tokenize_with_loss_mask")
