"""Reference solution for L30 Patch · Multi-turn Chat Mask."""

from __future__ import annotations

from typing import List, Mapping, Tuple

BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def tokenize_with_loss_mask(
    messages: List[Mapping[str, str]],
    tokenizer,
) -> Tuple[List[int], List[int], List[int]]:
    base_str = tokenizer.apply_chat_template(BASE_CONVERSATION, tokenize=False)

    token_ids: List[int] = []
    loss_mask: List[int] = []

    for msg in messages:
        full_str = tokenizer.apply_chat_template(
            list(BASE_CONVERSATION) + [dict(msg)], tokenize=False
        )
        delta_str = full_str[len(base_str):]
        delta_ids = tokenizer.encode(delta_str, add_special_tokens=False)

        token_ids.extend(delta_ids)
        if msg.get("role") == "assistant":
            loss_mask.extend(delta_ids)
        else:
            loss_mask.extend([-100] * len(delta_ids))

    attention_mask = [1] * len(token_ids)
    return token_ids, loss_mask, attention_mask
