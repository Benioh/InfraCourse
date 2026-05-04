"""Reference solution for L09.8 Patch."""

from __future__ import annotations

from typing import Any

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ALLOWED_ROLES = {ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT}


def _validate_messages(messages: list[dict]) -> None:
    if not messages:
        raise ValueError("messages must not be empty")
    last_role = None
    for message in messages:
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unknown role: {role}")
        if role == ROLE_USER and last_role == ROLE_USER:
            raise ValueError("two consecutive user messages")
        last_role = role
    if messages[-1]["role"] != ROLE_ASSISTANT:
        raise ValueError("last message must be assistant for SFT loss masking")


def _resolve_pad(tokenizer: Any, fallback: int | None) -> int:
    if fallback is not None:
        return int(fallback)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def tokenize_chat_with_loss_mask(
    messages: list[dict],
    tokenizer: Any,
    max_length: int,
    pad_token_id: int | None = None,
) -> dict:
    _validate_messages(messages)
    pad_id = _resolve_pad(tokenizer, pad_token_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = pad_id

    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = message["role"]
        text = f"<|{role}|>{message['content']}"
        segment = tokenizer.encode(text, add_special_tokens=False)
        if role == ROLE_ASSISTANT:
            input_ids.extend(segment)
            labels.extend(segment)
            input_ids.append(eos_id)
            labels.append(eos_id)
        else:
            input_ids.extend(segment)
            labels.extend([-100] * len(segment))

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)
    pad_count = max_length - len(input_ids)
    if pad_count > 0:
        input_ids.extend([pad_id] * pad_count)
        labels.extend([-100] * pad_count)
        attention_mask.extend([0] * pad_count)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def sft_loss(logits, labels, ignore_index: int = -100):
    import torch.nn.functional as F  # noqa: N812

    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
