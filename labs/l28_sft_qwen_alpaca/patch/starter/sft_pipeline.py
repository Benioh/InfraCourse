"""L09.8 Patch · SFT loss-mask + chat tokenization."""

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


def tokenize_chat_with_loss_mask(
    messages: list[dict],
    tokenizer: Any,
    max_length: int,
    pad_token_id: int | None = None,
) -> dict:
    _validate_messages(messages)
    # TODO(student): pick pad_token_id (default to tokenizer.pad_token_id, fallback to eos)
    # TODO(student): for each message, encode role+content with tokenizer.encode (no special tokens)
    #               and append a separator token (e.g. eos) between segments
    # TODO(student): for each token, set labels[i] = input_ids[i] if it belongs to an assistant
    #               segment (including the trailing EOS), else labels[i] = -100
    # TODO(student): truncate to max_length, then pad with pad_token_id; pad labels = -100; mask = 0
    # TODO(student): return {"input_ids", "labels", "attention_mask"} as plain lists[int]
    raise NotImplementedError("L09.8: implement tokenize_chat_with_loss_mask")


def sft_loss(logits, labels, ignore_index: int = -100):
    """Mean cross-entropy that ignores -100 labels."""
    # TODO(student): shift labels (causal LM convention) — this lab assumes pre-shifted
    # TODO(student): compute F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
    #               ignore_index=ignore_index, reduction="mean")
    raise NotImplementedError("L09.8: implement sft_loss")
