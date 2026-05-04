"""L09.8 Patch tests · CPU."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.sft_pipeline")


class _CharTokenizer:
    """Tiny char-level tokenizer with pad/eos."""

    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [(ord(ch) % 200) + 8 for ch in text]


def test_chat_template_concat_order():
    impl = _impl()
    out = impl.tokenize_chat_with_loss_mask(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ],
        _CharTokenizer(),
        max_length=64,
    )
    assert len(out["input_ids"]) == 64
    assert out["input_ids"][:1] != [0]


def test_loss_mask_blocks_prompt_tokens():
    impl = _impl()
    tok = _CharTokenizer()
    out = impl.tokenize_chat_with_loss_mask(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        tok,
        max_length=32,
    )
    user_segment = tok.encode("<|user|>hello")
    for i in range(len(user_segment)):
        assert out["labels"][i] == -100


def test_loss_mask_keeps_assistant_tokens():
    impl = _impl()
    tok = _CharTokenizer()
    out = impl.tokenize_chat_with_loss_mask(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        tok,
        max_length=32,
    )
    assistant_segment = tok.encode("<|assistant|>yo")
    masked = [label for label in out["labels"] if label != -100]
    # assistant tokens + EOS
    assert masked[: len(assistant_segment)] == assistant_segment
    assert masked[-1] == tok.eos_token_id


def test_pad_token_handled():
    impl = _impl()
    out = impl.tokenize_chat_with_loss_mask(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        _CharTokenizer(),
        max_length=32,
    )
    assert out["attention_mask"][-1] == 0
    assert out["labels"][-1] == -100


def test_sft_loss_ignores_minus_100():
    impl = _impl()
    logits = torch.zeros(1, 4, 5)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    labels = torch.tensor([[1, -100, 2, -100]])
    loss = impl.sft_loss(logits, labels)
    assert torch.isfinite(loss)


def test_sft_loss_matches_torch_ce():
    impl = _impl()
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 7)
    labels = torch.tensor([[1, -100, 3, 5, -100, 0], [-100, 2, 4, -100, -100, 6]])
    expected = F.cross_entropy(
        logits.reshape(-1, 7), labels.reshape(-1), ignore_index=-100, reduction="mean"
    )
    actual = impl.sft_loss(logits, labels)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_rejects_consecutive_user():
    impl = _impl()
    with pytest.raises(ValueError):
        impl.tokenize_chat_with_loss_mask(
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
            _CharTokenizer(),
            max_length=16,
        )


def test_rejects_no_assistant_at_end():
    impl = _impl()
    with pytest.raises(ValueError):
        impl.tokenize_chat_with_loss_mask(
            [{"role": "user", "content": "a"}], _CharTokenizer(), max_length=16
        )
