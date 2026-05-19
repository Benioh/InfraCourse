"""L28.5 Patch tests · CPU OK."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))

from _mock_tokenizer import MockTokenizer  # noqa: E402


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.multiturn_tokenizer")


SIMPLE_MESSAGES = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "user", "content": "Bye"},
    {"role": "assistant", "content": "OK"},
]


def test_returns_three_lists_of_equal_length():
    impl = _impl()
    tok = MockTokenizer()
    ids, loss, attn = impl.tokenize_with_loss_mask(SIMPLE_MESSAGES, tok)
    assert len(ids) == len(loss) == len(attn) > 0
    assert all(a == 1 for a in attn)


def test_loss_values_are_either_minus_100_or_token_id():
    impl = _impl()
    tok = MockTokenizer()
    ids, loss, _ = impl.tokenize_with_loss_mask(SIMPLE_MESSAGES, tok)
    for i, (tid, lm) in enumerate(zip(ids, loss)):
        assert lm == -100 or lm == tid, f"position {i}: id={tid}, loss={lm}"


def test_no_assistant_means_all_minus_100():
    impl = _impl()
    tok = MockTokenizer()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    _, loss, _ = impl.tokenize_with_loss_mask(msgs, tok)
    assert set(loss) == {-100}


def test_assistant_content_appears_in_loss_targets():
    impl = _impl()
    tok = MockTokenizer()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "ANSWER123"},
    ]
    ids, loss, _ = impl.tokenize_with_loss_mask(msgs, tok)
    decoded_loss = "".join(chr(t) for t in loss if t != -100)
    assert "ANSWER123" in decoded_loss, f"assistant content missing from loss: {decoded_loss!r}"


def test_multi_turn_alignment_both_assistants_in_loss():
    impl = _impl()
    tok = MockTokenizer()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "FIRST_ANS"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "SECOND_ANS"},
    ]
    _, loss, _ = impl.tokenize_with_loss_mask(msgs, tok)
    decoded = "".join(chr(t) for t in loss if t != -100)
    assert "FIRST_ANS" in decoded
    assert "SECOND_ANS" in decoded
    # User content must NOT be in loss
    assert "Q1" not in decoded
    assert "Q2" not in decoded


def test_handles_default_system_injection():
    """When tokenizer auto-injects DEFAULT_SYS for messages without leading system,
    the algorithm using BASE prefix should still produce correct masks for the
    user's actual messages (no DEFAULT_SYS leaks into output).
    """
    impl = _impl()
    tok = MockTokenizer(default_system_mode=True)
    # No leading system in the conversation we're tokenizing:
    msgs = [
        {"role": "user", "content": "ASK"},
        {"role": "assistant", "content": "REPLY"},
    ]
    ids, loss, _ = impl.tokenize_with_loss_mask(msgs, tok)
    decoded_all = "".join(chr(t) for t in ids)
    decoded_loss = "".join(chr(t) for t in loss if t != -100)
    assert "REPLY" in decoded_loss
    assert "ASK" not in decoded_loss
    # DEFAULT_SYS should NOT appear in the output at all (because BASE has its own system)
    assert "DEFAULT_SYS" not in decoded_all, f"default system leaked: {decoded_all!r}"


def test_handles_qwq_think_drop():
    """When qwq_mode strips <think> from non-last assistants, the BASE-trick algorithm
    should still preserve think tokens because each delta renders msg as last.
    """
    impl = _impl()
    tok = MockTokenizer(qwq_mode=True)
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "<think>SECRET</think>ANS1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "ANS2"},
    ]
    _, loss, _ = impl.tokenize_with_loss_mask(msgs, tok)
    decoded_loss = "".join(chr(t) for t in loss if t != -100)
    assert "<think>SECRET</think>" in decoded_loss, (
        f"think tokens should be preserved by fixed-base trick; got: {decoded_loss!r}"
    )
    assert "ANS1" in decoded_loss
    assert "ANS2" in decoded_loss


def test_tool_message_treated_as_non_assistant():
    impl = _impl()
    tok = MockTokenizer()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "calling tool"},
        {"role": "tool", "content": "TOOL_RESULT"},
        {"role": "assistant", "content": "FINAL"},
    ]
    _, loss, _ = impl.tokenize_with_loss_mask(msgs, tok)
    decoded_loss = "".join(chr(t) for t in loss if t != -100)
    assert "TOOL_RESULT" not in decoded_loss, "tool content must be masked"
    assert "FINAL" in decoded_loss
    assert "calling tool" in decoded_loss
