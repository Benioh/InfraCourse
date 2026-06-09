"""L26 Patch tests · CPU."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.eval_harness")


def test_extract_first_number():
    impl = _impl()
    assert impl.extract_first_number("Answer: 42.") == 42
    assert impl.extract_first_number("Answer is 3.14") == pytest.approx(3.14)
    assert impl.extract_first_number("no number here") is None


def test_extract_first_number_signed_decimal():
    impl = _impl()
    assert impl.extract_first_number("loss=-1.5") == -1.5
    assert impl.extract_first_number("$1,234") == 1234
    assert impl.extract_first_number("3%") == 3


def test_score_exact_match():
    impl = _impl()
    assert impl.score_exact_match("  42 ", "42")
    assert impl.score_exact_match("Hello", "hello")
    assert not impl.score_exact_match("42", "43")


def test_score_first_number_match():
    impl = _impl()
    assert impl.score_first_number_match("Answer: 42", "$42.0")
    assert not impl.score_first_number_match("no", "42")
    assert impl.score_first_number_match("approximately 3.14159", "3.14160", atol=1e-3)


def test_format_few_shot_prompt_order():
    impl = _impl()
    prompt = impl.format_few_shot_prompt(
        "Q3?",
        [("Q1?", "A1"), ("Q2?", "A2")],
        system="be helpful",
    )
    assert prompt.index("Q1?") < prompt.index("Q2?") < prompt.index("Q3?")
    assert prompt.endswith("Answer:")
    assert "be helpful" in prompt


def test_run_evaluation_with_stub_client():
    impl = _impl()

    class Stub:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, max_tokens=256, stop=None):
            self.calls += 1
            return "Answer: 42"

    items = [
        {"question": f"Q{i}", "answer": "42"} for i in range(5)
    ]
    items.append({"question": "Q5", "answer": "42"})
    result = impl.run_evaluation(items, Stub(), n_shots=2, max_tokens=8)
    assert result["n_total"] == 4
    assert result["first_number_match"] == 1.0
    assert len(result["samples"]) == 4


def test_run_evaluation_partial_correct():
    impl = _impl()

    class Stub:
        def __init__(self):
            self.responses = iter(["41", "42", "43", "42"])

        def complete(self, prompt, max_tokens=256, stop=None):
            return next(self.responses)

    items = [
        {"question": f"Q{i}", "answer": "42"} for i in range(2)
    ] + [
        {"question": f"Q{i+2}", "answer": "42"} for i in range(4)
    ]
    result = impl.run_evaluation(items, Stub(), n_shots=2)
    assert result["n_total"] == 4
    assert result["first_number_match"] == 0.5
