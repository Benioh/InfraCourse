"""Tests for ai_tools: path safety + line slicing + search hit cap."""

from __future__ import annotations

import shutil

import pytest

from app.backend import ai_tools


def test_navigate_to_source_resolves_real_path() -> None:
    out = ai_tools.navigate_to_source(
        {"path": "app/backend/main.py", "lines": [1, 10]}
    )
    assert out["path"] == "app/backend/main.py"
    assert out["lines"] == [1, 10]
    assert out["url"] == "/source?path=app/backend/main.py&lines=1-10"


def test_navigate_to_source_swaps_inverted_lines() -> None:
    out = ai_tools.navigate_to_source(
        {"path": "app/backend/main.py", "lines": [42, 10]}
    )
    assert out["lines"] == [10, 42]


def test_navigate_to_source_rejects_traversal() -> None:
    out = ai_tools.navigate_to_source({"path": "../../../etc/passwd"})
    assert "error" in out


def test_navigate_to_source_rejects_outside_allowed_roots() -> None:
    out = ai_tools.navigate_to_source({"path": "/etc/hosts"})
    assert "error" in out


def test_navigate_to_source_rejects_directory() -> None:
    out = ai_tools.navigate_to_source({"path": "app/backend"})
    assert "error" in out


def test_navigate_to_source_omits_lines_when_missing() -> None:
    out = ai_tools.navigate_to_source({"path": "app/backend/main.py"})
    assert out["lines"] is None
    assert out["url"] == "/source?path=app/backend/main.py"


def test_read_file_full() -> None:
    out = ai_tools.read_file({"path": "app/backend/main.py"})
    assert "error" not in out
    assert out["language"] == "python"
    assert "FastAPI" in out["content"]


def test_read_file_slice() -> None:
    out = ai_tools.read_file(
        {"path": "app/backend/main.py", "line_start": 1, "line_end": 5}
    )
    assert "error" not in out
    assert out["sliced"] is True
    assert out["line_start"] == 1
    assert out["line_end"] == 5
    assert out["content"].count("\n") <= 4


def test_read_file_rejects_outside_root() -> None:
    out = ai_tools.read_file({"path": "../../etc/passwd"})
    assert "error" in out


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_search_code_finds_hits() -> None:
    out = ai_tools.search_code({"query": "FastAPI", "path_glob": "*.py"})
    assert "error" not in out
    assert out["count"] >= 1
    assert any("main.py" in hit["path"] for hit in out["hits"])
    assert all(hit["line"] is not None for hit in out["hits"])


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_search_code_caps_hits() -> None:
    out = ai_tools.search_code({"query": "the"})
    assert "error" not in out
    assert out["count"] <= ai_tools.SEARCH_HIT_CAP


def test_search_code_rejects_empty_query() -> None:
    out = ai_tools.search_code({"query": "   "})
    assert "error" in out


def test_dispatch_unknown_tool() -> None:
    out = ai_tools.dispatch("nope", {})
    assert "error" in out


def test_dispatch_routes_correctly() -> None:
    out = ai_tools.dispatch("read_file", {"path": "app/backend/main.py"})
    assert "error" not in out
