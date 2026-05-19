"""Tests for ai_config: provider config loading + token masking."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from app.backend import ai_config


@pytest.fixture(autouse=True)
def reset_provider_cache():
    ai_config._PROVIDER_CACHE.clear()
    yield
    ai_config._PROVIDER_CACHE.clear()


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


def _write_claude(home: Path, **env: str) -> None:
    cfg_dir = home / ".claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "settings.json").write_text(
        json.dumps({"env": env, "model": "claude-opus-4-7"}),
        encoding="utf-8",
    )


def _write_codex(home: Path, *, key: str, wire_api: str = "responses") -> None:
    cfg_dir = home / ".codex"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": key}), encoding="utf-8"
    )
    (cfg_dir / "config.toml").write_text(
        textwrap.dedent(
            f"""
            model = "gpt-5.5"
            model_provider = "aigw"
            model_reasoning_effort = "xhigh"

            [model_providers.aigw]
            name = "AI Gateway"
            base_url = "https://aigw.example.com/v1"
            wire_api = "{wire_api}"
            requires_openai_auth = true
            """
        ).strip(),
        encoding="utf-8",
    )


def test_get_provider_claude_from_settings(fake_home: Path) -> None:
    _write_claude(
        fake_home,
        ANTHROPIC_BASE_URL="https://aigw.example.com",
        ANTHROPIC_AUTH_TOKEN="sk-aigw-secret",
    )
    cfg = ai_config.get_provider("claude")
    assert cfg.base_url == "https://aigw.example.com"
    assert cfg.api_key == "sk-aigw-secret"
    assert cfg.model == "claude-opus-4-7"


def test_get_provider_claude_env_overrides_settings(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_claude(
        fake_home,
        ANTHROPIC_BASE_URL="https://from-file.example.com",
        ANTHROPIC_AUTH_TOKEN="from-file",
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://from-env.example.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "from-env")
    cfg = ai_config.get_provider("claude")
    assert cfg.base_url == "https://from-env.example.com"
    assert cfg.api_key == "from-env"


def test_get_provider_claude_missing_raises(fake_home: Path) -> None:
    with pytest.raises(LookupError):
        ai_config.get_provider("claude")


def test_get_provider_codex_responses_api(fake_home: Path) -> None:
    _write_codex(fake_home, key="sk-aigw-secret")
    cfg = ai_config.get_provider("codex")
    assert cfg.api_key == "sk-aigw-secret"
    assert cfg.base_url == "https://aigw.example.com/v1"
    assert cfg.model == "gpt-5.5"
    assert cfg.extras.get("reasoning_effort") == "xhigh"


def test_get_provider_codex_chat_wire_api_rejected(fake_home: Path) -> None:
    _write_codex(fake_home, key="sk-aigw-secret", wire_api="chat")
    with pytest.raises(LookupError, match="responses"):
        ai_config.get_provider("codex")


def test_get_provider_unknown(fake_home: Path) -> None:
    with pytest.raises(LookupError):
        ai_config.get_provider("ollama")


def test_mask_token_in_error(fake_home: Path) -> None:
    _write_claude(fake_home, ANTHROPIC_AUTH_TOKEN="sk-aigw-supersecret")
    try:
        ai_config.get_provider("claude")
    except LookupError as exc:
        assert "sk-aigw-supersecret" not in str(exc)
