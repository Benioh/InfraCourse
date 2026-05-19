"""Provider config loader for the in-app AI tutor.

Reads `~/.claude/settings.json` and `~/.codex/{auth.json,config.toml}` at first
access and caches the result. Honors `CLAUDE_HOME` / `CODEX_HOME` env overrides
so tests can point at fixture directories.

Tokens never enter logs or response bodies. Use `mask()` whenever a config field
needs to surface in diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    extras: dict[str, Any] = field(default_factory=dict)


def _claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_HOME") or Path.home() / ".claude")


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def mask(token: str | None) -> str:
    if not token:
        return "<empty>"
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}…{token[-4:]}"


def load_claude_config() -> ProviderConfig | None:
    settings_path = _claude_home() / "settings.json"
    if not settings_path.exists():
        return None
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("failed to parse %s: %s", settings_path, exc)
        return None
    env = data.get("env") or {}
    # Env vars win over settings.json so an operator can override per-process
    # without editing the file the Claude Code CLI also writes to.
    base_url = (
        os.environ.get("ANTHROPIC_BASE_URL")
        or env.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com"
    )
    token = (
        os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
        or env.get("ANTHROPIC_AUTH_TOKEN")
        or env.get("ANTHROPIC_API_KEY")
    )
    if not token:
        logger.warning("claude config: no auth token found")
        return None
    return ProviderConfig(
        name="claude",
        base_url=base_url.rstrip("/"),
        api_key=token,
        model=data.get("model") or "claude-opus-4-7",
        extras={"effort": data.get("effortLevel")},
    )


def load_codex_config() -> ProviderConfig | None:
    auth_path = _codex_home() / "auth.json"
    config_path = _codex_home() / "config.toml"

    if not auth_path.exists():
        return None
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("failed to parse %s: %s", auth_path, exc)
        return None
    api_key = auth.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    if not config_path.exists():
        logger.warning("codex auth.json present but config.toml missing")
        return None
    try:
        cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.error("failed to parse %s: %s", config_path, exc)
        return None

    model = cfg.get("model") or "gpt-5.5"
    provider_name = cfg.get("model_provider")
    if not provider_name:
        logger.warning("codex config has no model_provider")
        return None
    providers = cfg.get("model_providers") or {}
    provider = providers.get(provider_name)
    if not provider:
        logger.warning("codex config missing [model_providers.%s]", provider_name)
        return None
    base_url = provider.get("base_url")
    wire_api = provider.get("wire_api") or "chat"
    if not base_url:
        return None
    if wire_api != "responses":
        raise RuntimeError(
            f"codex wire_api must be 'responses' for this gateway, got {wire_api!r}. "
            "Edit ~/.codex/config.toml to set wire_api = \"responses\"."
        )

    return ProviderConfig(
        name="codex",
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        extras={
            "wire_api": wire_api,
            "reasoning_effort": cfg.get("model_reasoning_effort"),
            "provider_name": provider_name,
        },
    )


# Cache holds one of: ProviderConfig (loaded ok), Exception (load tried but
# raised — surface to caller), None (load tried but config absent).
# An empty dict means "not loaded yet"; clearing it forces a reload.
_PROVIDER_CACHE: dict[str, ProviderConfig | Exception | None] = {}


def all_providers() -> dict[str, ProviderConfig]:
    """Return the dict of usable providers. Loads on first call (or after clear)."""
    if not _PROVIDER_CACHE:
        for name, loader in (
            ("claude", load_claude_config),
            ("codex", load_codex_config),
        ):
            try:
                _PROVIDER_CACHE[name] = loader()
            except Exception as exc:  # noqa: BLE001 - cache to surface via get_provider
                _PROVIDER_CACHE[name] = exc
                logger.warning("loading %s config raised: %s", name, exc)
        for name, val in _PROVIDER_CACHE.items():
            if isinstance(val, ProviderConfig):
                logger.info(
                    "ai provider ready: %s base=%s model=%s key=%s",
                    name,
                    val.base_url,
                    val.model,
                    mask(val.api_key),
                )
    return {k: v for k, v in _PROVIDER_CACHE.items() if isinstance(v, ProviderConfig)}


def get_provider(name: str) -> ProviderConfig:
    all_providers()  # ensure loaded
    cached = _PROVIDER_CACHE.get(name)
    if isinstance(cached, ProviderConfig):
        return cached
    if isinstance(cached, Exception):
        # Tokens never appear in our exception messages, so str(exc) is safe.
        raise LookupError(str(cached)) from cached
    raise LookupError(f"provider {name!r} not configured")


def reset_cache() -> None:
    """Test-only: drop cached providers so the next call re-reads the filesystem."""
    _PROVIDER_CACHE.clear()
