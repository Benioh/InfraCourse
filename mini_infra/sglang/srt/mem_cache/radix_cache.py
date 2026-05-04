from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RadixKey:
    token_ids: list[str]
    extra_key: str | None = None

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass
class MatchPrefixParams:
    key: RadixKey


@dataclass
class MatchResult:
    matched_prefix_len: int
    value: list[str] | None


@dataclass
class InsertParams:
    key: RadixKey
    value: list[str]
    priority: int = 0


@dataclass
class InsertResult:
    prefix_len: int


@dataclass
class TreeNode:
    children: dict[tuple[str, str | None], "TreeNode"] = field(default_factory=dict)
    value: list[str] | None = None
    priority: int = 0


class RadixCache:
    """Minimal SGLang-shaped radix cache.

    The real SGLang cache accepts ``MatchPrefixParams`` / ``InsertParams`` and
    returns structured results. MiniInfra preserves that API and namespace
    behavior while shrinking KV indices to string tokens.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.root_node = TreeNode(priority=-1)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        node = self.root_node
        best_length = 0
        best_value = None
        for index, token in enumerate(params.key.token_ids, start=1):
            child_key = (token, params.key.extra_key)
            if child_key not in node.children:
                break
            node = node.children[child_key]
            if node.value is not None:
                best_length = index
                best_value = node.value
        return MatchResult(matched_prefix_len=best_length, value=best_value)

    def insert(self, params: InsertParams) -> InsertResult:
        node = self.root_node
        for token in params.key.token_ids:
            node = node.children.setdefault((token, params.key.extra_key), TreeNode())
        old_prefix_len = len(params.key.token_ids) if node.value is not None else 0
        node.value = params.value
        node.priority = params.priority
        return InsertResult(prefix_len=old_prefix_len)

    def cache_finished_req(
        self, req: Any, is_insert: bool = True
    ) -> InsertResult | None:
        if not is_insert:
            return None
        token_ids = list(req.prompt_tokens)
        if not token_ids:
            return None
        return self.insert(
            InsertParams(
                key=RadixKey(token_ids=token_ids), value=["kv"] * len(token_ids)
            )
        )

    def evict(self, key: RadixKey) -> bool:
        node = self.root_node
        for token in key.token_ids:
            child_key = (token, key.extra_key)
            if child_key not in node.children:
                return False
            node = node.children[child_key]
        node.value = None
        return True
