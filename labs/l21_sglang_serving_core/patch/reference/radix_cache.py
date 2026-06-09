"""Reference solution for L22 Patch · RadixCache."""

from __future__ import annotations

from typing import Dict, List, Optional


class _Node:
    __slots__ = ("children", "last_access", "parent", "token_from_parent")

    def __init__(self, parent: Optional["_Node"] = None, token_from_parent: Optional[int] = None):
        self.children: Dict[int, "_Node"] = {}
        self.last_access: int = 0
        self.parent = parent
        self.token_from_parent = token_from_parent


class RadixCache:
    def __init__(self, max_tokens: int = 1024):
        self.root = _Node()
        self.max_tokens = max_tokens
        self._access_counter = 0
        self._total = 0

    def total_tokens(self) -> int:
        return self._total

    def match_prefix(self, token_ids: List[int]) -> int:
        node = self.root
        matched = 0
        for tid in token_ids:
            if tid in node.children:
                node = node.children[tid]
                matched += 1
                self._access_counter += 1
                node.last_access = self._access_counter
            else:
                break
        return matched

    def insert(self, token_ids: List[int]) -> int:
        node = self.root
        new_tokens = 0
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = _Node(parent=node, token_from_parent=tid)
                new_tokens += 1
                self._total += 1
            node = node.children[tid]
            self._access_counter += 1
            node.last_access = self._access_counter
        if self._total > self.max_tokens:
            self.evict(self._total - self.max_tokens)
        return new_tokens

    def evict(self, num_tokens: int) -> int:
        evicted = 0
        while evicted < num_tokens:
            leaves = self._collect_leaves()
            if not leaves:
                break
            lru = min(leaves, key=lambda n: n.last_access)
            parent = lru.parent
            assert parent is not None and lru.token_from_parent is not None
            del parent.children[lru.token_from_parent]
            evicted += 1
            self._total -= 1
        return evicted

    def _collect_leaves(self) -> List[_Node]:
        leaves: List[_Node] = []

        def _dfs(node: _Node) -> None:
            if not node.children and node is not self.root:
                leaves.append(node)
                return
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root)
        return leaves
