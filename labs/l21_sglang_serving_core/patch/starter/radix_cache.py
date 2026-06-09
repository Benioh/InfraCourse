"""
L22 Patch · RadixCache (prefix cache trie)

填空规则：
- TODO(student) 必须自己写
- 不许 import pyradix 等第三方库
- 允许 dict / collections.OrderedDict

完成度自检：
    make patch-test M=l21_sglang_serving_core
"""

from __future__ import annotations

from typing import Dict, List


class _Node:
    __slots__ = ("children", "last_access", "parent", "token_from_parent")

    def __init__(self, parent: "_Node | None" = None, token_from_parent: int | None = None):
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
        """返回 token_ids 在 cache 中已存在的最长前缀长度，并更新 LRU 时间戳。"""
        # TODO(student):
        #   node = self.root; matched = 0
        #   for tid in token_ids:
        #       if tid in node.children:
        #           node = node.children[tid]
        #           matched += 1
        #           self._access_counter += 1
        #           node.last_access = self._access_counter
        #       else:
        #           break
        #   return matched
        raise NotImplementedError("L22 Patch: implement match_prefix")

    def insert(self, token_ids: List[int]) -> int:
        """插入序列，返回新增 token 数。"""
        # TODO(student):
        #   node = self.root
        #   new_tokens = 0
        #   for tid in token_ids:
        #       if tid not in node.children:
        #           node.children[tid] = _Node(parent=node, token_from_parent=tid)
        #           new_tokens += 1
        #           self._total += 1
        #       node = node.children[tid]
        #       self._access_counter += 1
        #       node.last_access = self._access_counter
        #   if self._total > self.max_tokens:
        #       self.evict(self._total - self.max_tokens)
        #   return new_tokens
        raise NotImplementedError("L22 Patch: implement insert")

    def evict(self, num_tokens: int) -> int:
        """LRU 驱逐 num_tokens 个 token；返回实际驱逐数。

        策略：找所有 **叶子节点**（无 children 的）中 last_access 最小的，删之；
        删除后如果父节点变成叶子，下一轮可能也被 evict。
        """
        # TODO(student):
        #   evicted = 0
        #   while evicted < num_tokens:
        #       leaves = self._collect_leaves()  # 遍历整个 trie 收集 children=={} 的非 root 节点
        #       if not leaves: break
        #       lru_leaf = min(leaves, key=lambda n: n.last_access)
        #       parent = lru_leaf.parent
        #       del parent.children[lru_leaf.token_from_parent]
        #       evicted += 1
        #       self._total -= 1
        #   return evicted
        raise NotImplementedError("L22 Patch: implement evict")

    def _collect_leaves(self) -> List[_Node]:
        # TODO(student): DFS 遍历 self.root，返回所有 children == {} 且非 root 的节点
        raise NotImplementedError("L22 Patch: implement _collect_leaves")
