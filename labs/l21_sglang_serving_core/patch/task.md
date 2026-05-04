# L08 Patch · RadixCache（前缀缓存 trie）

## 你要交付什么

实现 SGLang 风格的**前缀缓存 trie**——多请求共享相同 prompt 前缀，避免重复 KV 计算：

```python
class RadixCache:
    def __init__(self, max_tokens: int = 1024): ...
    def match_prefix(self, token_ids: List[int]) -> int:
        """返回 token_ids 在 cache 中最长前缀长度。"""
    def insert(self, token_ids: List[int]) -> int:
        """插入序列，返回新增 token 数（已存在的 prefix 不重复算）。"""
    def total_tokens(self) -> int: ...
    def evict(self, num_tokens: int) -> int:
        """LRU 驱逐 num_tokens 个 token，返回实际驱逐数。"""
```

**禁止** 用 `pyradix` / 第三方 trie 库。
**允许** `dict` / `collections` 等 stdlib。

补丁规模目标：60–100 行。

## 数据结构（简化版 radix tree）

我们用 dict-of-dict trie（不做 path compression，便于实现）：

```python
class _Node:
    children: Dict[int, _Node]   # token_id → child node
    last_access: int             # LRU 排序用
```

**match_prefix**：从 root 沿 token_ids 走，能走多深就走多深，返回长度。
**insert**：能匹配的部分走过去；剩余 token 创建新节点。每次访问更新 last_access。
**evict**：找叶子节点（无 children 的）中 last_access 最小的，删除；如果父节点变成叶子，递归。

## 不变量

1. `match_prefix(insert(seq))` 之后 `match_prefix(seq) == len(seq)`。
2. `match_prefix([t1, t2, t3])` 在 cache 只有 [t1, t2] 的情况下返回 2。
3. `insert` 同一序列两次：第二次 returns 0（无新增）。
4. `total_tokens()` == 实际节点数。
5. evict 后被 evict 的 prefix `match_prefix` 返回 0（除非该 prefix 仍是其它 inserted 序列的子前缀）。

## 怎么验证

```bash
make patch-test M=l21_sglang_serving_core
```

7 个测试：

| 测试 | 验证 |
|---|---|
| `test_empty_cache_match_zero` | 空 cache 匹配 0 |
| `test_insert_then_match_full` | 插入后能完全匹配 |
| `test_partial_prefix_match` | 短查询匹配长 cache 的前缀 |
| `test_repeated_insert_no_double_count` | 同序列插两次只算一次 |
| `test_total_tokens_correct` | total_tokens 反映节点数 |
| `test_evict_removes_lru` | 旧序列先被 evict |
| `test_evict_preserves_shared_prefix` | 共享前缀不被 evict（其它序列还在用）|

## 写完之后你能做什么

- 看懂 SGLang `radix_cache.py` 真实实现（带 path compression / GPU KV 块绑定）。
- 解释为什么 ChatGPT API 的 system prompt 能省钱（OpenAI 自动 prefix cache）。
- 在 Capstone Stage B 给多模态请求加 image-token prefix cache：同一张图的不同问题共享前缀。
