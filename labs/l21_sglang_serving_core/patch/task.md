# L22 Patch · RadixCache Prefix Trie

## 你要交付什么

实现一个 SGLang-shaped 的前缀缓存 trie。它用 token id 序列作为 key，记录哪些前缀已经有可复用 KV，并在容量不足时按 LRU 删除叶子节点。

```python
class RadixCache:
    def __init__(self, max_tokens: int = 1024): ...

    def match_prefix(self, token_ids: List[int]) -> int:
        """返回 token_ids 在 cache 中已存在的最长前缀长度。"""

    def insert(self, token_ids: List[int]) -> int:
        """插入 token 序列，返回新增节点数。"""

    def total_tokens(self) -> int: ...

    def evict(self, num_tokens: int) -> int:
        """按 LRU 删除叶子节点，返回实际删除的 token 数。"""
```

禁止使用 `pyradix` 或其它第三方 trie 库。允许使用 `dict`、`list` 和 Python stdlib。

## 数据结构

教学版不做 path compression，每个 token id 是一个节点：

```python
class _Node:
    children: Dict[int, _Node]
    last_access: int
    parent: _Node | None
    token_from_parent: int | None
```

`parent` 和 `token_from_parent` 用于从叶子回删父节点的 child。`last_access` 用单调递增计数器维护 LRU。

## 不变量

1. 空 cache 对任意序列 `match_prefix` 都返回 0。
2. 插入 `[1, 2, 3]` 后，`match_prefix([1, 2, 3]) == 3`。
3. cache 中有 `[1, 2, 3]` 时，`match_prefix([1, 2, 9]) == 2`。
4. 同一序列重复插入时，第二次返回 0，`total_tokens()` 不增加。
5. `[1, 2, 3]` 与 `[1, 2, 4]` 共享 `[1, 2]`，节点数是 4。
6. `match_prefix` 和 `insert` 都要刷新访问时间，否则热前缀可能被误删。
7. `evict` 只删除叶子节点；如果父节点仍有其它 child，父节点不能被删除。

## 怎么验证

```bash
make patch-test M=l21_sglang_serving_core
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_empty_cache_match_zero` | 空 cache 匹配 0，节点数为 0 |
| `test_insert_then_match_full` | 插入后完整命中 |
| `test_partial_prefix_match` | query 和 cached sequence 可以部分前缀命中 |
| `test_repeated_insert_no_double_count` | 重复插入不重复计数 |
| `test_total_tokens_correct` | 共享前缀只存一份 |
| `test_evict_removes_lru` | LRU 叶子优先被删 |
| `test_evict_preserves_shared_prefix` | 删除一个叶子不会破坏仍被其它路径使用的共享前缀 |

## 写完之后你能做什么

- 解释 SGLang prefix cache 为什么能降低 repeated-prefix workload 的 TTFT。
- 看懂真实 SGLang `RadixCache` 为什么还需要 page alignment、KV indices、lock ref 和 namespace。
- 判断 repeated-prefix benchmark 的产物是否能支持“cache 提升了 TTFT”的结论。
