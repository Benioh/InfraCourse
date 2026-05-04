# L08 · SGLang Core：RadixCache（前缀缓存）

> 本关只做一件事：**实现一个 trie-based 前缀缓存**，支持 match_prefix / insert / LRU evict。

写完这关你能解释 SGLang / OpenAI API 的"prefix caching"省钱机制。

## 闭环

```bash
cat labs/l21_sglang_serving_core/patch/task.md
$EDITOR labs/l21_sglang_serving_core/patch/starter/radix_cache.py
make patch-test M=l21_sglang_serving_core
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_empty_cache_match_zero` | 空 cache 返回 0 |
| `test_insert_then_match_full` | 插入后完全匹配 |
| `test_partial_prefix_match` | 部分前缀匹配 |
| `test_repeated_insert_no_double_count` | 重复插入不重复计数 |
| `test_total_tokens_correct` | 节点数正确 |
| `test_evict_removes_lru` | LRU 优先 evict |
| `test_evict_preserves_shared_prefix` | 共享前缀不被错杀 |

## 卡住怎么办

1. 看 `notebooks/n09_prefix_cache.ipynb`。
2. `make patch-hint M=l21_sglang_serving_core`。
3. `make patch-show-solution M=l21_sglang_serving_core`。

## 进入下一关

`make patch-test` 全绿后，继续做源码理解口试。下一关 [L08.5 量化](../l23_quant_serving/README.md) 让你实现 W8A16 校准。
