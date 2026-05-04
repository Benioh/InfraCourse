# L04.5 · 长上下文：Ring Attention Forward

> 本关只做一件事：**用 online softmax 写一个 ring attention forward**——FlashAttention 与 Context Parallel 的核心算法。

写完这关你能解释 FlashAttention 论文 Algorithm 1 每一步。

## 闭环

```bash
cat labs/l09_long_context_cp/patch/task.md
$EDITOR labs/l09_long_context_cp/patch/starter/ring_attention.py
make patch-test M=l09_long_context_cp   # 5 个测试，CPU OK
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_matches_full_attention_one_chunk` | num_chunks=1 与 F.SDPA `allclose` |
| `test_matches_full_attention_four_chunks` | num_chunks=4 同上 |
| `test_handles_uneven_chunks` | Sk=7, num_chunks=2 |
| `test_long_seq` | Sk=1024, num_chunks=8 |
| `test_grads_are_continuous` | backward 不报错 |

## 卡住怎么办

1. 看 `notebooks/n14_context_parallel_ringattn.ipynb`。
2. `make patch-hint M=l09_long_context_cp`。
3. `make patch-show-solution M=l09_long_context_cp`。

## 进入下一关

`make patch-test` 全绿后，下一关 [L04.8 Megatron lifecycle](../l10_megatron_pretrain_lifecycle/README.md) 会把训练主线接回真实源码。
