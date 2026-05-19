# L02.5 · GPU Memory Snapshot：从堆栈定位泄露

> RL 训练里最让人崩溃的 OOM 不是"前向后向爆炸"，而是"看似一切正常，跑了 200 step 突然 OOM"。
> 这种通常是隐藏的 tensor 引用没释放（典型例子：closure 持有 tensor、KV cache 没 evict 干净、forward hook 累计输入）。
>
> 找它的标准武器是 `torch.cuda.memory._record_memory_history` 录制的 snapshot。
> 本关你写一个 CPU 友好的简化版：记录 alloc/free 事件，按 stack 聚合定位 top-K 泄露源。
> 训练过程中，这把刀的钝锋面：能告诉你"哪一行代码累计 leak 了多少 GB"。

## 真实事故

参考 [通过 Torch Memory Snapshot 分析 VLM RL 训练中的显存泄露问题](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/torch/mem-snapshot/readme.md)。
SGLang VLM RL 训练在 ~80 步出现奇怪的 OOM，常规 `torch.cuda.memory_summary()` 看不出问题。
最终是 dump 了一份 memory snapshot，按 stack 聚合后发现某个 hook 反复 retain 了 image tensor。

## 闭环

```bash
cat labs/l02.5_memory_snapshot/patch/task.md
$EDITOR labs/l02.5_memory_snapshot/patch/starter/memory_snapshot.py
make patch-test M=l02.5_memory_snapshot
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_disabled_tracker_returns_invalid_addr` | tracker 没 start 时 alloc 返回 -1 |
| `test_basic_alloc_free_balance` | alloc + free 不留泄露 |
| `test_alloc_without_free_leaks` | 只 alloc 不 free 进 leak 列表 |
| `test_double_free_is_safe` | 重复 free 不抛错 |
| `test_dump_snapshot_shape` | snapshot 三键齐全（events/live_allocations/total_leaked_bytes） |
| `test_find_top_leaks_groups_by_stack` | 同一 stack 的多次 alloc 聚合，按 size 降序 |
| `test_find_top_leaks_respects_k` | k=2 时只返回 2 项 |

## 卡住怎么办

1. 跑 `notebooks/n21_memory_snapshot_walk.ipynb` 看 PyTorch 真实 snapshot 的 JSON 长什么样。
2. `make patch-hint` 看 TODO；`make patch-show-solution` 看参考解。

## 写完之后你能做什么

- 在大模型训练 OOM 的时候立刻知道"哪个 hook / 哪个 forward / 哪个 closure 泄露最多"。
- 解释 `torch.cuda.memory._record_memory_history` 与 `torch.cuda.memory._dump_snapshot` 的工作机制。
- 看懂 SGLang `torch_memory_saver` 与 Megatron `CuMemAllocator` 的 release/restore 行为为什么需要 snapshot 验证。

## 配套源码研读（可选）

- `github_repo/Awesome-ML-SYS-Tutorial/torch/mem-snapshot/readme.md` — 真实事故复盘
- `pytorch.memory._snapshot.py` — 上游实现细节
