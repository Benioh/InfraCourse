# L03 · GPU Memory Snapshot：按调用栈定位泄露

这一讲解决长时间训练或 rollout 中的显存泄露定位问题：进程前几十步正常，显存每隔几步上涨一点，最后在某个 step OOM。`nvidia-smi` 只能给进程级总量；Memory Snapshot 把 alloc/free 事件、live allocations 和 Python 调用栈放到同一份证据里，让排查从“显存涨了”变成“哪条调用栈留下了多少 bytes”。

本讲的 lab 是 CPU 友好的 `MemoryTracker`。它不调用 PyTorch 的私有 CUDA snapshot API，只保留调试合同：记录分配、释放、当前仍存活的 allocation，并按 stack 聚合 top-k 泄露来源。真实 GPU snapshot、PyTorch caching allocator 和多进程 dump 会在讲义里解释边界。

## 学习路线

建议按下面顺序走，先把调试问题讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L03 在训练系统基础课里的位置。
2. 读 [lecture.md](lecture.md)：理解长期 OOM、snapshot 事件流、live set、caller stack 和聚合归因。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 starter、reference、tests 的顺序读源码。
4. 跑 notebook：[n21_memory_snapshot_walk.ipynb](../../notebooks/n21_memory_snapshot_walk.ipynb)，观察 hook 泄露和 stack 聚合。
5. 做 quiz：确认 snapshot 能回答什么、不能回答什么。
6. 做 patch：实现 `MemoryTracker`、`get_caller_stack` 和 `find_top_leaks_by_stack`。
7. 跑一次 CPU drill：用 reference 手动制造 leak，确认 top-k 排名。
8. 填写 [outputs/performance_metrics_template.md](outputs/performance_metrics_template.md)，沉淀一次 OOM 归因记录。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / debugging basics |
| 它解决什么问题 | 长时间运行后显存持续上涨时，怎样用 snapshot 找到未释放 allocation 的调用栈 |
| 它连接哪些指标或证据 | alloc/free events、live_allocations、total_leaked_bytes、top stack bytes、step 号、rank/worker |
| 它连接哪些源码 | `patch/starter/memory_snapshot.py`、`patch/reference/memory_snapshot.py`、`patch/tests/test_patch.py` |
| lab 检验什么 | start/stop、alloc/free、snapshot shape、double free safe、stack 获取和按 stack 聚合 top-k |

## 你会学到什么

- 区分单步 activation 峰值 OOM 和长期泄露型 OOM。
- 解释 events、live allocations、total leaked bytes 和 caller stack 的关系。
- 说明 PyTorch caching allocator 的 reserved memory 为什么不能直接等同于泄露。
- 用 `inspect.stack()` 把调用路径保存成可聚合的字符串 tuple。
- 实现一个最小 `MemoryTracker`，并让 double free、disabled tracker 等边界行为可预测。
- 用 top-k stack 聚合把大量 live allocations 转成可排查候选。
- 在真实多进程训练或 rollout 中选择正确 rank、worker 和时间点 dump snapshot。

## Patch 闭环

```bash
cat labs/l02.5_memory_snapshot/patch/task.md
$EDITOR labs/l02.5_memory_snapshot/patch/starter/memory_snapshot.py
make patch-test M=l02.5_memory_snapshot
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_disabled_tracker_returns_invalid_addr` | tracker 未 start 时不记录状态 |
| `test_basic_alloc_free_balance` | alloc 后 free 不留下 live allocation |
| `test_alloc_without_free_leaks` | 未释放 allocation 进入 snapshot |
| `test_double_free_is_safe` | 重复 free 和未知 addr 不抛错 |
| `test_dump_snapshot_shape` | snapshot 三个 key 和总 bytes 正确 |
| `test_find_top_leaks_groups_by_stack` | 同 stack 多次 alloc 会聚合 |
| `test_find_top_leaks_respects_k` | top-k 截断和降序排序生效 |
| `test_get_caller_stack_returns_tuple_of_strings` | caller stack 返回字符串 tuple |

## CPU Drill

不改 starter 时，可以先用 reference 跑一个最小泄露现场：

```bash
PYTHONPATH=labs/l02.5_memory_snapshot/patch python - <<'PY'
from reference.memory_snapshot import MemoryTracker, find_top_leaks_by_stack

tracker = MemoryTracker()
tracker.start()
stack_a = ("train.py:42:hook", "train.py:80:step")
stack_b = ("cache.py:17:append", "train.py:80:step")
tracker.alloc(1024, stack_a)
tracker.alloc(2048, stack_a)
addr = tracker.alloc(4096, stack_b)
tracker.free(addr)
snapshot = tracker.dump_snapshot()
print(snapshot["total_leaked_bytes"])
print(find_top_leaks_by_stack(snapshot, k=3))
PY
```

预期输出的泄露总量是 `3072`，top stack 应指向 `stack_a`。这条 drill 验证的是归因流程，不代表真实 CUDA allocator 行为。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查长期 OOM、snapshot dump 选点和 stack 聚合误读 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 MemoryTracker、reference 和测试主路径 |
| [outputs/performance_metrics_template.md](outputs/performance_metrics_template.md) | 记录一次显存泄露排查的命令、时间线、top stack 和结论 |

## 进入下一讲

`make patch-test M=l02.5_memory_snapshot` 通过，并能解释 top stack 为什么代表 live allocation 的来源后，进入 [L04 ManualDDP](../l03_nccl_ddp_smoke/README.md)。下一讲会把单进程训练证据扩展到多进程梯度同步。
