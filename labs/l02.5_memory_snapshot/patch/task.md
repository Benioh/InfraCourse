# L02.5 Patch · Memory Snapshot · 按 Stack 定位泄露

## 你要交付什么

```python
@dataclass
class AllocEvent:
    addr: int
    size: int
    stack: tuple[str, ...]
    timestamp: float

class MemoryTracker:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def alloc(self, size: int, stack: tuple[str, ...]) -> int: ...
    def free(self, addr: int) -> None: ...
    def dump_snapshot(self) -> dict: ...

def get_caller_stack(depth: int = 4) -> tuple[str, ...]: ...
def find_top_leaks_by_stack(snapshot: dict, k: int = 3) -> list[tuple[tuple, int]]: ...
```

**禁止** 用 `torch.cuda.memory._record_memory_history`（本关 CPU 模拟）。
**允许** `inspect.stack()` / `dataclasses` / `time.monotonic()`。

补丁规模目标：60–110 行 Python。

## 接口契约

```python
tr = MemoryTracker()
tr.start()
addr1 = tr.alloc(size=1024, stack=get_caller_stack())
addr2 = tr.alloc(size=8192, stack=get_caller_stack())
tr.free(addr1)
snap = tr.dump_snapshot()

assert snap["total_leaked_bytes"] == 8192
assert len(snap["live_allocations"]) == 1
top = find_top_leaks_by_stack(snap, k=3)  # [(stack_tuple, total_size), ...]
```

## 不变量

1. `start` / `stop` 控制是否记录；`stop` 后 `alloc` 返回 `-1` 且不修改内部状态。
2. `alloc` 单调返回新 addr（每次至少加 size + 64 padding，避免地址相同）。
3. `free` 在 addr 不存在时安静返回（双 free 安全），并把对应 AllocEvent 从 `_live` 移除、加入 `_events`。
4. `dump_snapshot` 返回的 dict 必须有三个 key：`events`、`live_allocations`、`total_leaked_bytes`。
5. `find_top_leaks_by_stack` 按 stack tuple 聚合 size，按 size 降序返回前 k 个 `(stack, total_bytes)`。
6. `get_caller_stack(depth)` 返回 `tuple` of 字符串 `"file:line:func"`，长度等于 depth（或 stack 不够时取 max available）。

## 怎么验证

```bash
make patch-test M=l02.5_memory_snapshot
```

7 个测试：

| 测试 | 验证 |
|---|---|
| `test_disabled_tracker_returns_invalid_addr` | 未 start 时 alloc=-1 |
| `test_basic_alloc_free_balance` | alloc+free → 0 leak |
| `test_alloc_without_free_leaks` | 只 alloc 进 live_allocations |
| `test_double_free_is_safe` | 双 free 不抛错 |
| `test_dump_snapshot_shape` | 三键齐全 |
| `test_find_top_leaks_groups_by_stack` | 同 stack 聚合，size 降序 |
| `test_find_top_leaks_respects_k` | top k=2 只返 2 项 |

## 卡住怎么办

1. 跑 `notebooks/n21_memory_snapshot_walk.ipynb` 看 torch 原生 snapshot JSON 结构。
2. `make patch-hint` 看 TODO；`make patch-show-solution` 看参考解。

## 写完之后你能做什么

- 在 RL/LLM 训练 OOM 现场快速给出"top-3 泄露源代码位置"。
- 解释为什么仅看 `nvidia-smi` 经常误判（CUDA caching allocator）。
- 看懂 SGLang torch-memory-saver / Megatron CuMemAllocator 的 release/restore 流。
