# L03 Patch · Memory Snapshot · 按 Stack 定位泄露

## 你要交付什么

在 `patch/starter/memory_snapshot.py` 中实现一个 CPU 友好的 memory snapshot mock：

```python
@dataclass
class AllocEvent:
    addr: int
    size: int
    stack: tuple[str, ...]
    timestamp: float

def get_caller_stack(depth: int = 4) -> tuple[str, ...]: ...

class MemoryTracker:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def alloc(self, size: int, stack: tuple[str, ...]) -> int: ...
    def free(self, addr: int) -> None: ...
    def dump_snapshot(self) -> dict: ...

def find_top_leaks_by_stack(snapshot: dict, k: int = 3) -> list[tuple[tuple[str, ...], int]]: ...
```

**禁止** 调用 `torch.cuda.memory._record_memory_history` 或真实 CUDA snapshot API。
**允许** 使用 `inspect.stack()`、`dataclasses`、`time.monotonic()` 和标准容器。

补丁规模目标：60 到 110 行 Python。

## 接口契约

```python
tracker = MemoryTracker()
tracker.start()
stack = get_caller_stack()
addr1 = tracker.alloc(size=1024, stack=stack)
addr2 = tracker.alloc(size=8192, stack=stack)
tracker.free(addr1)
snapshot = tracker.dump_snapshot()

assert snapshot["total_leaked_bytes"] == 8192
assert len(snapshot["live_allocations"]) == 1
assert find_top_leaks_by_stack(snapshot, k=3)[0][1] == 8192
```

## 不变量

1. `start()` 把 tracker 切到 enabled；`stop()` 把 tracker 切到 disabled。
2. disabled 状态下 `alloc()` 返回 `-1`，并且不能修改 `_events`、`_live` 或 `_next_addr`。
3. enabled 状态下 `alloc()` 返回单调递增的新 addr，每次至少推进 `size + 64`。
4. `alloc()` 必须把 `AllocEvent` 同时写入 `_events` 和 `_live`。
5. `free(addr)` 在 disabled、负地址、未知地址或 double free 时安静返回。
6. 已知地址被 free 时，必须从 `_live` 移除，并向 `_events` 追加 `("free", ev)`。
7. `dump_snapshot()` 返回三个 key：`events`、`live_allocations`、`total_leaked_bytes`。
8. `total_leaked_bytes` 只统计当前 live allocations。
9. `find_top_leaks_by_stack()` 按完整 stack tuple 聚合 size，并按累计 bytes 降序返回前 k 个。
10. `get_caller_stack(depth)` 返回 tuple of string，元素格式为 `file:line:function`。

## 怎么验证

```bash
make patch-test M=l02.5_memory_snapshot
```

8 个测试，全是 CPU 友好：

| 测试 | 验证 |
|---|---|
| `test_disabled_tracker_returns_invalid_addr` | 未 start 时 `alloc()` 返回 `-1`，snapshot 为 0 leak |
| `test_basic_alloc_free_balance` | alloc 后 free 不留下 live allocation |
| `test_alloc_without_free_leaks` | 未释放 allocation 进入 live set |
| `test_double_free_is_safe` | 重复 free 和未知 addr 不抛错 |
| `test_dump_snapshot_shape` | snapshot 三键齐全，bytes 和 alloc event 正确 |
| `test_find_top_leaks_groups_by_stack` | 同 stack 聚合，按 size 降序 |
| `test_find_top_leaks_respects_k` | top-k 截断生效 |
| `test_get_caller_stack_returns_tuple_of_strings` | caller stack 是字符串 tuple，第一帧指向调用者 |

## 实现提示

- `inspect.stack()[1 : 1 + depth]` 可以跳过 `get_caller_stack` 自身。
- `time.monotonic()` 适合记录相对时间戳。
- `_events` 保存 `("alloc", ev)` 和 `("free", ev)`。
- `_live` 用 addr 做 key，value 是对应 `AllocEvent`。
- 聚合函数只读 snapshot，不应修改 tracker 状态。

## 写完之后你能做什么

- 解释 Memory Snapshot 为什么能把 OOM 从总量问题变成归因问题。
- 在真实训练或 rollout 中把 top stack、bytes、rank 和 step 区间写进 incident report。
- 看懂 PyTorch snapshot、hook 泄露、cache 未释放和 closure 捕获 tensor 的共同调试模式。
