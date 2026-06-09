# L34 Patch · CUDA Graph Cache + Memory Savor

## 你要交付什么

实现两个 CPU-safe primitive：

```python
class GraphCache:
    def capture_or_replay(self, fn: Callable, *args, **kwargs) -> Any: ...

class MemorySavor:
    def pause(self, tensor: torch.Tensor) -> int: ...
    def resume(self, handle: int) -> torch.Tensor: ...
    def total_paused_bytes(self) -> int: ...
    def is_paused(self, handle: int) -> bool: ...
```

禁止 import `torch.cuda.graph`。本关没有 GPU 依赖，只验证结构语义。

## GraphCache 合同

`GraphCache` 根据输入生成 cache key：

- tensor 输入：`("tensor", shape, dtype)`
- scalar 输入：`("scalar", type_name, value)`
- kwargs：按 key 排序后进入 cache key，避免调用顺序影响命中

首次见到 key：

```text
self._graphs[key] = fn
self.capture_count += 1
return fn(*args, **kwargs)
```

再次见到相同 key：

```text
self.replay_count += 1
return self._graphs[key](*args, **kwargs)
```

注意：replay 使用首次 capture 时保存的函数引用，不使用本次传入的新函数。这模拟真实 CUDA Graph 已经录制好的执行路径。

## MemorySavor 合同

`pause(tensor)`：

1. 生成递增 handle。
2. 保存 shape、dtype 和 `tensor.detach().clone().cpu()`。
3. 返回 handle。

`resume(handle)`：

1. 从 `_paused` 中 `pop` 该 handle。
2. 返回保存数据的 clone。
3. resume 后 `is_paused(handle)` 应为 false。

`total_paused_bytes()` 返回所有暂停数据的 `numel * element_size` 总和。

## 不变量

1. 第一次同 shape 调用会增加 `capture_count`。
2. 第二次同 shape 调用会增加 `replay_count`。
3. 不同 shape 触发新的 capture。
4. replay 输出与 eager 函数输出数值一致。
5. pause 后 handle 处于 paused。
6. resume 返回与 pause 前相同的数据。
7. paused bytes 统计等于池内 tensor 数据规模。
8. resume 后 handle 从池中移除，bytes 归零。

## 怎么验证

```bash
make patch-test M=l30.5_cuda_graph_savor
```

8 个 CPU 测试：

| 测试 | 验证 |
|---|---|
| `test_first_call_captures` | 首次调用 capture，输出正确 |
| `test_second_call_replays` | 第二次同 shape replay |
| `test_different_shapes_distinct_graphs` | 不同 shape 新建 graph entry |
| `test_replay_output_equal_to_eager` | replay 与 eager 数值一致 |
| `test_savor_pause_records_metadata` | pause 后 handle 存在 |
| `test_savor_resume_returns_same_data` | resume 返回相同数据 |
| `test_total_paused_bytes_tracks_storage` | paused bytes 统计正确 |
| `test_resume_removes_from_paused_pool` | resume 后移除 handle 并归零 bytes |

## 边界

patch 不验证真实 GPU `data_ptr()`、CUDA stream、VMM physical page、graph replay latency 或多进程 co-locate。报告中必须把 CPU 已验证和 GPU 待验证分开写。
