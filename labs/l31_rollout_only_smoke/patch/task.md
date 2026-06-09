# L35 Patch · Async Rollout Pool

## 你要交付什么

实现一个带并发上限的 async rollout pool。它接收一组 prompt，并发调用 `generate_fn(prompt)`，但任意时刻同时运行的生成协程不能超过 `max_concurrency`。最终返回值必须和输入 prompt 的顺序一一对应。

```python
class RolloutPool:
    def __init__(
        self,
        generate_fn: Callable[[str], Awaitable[str]],
        max_concurrency: int = 4,
    ) -> None: ...

    async def rollout(self, prompts: List[str]) -> List[str]:
        """并发运行 generate_fn(prompt)，返回与 prompts 同顺序的结果。"""
```

允许使用 `asyncio.Semaphore`、`asyncio.gather` 和 `async/await`。不要使用 `aiomultiprocess`、进程池或第三方 async 框架。补丁规模目标是 30 到 50 行。

## 接口合同

```python
async def slow_gen(prompt: str) -> str:
    await asyncio.sleep(0.1)
    return f"Generated: {prompt}"

pool = RolloutPool(slow_gen, max_concurrency=3)
results = await pool.rollout(["a", "b", "c", "d", "e"])

assert results == [
    "Generated: a",
    "Generated: b",
    "Generated: c",
    "Generated: d",
    "Generated: e",
]
```

五个 prompt、每个 0.1 秒、并发上限 3 时，总耗时应接近两轮请求，而不是串行的五轮请求。

## 不变量

1. 输出列表顺序必须和输入 prompt 顺序一致。
2. 任意时刻运行中的 `generate_fn` 协程数不能超过 `max_concurrency`。
3. 空 prompt 列表返回空列表。
4. 任一 `generate_fn` 抛异常时，异常应传给调用者，不要吞掉。
5. 高并发场景下不能退化成串行调用。

## 验证命令

```bash
make patch-test M=l31_rollout_only_smoke
```

参考实现验收命令：

```bash
IMPL=reference make patch-test M=l31_rollout_only_smoke
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_basic_rollout` | 输出来自 `generate_fn` |
| `test_preserves_order` | 不同 latency 下输出顺序仍和输入一致 |
| `test_concurrency_bounded` | 同时运行的协程数不超过 `max_concurrency` |
| `test_empty_input` | 空列表返回空列表 |
| `test_propagates_errors` | `generate_fn` 抛错时继续向外抛 |

## 写完后要能解释

- `asyncio.Semaphore` 控制的是在途请求数，不是服务端 GPU batch size。
- `asyncio.gather` 按传入顺序返回结果，因此可以保留 prompt/response 对齐。
- 真实 vLLM/SGLang 会在服务端做动态 batching；客户端的职责是并发提交和限流。
