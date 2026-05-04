# L10.5 Patch · 异步 Rollout Pool

## 你要交付什么

实现一个带并发上限的 **async rollout pool**——RL 训练 rollout 阶段的标准组件：

```python
class RolloutPool:
    def __init__(self, generate_fn: Callable[[str], Awaitable[str]], max_concurrency: int = 4): ...
    async def rollout(self, prompts: List[str]) -> List[str]:
        """并发跑 generate_fn(prompt)，返回与 prompts 同顺序的输出列表，
        并发数不超过 max_concurrency。"""
```

**禁止** 用 `aiomultiprocess` / `asyncio.gather` 之外的高级库（asyncio 内置 OK）。
**允许** asyncio.Semaphore / asyncio.gather / async/await。

补丁规模目标：30–50 行。

## 接口契约

```python
async def slow_gen(p):
    await asyncio.sleep(0.1)
    return f"Generated: {p}"

pool = RolloutPool(slow_gen, max_concurrency=3)
results = await pool.rollout(["a", "b", "c", "d", "e"])
assert results == ["Generated: a", ..., "Generated: e"]
# 5 prompts 用并发 3：总时间 ≈ 2 × 0.1s = 0.2s（不是 0.5s 串行）
```

## 不变量

1. 输出顺序与输入顺序一致（不能因为完成时间不同打乱）。
2. 任意时刻 generate_fn 同时运行的协程数 ≤ max_concurrency。
3. 空 prompts 列表返回空列表。
4. 任一 generate_fn 抛异常应原样传递给调用者（不吞错误）。
5. 5 prompt 在 max_concurrency=5 + 0.1s 延时下，总时间 < 0.2s（充分并发）。

## 怎么验证

```bash
make patch-test M=l31_rollout_only_smoke
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_basic_rollout` | 输出与 generate_fn 期望一致 |
| `test_preserves_order` | 不同 latency 下输出顺序仍正确 |
| `test_concurrency_bounded` | 同时运行的协程数 ≤ max_concurrency |
| `test_empty_input` | 空列表返回空列表 |
| `test_propagates_errors` | generate_fn 抛错时 rollout 抛错 |

## 写完之后你能做什么

- 解释 vLLM async engine / SGLang scheduler 的并发模型。
- 在 Capstone Stage C 写 RL rollout：并发调多模态 inference endpoint，控制 QPS 防止过载。
- 看懂 SLiME RolloutController 的接口设计。
