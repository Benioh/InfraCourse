# L10.5 · Async Rollout Pool

> 本关只做一件事：**用 asyncio.Semaphore 实现一个 max_concurrency 受限的 async rollout pool**。

## 闭环

```bash
cat labs/l31_rollout_only_smoke/patch/task.md
$EDITOR labs/l31_rollout_only_smoke/patch/starter/rollout_pool.py
make patch-test M=l31_rollout_only_smoke
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_basic_rollout` | 输出对应 generate_fn |
| `test_preserves_order` | 不同延迟下顺序仍正确 |
| `test_concurrency_bounded` | 同时运行 ≤ max_concurrency |
| `test_empty_input` | 空列表 → 空列表 |
| `test_propagates_errors` | generate_fn 抛错正确传递 |

## 卡住怎么办

`make patch-hint M=l31_rollout_only_smoke` / `make patch-show-solution`

## 进入下一关

下一关 [L11 SLiME](../l32_slime_rl_core/README.md) 让你写 RL 训练 ↔ rollout 的 weight sync。
