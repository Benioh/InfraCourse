# L11.7 · GAE 的分 chunk 并行计算（100-300× 加速）

> **真实背景**：标准 GAE 是反向递推 $A_t = \delta_t + \gamma\lambda A_{t+1}$，
> 数学上"必须"串行。但 RL 长上下文场景里 (T = 32K+)，CPU/Python 上跑这条链
> 实测占整个训练 step 的 30%+。slime 团队把它改成了"分 chunk 并行 + boundary
> correction"：chunk 内部仍递推，但 chunk 之间可以并行；只要传一个边界值就能
> 把数学对齐回 naive 版。在 slime 中实测**约 100-300× 加速**。
>
> 数学结论本身很短，只是没人注意。本关把它写出来，体会"算法不变 + 工程精明 =
> 巨大收益"这种最美的优化。

灵感来源：`Awesome-ML-SYS-Tutorial / rlhf/slime/batch-GAE/ppo-gae-chunk.md`。

## 闭环

```bash
cat labs/l34.5_gae_chunked_parallel/patch/task.md
$EDITOR labs/l34.5_gae_chunked_parallel/patch/starter/gae_chunk.py
make patch-test M=l34.5_gae_chunked_parallel
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_naive_known_values` | 短序列手算结果对得上 |
| `test_chunked_matches_naive_short` | T=10, chunk=3 时数值精确等价 |
| `test_chunked_matches_naive_long` | T=1024, chunk=64 时数值等价 |
| `test_chunked_handles_remainder` | T 不被 chunk_size 整除时也对 |
| `test_chunked_one_chunk_equals_naive` | chunk_size ≥ T 退化成 naive |
| `test_terminal_value_propagates` | last_value 在最后一个 chunk 边界正确处理 |

## 卡住怎么办

`make patch-hint M=l34.5_gae_chunked_parallel` / `make patch-show-solution`。
