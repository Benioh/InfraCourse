# L11 · SLiME RL：Weight Sync Coordinator

> 本关只做一件事：**实现训练 ↔ 推理之间的 weight 同步逻辑**——shape/dtype 校验 + 跳过 mismatch + 统计字节数。

## 闭环

```bash
cat labs/l32_slime_rl_core/patch/task.md
$EDITOR labs/l32_slime_rl_core/patch/starter/weight_sync.py
make patch-test M=l32_slime_rl_core
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_basic_sync` | inference 数值与 train 完全相等 |
| `test_shape_mismatch_skipped` | shape 不一致进 mismatched_keys |
| `test_dtype_mismatch_skipped` | dtype 不一致也算 mismatch |
| `test_extra_train_keys_skipped` | train 多余 key 不创建到 inference |
| `test_stats_correct` | bytes_synced / num_tensors 数字对 |

## 卡住怎么办

`make patch-hint M=l32_slime_rl_core` / `make patch-show-solution`。

## 进入 Capstone

`make patch-test` 全绿后，下一关 [L11.5 versioned RolloutManager](../l33_rl_rollout_freshness/README.md) 会补 rollout 权重版本主线。
