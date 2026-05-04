# L10 · verl RL baseline：Adaptive KL Controller

> 本关只做一件事：**实现 InstructGPT / PPO 的 adaptive KL coefficient controller**。

## 闭环

```bash
cat labs/l29_verl_rl_baseline/patch/task.md
$EDITOR labs/l29_verl_rl_baseline/patch/starter/kl_controller.py
make patch-test M=l29_verl_rl_baseline
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_initial_coef` | 初始 coef = init_kl_coef |
| `test_increase_when_kl_too_high` | current > target → coef 增大 |
| `test_decrease_when_kl_too_low` | current < target → coef 减小 |
| `test_no_update_when_at_target` | current == target → coef 不变 |
| `test_clip_limits_max_change` | 极端 KL 单步变化受 ±0.2 clip 限制 |

## 卡住怎么办

1. 看 `notebooks/n10_rl_kl_reward.ipynb`。
2. `make patch-hint M=l29_verl_rl_baseline`。
3. `make patch-show-solution M=l29_verl_rl_baseline`。

## 进入下一关

`make patch-test` 全绿后，继续做源码理解口试。下一关 [L10.5 Rollout-only smoke](../l31_rollout_only_smoke/README.md)。
