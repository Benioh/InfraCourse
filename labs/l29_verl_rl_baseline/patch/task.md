# L31 Patch · Adaptive KL Controller (PPO)

## 你要交付什么

实现 InstructGPT App C 风格的 adaptive KL coefficient controller。它根据实际 KL 与目标 KL 的偏差，自动调节 KL penalty 系数。

```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int): ...
    def update(self, current_kl: float, n_steps: int = 1) -> None: ...
    def get_coef(self) -> float: ...
```

限制：

- 禁止调用 `trl` 或 `verl` 的现成实现。
- 可以使用 Python 浮点运算和标准库。
- 补丁规模目标为 30 到 60 行。

## 算法

每次 update 使用下面的公式：

```python
proportional_error = current_kl / target_kl - 1.0
proportional_error = clip(proportional_error, -0.2, 0.2)
kl_coef *= 1.0 + proportional_error * n_steps / horizon
```

直觉：

- `current_kl > target_kl`：KL 过高，系数增大，下一步惩罚更重。
- `current_kl < target_kl`：KL 过低，系数减小，下一步允许更多探索。
- `current_kl == target_kl`：误差为 0，系数不变。
- `horizon` 越大，响应越慢；`horizon` 越小，响应越快。
- `clip` 限制单次误差，避免极端 KL 造成系数剧烈跳变。

## 不变量

1. 初始化后 `get_coef()` 返回 `init_kl_coef`。
2. 当前 KL 高于 target 时，coef 增大。
3. 当前 KL 低于 target 时，coef 减小。
4. 当前 KL 等于 target 时，coef 不变。
5. 极端 KL 的单步变化受 `[-0.2, 0.2]` clip 和 horizon 共同限制。
6. 更新使用乘法缩放，不能每次重置为初始值。

## 怎么验证

```bash
make patch-test M=l29_verl_rl_baseline
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_initial_coef` | 初始值等于 `init_kl_coef` |
| `test_increase_when_kl_too_high` | KL 过高时系数增大 |
| `test_decrease_when_kl_too_low` | KL 过低时系数减小 |
| `test_no_update_when_at_target` | KL 等于 target 时不变 |
| `test_clip_limits_max_change` | 极端 KL 单步只按 clip 后误差调节 |

Smoke：

```bash
python labs/l29_verl_rl_baseline/scripts/run_verl_lab.py --run-id l31_smoke
```

## 写完之后你能做什么

- 解释 RLHF/PPO 中 KL penalty 的控制作用。
- 判断 KL 爆炸时应检查 target、horizon、初始系数、reward 和 reference。
- 区分 controller bug、KL estimator bug、reward parser bug 和 rollout 慢。
