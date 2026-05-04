# L10 Patch · Adaptive KL Controller (PPO)

## 你要交付什么

实现 InstructGPT / OpenAI PPO 用的 **adaptive KL controller**——根据实际 KL 与目标 KL 的偏差，自动调节 KL 惩罚系数：

```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int): ...
    def update(self, current_kl: float, n_steps: int = 1) -> None: ...
    def get_coef(self) -> float: ...
```

**禁止** 用 `trl` / `verl` 的现成实现。
**允许** stdlib `math` / Python 浮点运算。

补丁规模目标：30–60 行。

## 算法（Ouyang 2022 InstructGPT, App C）

每 step 计算 `proportional_error` 并按 horizon 长度缩放：

```python
proportional_error = clip(current_kl / target_kl - 1, -0.2, 0.2)
kl_coef *= 1 + proportional_error * n_steps / horizon
```

直觉：
- 当前 KL **太高**（current > target）→ error 正，coef 增大 → 下一步惩罚更重 → KL 下降
- 当前 KL **太低**（current < target）→ error 负，coef 减小 → 下一步允许更多漂移 → KL 上升
- clip(±0.2) 防止 coef 单步剧烈震荡

`horizon` 是控制器响应速度：horizon 大则慢调（稳定），小则快调（敏感）。

## 不变量

1. `current_kl == target_kl` 时 update 不改 coef（error=0）。
2. `current_kl >> target_kl` 时 coef **增大**。
3. `current_kl << target_kl` 时 coef **减小**。
4. 单次极端 KL 不会让 coef 翻倍（clip 限制）。
5. coef 永不为负（数学要求）。

## 接口契约

```python
ctrl = AdaptiveKLController(init_kl_coef=0.2, target_kl=0.05, horizon=10000)
for step in range(50):
    current_kl = compute_kl_from_rollout(...)
    ctrl.update(current_kl, n_steps=1)
    kl_loss = ctrl.get_coef() * current_kl
    total_loss = pg_loss + kl_loss
    total_loss.backward()
```

## 怎么验证

```bash
make patch-test M=l29_verl_rl_baseline
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_initial_coef` | get_coef() 初始值 = init_kl_coef |
| `test_increase_when_kl_too_high` | current > target → coef 增大 |
| `test_decrease_when_kl_too_low` | current < target → coef 减小 |
| `test_no_update_when_at_target` | current == target → coef 不变 |
| `test_clip_limits_max_change` | 极端 KL 单步变化 ≤ 20% |

## 写完之后你能做什么

- 解释 PPO 论文里 KL controller 的全部数学。
- 在 verl / SLiME 调 RL 时知道为什么 KL 会发散（horizon 太小、init_kl_coef 太低、target_kl 太严）。
- Capstone Stage C 的 RL 阶段直接用这个 controller 防止 reward hacking。
