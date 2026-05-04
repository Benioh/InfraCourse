# L04 Patch · Cosine With Restarts LR Scheduler

## 你要交付什么

实现一个**带热重启的 cosine LR scheduler**——这是 Megatron 没有内置但很多论文要求的调度策略：

```python
class CosineWithRestartsLR:
    def __init__(self, optimizer, max_lr, min_lr, restart_steps, total_steps): ...
    def step(self): ...                                # 每步调用，更新 optimizer 的 lr
    def get_lr(self) -> float: ...                     # 当前 lr
```

`restart_steps` 是一个升序 step 列表，比如 `[1000, 3000]` 表示：
- step 0 → 1000：从 max_lr 余弦衰减到 min_lr
- step 1000：**重启** lr 回到 max_lr
- step 1000 → 3000：再次余弦衰减到 min_lr
- step 3000：再次重启
- step 3000 → total_steps：最后一段衰减到 min_lr

**禁止** 用 `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` 偷懒（它有自己的语义）。
**允许** 用 `math.cos` / `optimizer.param_groups` 等基础 API。

补丁规模目标：30–60 行。

## 接口契约

```python
optim = torch.optim.SGD(params, lr=0.0)
sched = CosineWithRestartsLR(
    optim, max_lr=1e-3, min_lr=1e-5,
    restart_steps=[1000, 3000], total_steps=5000,
)

for step in range(5000):
    optim.zero_grad()
    loss = ...
    loss.backward()
    optim.step()
    sched.step()  # 每步更新 lr
    print(sched.get_lr())  # 当前 lr
```

## 不变量

1. step 0 时 lr == max_lr。
2. 在每段衰减末端（restart 之前一刻）lr → min_lr。
3. 在每个 restart_step 时刻 lr 重置回 max_lr。
4. step >= total_steps 时 lr 钳制到 min_lr。
5. 所有 optimizer.param_groups 的 lr 同步更新。

## 余弦公式

每段内（设当前位于第 i 段，相对位置 t ∈ [0, segment_len]）：

```
lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t / segment_len))
```

t=0：cos(0)=1，lr=max_lr ✓
t=segment_len：cos(π)=-1，lr=min_lr ✓
中间：余弦平滑过渡

## 怎么验证

```bash
make patch-test M=l08_megatron_text_pretrain
```

7 个测试（CPU 即可）：

| 测试 | 验证 |
|---|---|
| `test_initial_lr_is_max` | step 0 时 lr == max_lr |
| `test_lr_decreases_within_segment` | 段内 lr 单调递减 |
| `test_lr_at_restart_step_is_max` | restart_step 时 lr == max_lr |
| `test_lr_clamps_after_total` | step >= total_steps 时 lr == min_lr |
| `test_multiple_param_groups_synced` | 多个 param_group 同步更新 |
| `test_get_lr_matches_optimizer` | get_lr() 与 optimizer.param_groups[0]['lr'] 一致 |
| `test_no_restarts_is_pure_cosine` | restart_steps=[] 时退化为标准 cosine |

## 写完之后你能做什么

- 解释 PaLM-2、Mistral 等论文里 "linear warmup + cosine decay + restart" 的代码实现。
- 在 Capstone Stage A 给 projector 训练加一个 warm-restart 调度。
- 看懂 Megatron `optimizer_param_scheduler.py` 里其他 scheduler 的实现模式。
