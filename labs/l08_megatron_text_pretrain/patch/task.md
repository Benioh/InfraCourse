# L09 Patch · Cosine With Restarts LR Scheduler

## 你要交付什么

实现一个带热重启的 cosine LR scheduler。这个 patch 用来练习训练框架组件的最小合同：scheduler 需要保存计数状态，按 boundaries 计算学习率，把结果写入 optimizer 的所有 param groups，并在 total steps 之后钳到 `min_lr`。

接口如下：

```python
class CosineWithRestartsLR:
    def __init__(self, optimizer, max_lr, min_lr, restart_steps, total_steps): ...
    def step(self): ...
    def get_lr(self) -> float: ...
```

`restart_steps` 是升序 step 列表，例如 `[1000, 3000]`：

- step 0 到 999：第一段 cosine，从 `max_lr` 向 `min_lr` 衰减。
- step 1000：进入新段，lr 回到 `max_lr`。
- step 1000 到 2999：第二段 cosine。
- step 3000：再次进入新段。
- step 3000 到 4999：最后一段 cosine。
- step 5000 及之后：lr 固定为 `min_lr`。

禁止调用 `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`。本关需要你手写 boundaries、segment 查找和公式。允许使用 `math.cos` 和 `optimizer.param_groups` 等基础 API。

补丁规模目标：30 到 60 行。

## 接口契约

```python
optim = torch.optim.SGD(params, lr=0.0)
sched = CosineWithRestartsLR(
    optim,
    max_lr=1e-3,
    min_lr=1e-5,
    restart_steps=[1000, 3000],
    total_steps=5000,
)

for _ in range(5000):
    optim.zero_grad()
    loss = ...
    loss.backward()
    optim.step()
    sched.step()
    print(sched.get_lr())
```

## 不变量

1. 初始化后 lr 等于 `max_lr`。
2. boundaries 为 `[0] + restart_steps + [total_steps]`。
3. 段内使用相对位置 `step - segment_start` 计算 cosine。
4. restart step 本身属于新段起点，lr 等于 `max_lr`。
5. `step >= total_steps` 时 lr 等于 `min_lr`。
6. 所有 optimizer param groups 的 lr 同步更新。
7. `get_lr()` 返回 optimizer 第一个 param group 的当前 lr。

## 余弦公式

某一段内：

```text
t = step - segment_start
segment_len = segment_end - segment_start
ratio = t / segment_len
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * ratio))
```

段起点 `ratio=0`，lr 为 `max_lr`。段中点 `ratio=0.5`，lr 为 `(max_lr + min_lr) / 2`。接近段尾时，lr 接近 `min_lr`。

## 怎么验证

```bash
make patch-test M=l08_megatron_text_pretrain
```

7 个测试，CPU 即可：

| 测试 | 验证 |
|---|---|
| `test_initial_lr_is_max` | step 0 时 lr 等于 `max_lr` |
| `test_lr_decreases_within_segment` | 段内 lr 单调下降 |
| `test_lr_at_restart_step_is_max` | restart step 回到 `max_lr` |
| `test_lr_clamps_after_total` | step 超过 total 后钳到 `min_lr` |
| `test_multiple_param_groups_synced` | 多 param group 同步更新 |
| `test_get_lr_matches_optimizer` | `get_lr()` 与 optimizer 当前 lr 一致 |
| `test_no_restarts_is_pure_cosine` | `restart_steps=[]` 时退化为单段 cosine |

## 写完之后你要能解释

- 这个 scheduler 在训练 step 中应该放在 optimizer 更新前还是后。
- 为什么 restart step 是新 segment 起点。
- 为什么 `step_count` 是 checkpoint/resume 需要保存的状态。
- 为什么真实 Megatron 还要处理 warmup、sample-based increment、param group override、weight decay 和 canonical lr logging。
