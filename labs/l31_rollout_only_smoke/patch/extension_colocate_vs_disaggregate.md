# L31 扩展 · CoLocate vs Disaggregate Placement

> 这是 L31 的可选扩展。**不在 patch-test 范围内**。
> 完成后你会理解 RL 框架最关键的 placement 设计选择，并能解释 verl / slime / OpenRLHF / AReaL 各自的取舍。

## 背景

RL 训练的 train engine 与 rollout engine 怎么放在 GPU 上有两种主流策略：

- **Co-locate**：两个 engine 在同一组 GPU 上，靠 offload/upload 轮流让出显存。优点是
  resource utilization 高，缺点是 weight sync 通过 IPC 在同进程间走（见 L32.5），
  以及 engine 切换时需要 memory savor pause/resume（见 L30.5）。verl 默认走这条路。
- **Disaggregate**：train 与 rollout 在不同 GPU 资源组，常驻。weight sync 通过 NCCL/IB
  跨组传递（见 L32.5 三种接口对比里的 `update_weights_from_distributed`）。优点是
  避免 offload/upload 开销，缺点是显存利用率低，rollout 扩缩容协议复杂。AReaL / SLiME
  支持这条路。

L31 主线 `RolloutPool` 与 placement 无关。本扩展让你把这两种 driver 都写出来。

## 你要扩展什么

在 `patch/starter/rollout_pool.py` 同目录新建 `patch/starter/placement_drivers.py`：

```python
class PlacementDriver:
    async def rollout_step(self, prompts: list[str]) -> list[str]: ...
    async def sync_weights(self, train_state: dict) -> None: ...

class CoLocateDriver(PlacementDriver):
    """每个 step：
        1. memory_savor.pause(train_state)
        2. memory_savor.resume(rollout_state)  # restore from previous pause
        3. await pool.rollout(prompts)
        4. memory_savor.pause(rollout_state)
        5. memory_savor.resume(train_state)
        6. (optional) sync_weights via IPC handle (L32.5 mechanism)
    """

class DisaggregateDriver(PlacementDriver):
    """train / rollout 各自常驻；sync 走 NCCL broadcast：
        - rollout 不 pause/resume
        - sync_weights 通过 dist.broadcast (跨进程) 推送
    """
```

## 不变量

1. `CoLocateDriver` 在每个 step 都触发 pause/resume，`DisaggregateDriver` 不触发。
2. 两种 driver 跑相同的 prompt 序列，输出 token 序列必须完全一致（除非显式注入 noise）。
3. `CoLocateDriver` 的 sync_weights 延迟主要是 handle gather + IPC，`DisaggregateDriver`
   的延迟主要是 NCCL broadcast bandwidth。

## 怎么验证

自己加 `patch/tests/test_placement.py`（可选）：

```python
def test_colocate_pause_resume_called_each_step():
    ...

def test_disaggregate_no_pause_during_step():
    ...

def test_both_produce_same_outputs():
    ...
```

## 写完之后你能做什么

- 在 RL 框架选型时给出 placement 推荐：什么情况下选 co-locate（小集群、资源紧），
  什么情况下选 disaggregate（大模型、动态扩缩 rollout 节点）。
- 解释 verl 默认 co-locate、AReaL 默认 disaggregate、SLiME 两者都支持的设计动机。
- 评估为多模态 RL 选哪种更合适（提示：图像 encoder 的 batch 弹性会影响选择）。

## 配套阅读

- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/sys-design/readme-1.md` —— 三种 weight sync 接口对比
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/areal/code-walk-through_CN.md` —— AReaL disaggregate
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/slime/code-walk-through/readme.md` —— SLiME placement
