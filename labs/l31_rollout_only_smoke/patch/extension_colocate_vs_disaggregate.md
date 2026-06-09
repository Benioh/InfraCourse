# L35 扩展 · Co-locate vs Disaggregate Placement

这份扩展不在 patch-test 范围内。它用来帮助你把 L35 的 rollout pool 放进真实 RL 资源布局里。

## 背景

RL 训练通常同时需要 train engine 和 rollout engine。两者有两种常见部署方式：

| 方式 | 特点 | 代价 |
|---|---|---|
| Co-locate | 训练和 rollout 共享同一组 GPU，靠 pause/resume、offload 或 upload 轮流使用显存 | 切换和显存整理成本高，调度实现复杂 |
| Disaggregate | 训练和 rollout 常驻不同 GPU 资源组，weight sync 跨组传参 | 显存利用率较低，扩缩容和同步协议更复杂 |

L35 主线只实现 `RolloutPool` 的并发控制，不实现 placement driver。扩展练习的目标是让你能解释：为什么并发上限、placement 和 weight sync 必须一起看。

## 可选扩展接口

在 `patch/starter/rollout_pool.py` 同目录新建 `placement_drivers.py`：

```python
class PlacementDriver:
    async def rollout_step(self, prompts: list[str]) -> list[str]: ...
    async def sync_weights(self, train_state: dict) -> None: ...


class CoLocateDriver(PlacementDriver):
    """每个 step 暂停训练状态，恢复 rollout 状态，运行 rollout，再切回训练。"""


class DisaggregateDriver(PlacementDriver):
    """train / rollout 常驻不同资源组，sync_weights 通过跨组通信推送权重。"""
```

## 行为要求

1. `CoLocateDriver` 每个 step 都触发 train/rollout 状态切换。
2. `DisaggregateDriver` 不在 rollout step 中 pause/resume 训练状态。
3. 两种 driver 对相同 prompt 序列应返回相同输出，除非显式注入采样噪声。
4. 复盘时记录 rollout time、sync time、GPU memory 和 server queue。

## 自检问题

- 小集群资源紧张时，为什么 co-locate 可能更合适？
- 大模型或多 rollout 节点场景下，为什么 disaggregate 更容易扩展？
- `max_concurrency`、server queue 和 weight sync interval 分别限制哪一层吞吐？
