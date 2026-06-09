# L38 Patch · SLiME Rollout Freshness

## 你要交付什么

实现一个带版本准入的 `RolloutManager`。它管理多台 `RolloutServer`，只允许足够新的 server 生成 rollout，并支持把 actor 的新版本更新到全部或部分 server。

```python
class RolloutManager:
    def generate(self, prompts, actor_version) -> RolloutData: ...
    def update_weights(self, version, server_ids=None) -> list[str]: ...
    def freshness(self, actor_version) -> dict[str, int]: ...
```

## 不变量

1. `staleness = actor_version - server.weight_version`。
2. `generate` 必须按 round-robin 顺序寻找 server，并跳过 `staleness > max_staleness` 的 server。
3. 没有足够新的 server 时必须抛 `RuntimeError`，不能继续使用 stale 权重。
4. `update_weights(version)` 默认更新所有 server。
5. `update_weights(version, server_ids=[...])` 只更新指定 server，并返回实际更新的 server id。
6. `freshness(actor_version)` 返回每个 server 当前落后 actor 的步数。

## 怎么验证

```bash
make patch-test M=l33_rl_rollout_freshness
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l33_rl_rollout_freshness
```

drill：

```bash
python labs/l33_rl_rollout_freshness/scripts/run_rl_drill.py --run-id l38_local
```

## 写完之后你能做什么

- 解释 SLiME 中 actor 权重同步后，rollout engine 还需要哪些版本证据。
- 判断 stale rollout 是吞吐取舍还是训练稳定性风险。
- 给 RL debug 报告补上 `actor_version`、`weight_version`、`staleness` 和 `server_id`。
