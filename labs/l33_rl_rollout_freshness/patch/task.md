# L11.5 Patch · SLiME-shaped Versioned Rollout Manager

## 你要交付什么

实现 `mini_infra/slime/ray/rollout.py` 的增强版：RolloutServer 带 `weight_version`，RolloutManager 只从足够新的 server 生成 rollout，并能更新指定 server 的权重版本。

```python
class RolloutManager:
    def generate(self, prompts, actor_version) -> RolloutData: ...
    def update_weights(self, version, server_ids=None) -> list[str]: ...
    def freshness(self, actor_version) -> dict[str, int]: ...
```

## 不变量

1. `staleness = actor_version - server.weight_version`。
2. `generate` 必须跳过 staleness 超过 `max_staleness` 的 server。
3. 没有足够新的 server 时必须抛错，而不是悄悄用 stale 权重。
4. `update_weights(version)` 默认更新所有 server。
5. `update_weights(version, server_ids=[...])` 只更新指定 server，并返回更新列表。

## 怎么验证

```bash
make patch-test M=l33_rl_rollout_freshness
```

## 写完之后你能做什么

- 解释 SLiME 中 TrainRayActor.update_weights 如何影响 rollout engine。
- 判断 stale rollout 是吞吐优化还是训练稳定性风险。
- 给 RL debug 报告补上 actor_version / weight_version 证据。
