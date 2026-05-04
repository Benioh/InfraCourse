# L11.5 · SLiME / verl Rollout Freshness：版本化 RolloutManager

> 本关补上 SLiME / verl 的框架主线：实现带 `weight_version` 的 RolloutManager，
> 然后用 `scripts/run_rl_drill.py` 演练 actor 与多个 rollout server 之间的权重同步
> 节奏，看到 stale rollout 和 weight sync stall 的 trade-off。

## 闭环

```bash
cat labs/l33_rl_rollout_freshness/patch/task.md
$EDITOR labs/l33_rl_rollout_freshness/patch/starter/rollout_manager.py
make patch-test M=l33_rl_rollout_freshness

bash labs/l33_rl_rollout_freshness/scripts/run_rl_drill.sh
```

## Drill 演练

`scripts/run_rl_drill.py` 模拟一段 RL：

1. 初始化 `N_servers` 个 RolloutServer
2. actor 每个 step++，按概率 `update_rate` 推一次新权重到子集 server
3. 每个 step 调用 `manager.generate(prompts, actor_version)` 收 rollout
4. 收集：rollout 成功率、stale fraction、p50/p99 staleness、no-fresh-server 失败次数

acceptance：
- 同步充分时（每 step 都更新所有 server）— 不能有 RuntimeError
- `max_staleness=1` + `update_rate=0.5` 时大多数 step 仍能找到 fresh server
- `update_rate=0` 且 actor 持续推进时必须最终抛 RuntimeError（不能用 stale）

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 默认 drill：4 server，50 step |
| `configs/h200_slime.yaml` | 真实 SLiME train+rollout 启动模板 |

## 调试工单

见 `tickets/INDEX.md`。建议至少做 `slime_stale_rollout_silent` 和 `slime_weight_sync_stall`。

## 进入下一关

通过后进入 [L12 多模态 Capstone](../l35_multimodal_capstone/README.md)。
