# L38 Source Reading Card：Rollout Freshness

## 主路径

1. `labs/l33_rl_rollout_freshness/patch/starter/rollout_manager.py`：学生要补的 versioned manager 合同。
2. `labs/l33_rl_rollout_freshness/patch/reference/rollout_manager.py`：freshness、round-robin、局部更新和诊断视图。
3. `labs/l33_rl_rollout_freshness/patch/tests/test_patch.py`：五个行为测试。
4. `mini_infra/slime/ray/rollout.py`：最小 rollout server/manager 版本链路。
5. `mini_infra/slime/ray/train_actor.py`：actor train 后递增版本，并写回 rollout manager。
6. `mini_infra/slime/train.py`：generate、train、update_weights 的最小循环。
7. `github_repo/slime/train.py`：真实 SLiME 训练循环。
8. `github_repo/slime/slime/ray/rollout.py`：生产 rollout manager、updatable engines 和 rollout data。
9. `github_repo/slime/slime/utils/types.py`：Sample 如何收集 `weight_version`。
10. `github_repo/slime/slime/backends/sglang_utils/sglang_engine.py`：SGLang engine 如何接收和查询 weight version。
11. `labs/l33_rl_rollout_freshness/scripts/run_rl_drill.py`：drill 如何落盘 metrics 和 acceptance。

## 阅读方法

1. 先看 `meta_info` 里有哪些版本字段。
2. 再看 manager 怎么计算 `actor_version - weight_version`。
3. 接着看 `update_weights` 对全部 server 和子集 server 的不同处理。
4. 继续看真实 SLiME 中哪些 server 可以接收 actor 权重。
5. 最后看 drill 的 metrics 和 report 如何支撑结论。

## 自检

- 我能否解释 stale rollout 为什么会影响 reward/KL 解释？
- 我能否指出 patch reference 和 MiniInfra 的同构关系？
- 我能否说明真实 SLiME 比 patch 多了哪些生产分支？
- 我能否从一个 `server_id` 的长期 staleness 判断局部更新问题？
