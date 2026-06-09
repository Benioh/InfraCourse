# L38 · SLiME Rollout Freshness：Versioned RolloutManager

<!-- LECTURE_FIRST_START -->

本讲把 L36/L37 的权重同步结果接回 rollout 生成路径：每个 rollout server 都要暴露自己的 `weight_version`，manager 只允许足够新的 server 继续产训练样本。

## 学习路线

建议按下面顺序走，先理解系统问题，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L38 在 RL rollout 主线里的位置。
2. 读 [lecture.md](lecture.md)：从 stale rollout 问题、freshness 合同、SLiME 源码到排障指标。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 MiniInfra、patch、SLiME rollout manager、SGLang engine 和 drill 读源码。
4. 跑 notebook：[n10_rl_kl_reward.ipynb](../../notebooks/n10_rl_kl_reward.ipynb)
5. 做 quiz：确认版本字段、staleness、局部更新和排障边界。
6. 做 patch：实现最小 versioned manager 行为合同。
7. 跑 drill：观察 update 频率、subset size 和 max_staleness 如何改变 failure 与 staleness 分布。
8. 使用 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) 记录一次复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | 第 4 章：RL 与对齐 / RL rollout systems |
| 解决什么问题 | actor 已经训练到新版本时，rollout engine 还能不能继续用旧权重生成样本 |
| 连接哪些源码 | `mini_infra/slime/ray/rollout.py`, `github_repo/slime/train.py`, `github_repo/slime/slime/ray/rollout.py`, `github_repo/slime/slime/utils/types.py`, `github_repo/slime/slime/backends/sglang_utils/sglang_engine.py` |
| lab 检验什么 | `RolloutManager.generate` 跳过 stale server，`update_weights` 支持全量/局部更新，`freshness` 返回 actor 与 server 的版本差 |

## 你会学到什么

- 解释 `actor_version - weight_version` 为什么是 rollout freshness 的最小证据。
- 判断 stale rollout 对 PPO/GRPO 的 ratio、KL、reward 和 advantage 解释有什么影响。
- 读懂 MiniInfra 和 SLiME 中 rollout manager、sample meta_info、SGLang weight version 的源码落点。
- 完成 versioned `RolloutManager` patch，并说明测试覆盖的行为边界。
- 用 drill 产物复盘 update 频率、局部更新范围、max_staleness 和失败率之间的关系。

## Patch 闭环

```bash
cat labs/l33_rl_rollout_freshness/patch/task.md
$EDITOR labs/l33_rl_rollout_freshness/patch/starter/rollout_manager.py
make patch-test M=l33_rl_rollout_freshness
```

drill：

```bash
python labs/l33_rl_rollout_freshness/scripts/run_rl_drill.py --run-id l38_local
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l33_rl_rollout_freshness
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 stale rollout、局部更新和版本字段缺失 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习源码主路径和关键结论 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录 drill 或真实 SLiME 运行的版本、指标和结论 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L38 后进入 [L39 · GRPO](../l34_grpo/README.md)。
