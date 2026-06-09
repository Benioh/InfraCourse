# L39 · GRPO / RLOO：Critic-free Advantage 与 Clipped KL Loss

<!-- LECTURE_FIRST_START -->

本讲把 RL 主线从 rollout freshness 推到算法更新本身：不用 value critic 时，怎样用同一 prompt 下的一组 response reward 构造 advantage，并把它放进带 ratio clip 和 reference KL 的 policy loss。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L39 在 RL 与对齐主线中的位置。
2. 读 [lecture.md](lecture.md)：从 critic-free 训练问题、GRPO/RLOO advantage、loss 公式、mask 和 KL 边界讲到 smoke。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、测试、reward parser、SLiME PPO utils 和 smoke 脚本读源码。
4. 做 quiz：确认公式、边界、mask、clip 和 KL 的语义。
5. 做 patch：实现 `grpo_advantage`、`rloo_advantage` 和 `grpo_loss`。
6. 跑 smoke：观察合成 reward + log-prob 下的 loss、KL 和 ratio 指标。
7. 使用 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) 记录一次复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | 第 4 章：RL 与对齐 / policy optimization |
| 解决什么问题 | 不训练 value critic 时，如何从同组 responses 的 reward 得到 advantage，并约束 policy 更新幅度 |
| 连接哪些源码 | `labs/l34_grpo/patch/reference/grpo.py`, `labs/l34_grpo/patch/tests/test_patch.py`, `mini_infra/rl/reward.py`, `github_repo/slime/slime/utils/ppo_utils.py`, `labs/l34_grpo/scripts/run_grpo_smoke.py` |
| lab 检验什么 | 组内 advantage、RLOO baseline、ratio clip、k3 KL、completion mask 和指标返回 |

## 你会学到什么

- 解释 GRPO 为什么可以不用 value model，并说清它的边界。
- 手写 GRPO 和 RLOO advantage，处理 G=1、常数 reward 和 std=0。
- 实现带 mask、ratio clip 和 reference KL 的最小 `grpo_loss`。
- 读懂 SLiME PPO utils 中 KL estimator、OPSM 和 policy loss 的对应关系。
- 用 smoke 产物判断 loss、KL、ratio_mean 和 clipped fraction 是否支持结论。

## Patch 闭环

```bash
cat labs/l34_grpo/patch/task.md
$EDITOR labs/l34_grpo/patch/starter/grpo.py
make patch-test M=l34_grpo
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l34_grpo
IMPL=reference python labs/l34_grpo/scripts/run_grpo_smoke.py --run-id l39_local
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 NaN advantage、mask 归一化、KL 爆炸和 ratio clip 异常 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 GRPO/RLOO 源码主路径 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录 reward、advantage、loss、KL、ratio 和 smoke 产物 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L39 后进入 [L40 · GAE Chunked Parallel](../l34.5_gae_chunked_parallel/README.md)。
