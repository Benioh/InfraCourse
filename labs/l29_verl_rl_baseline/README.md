# L31 · verl / SLiME RL Baseline：Adaptive KL Controller

<!-- LECTURE_FIRST_START -->

本讲进入 RL 与对齐主线。PPO/RLHF 训练不能只追 reward；policy 偏离 reference 太远时，reward 可能上升但输出质量下降、重复、拒答或钻 reward parser 漏洞。L31 用一个 CPU-safe 的 Adaptive KL Controller patch 讲清 `current_kl`、`target_kl`、`kl_coef`、`horizon` 和 `clip` 的控制关系，再用 toy GSM8K reward smoke 建立本地证据链。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L31 在 RLHF / rollout systems 主线中的位置。
2. 读 [lecture.md](lecture.md)：从 KL penalty、ratio clipping、reference model、controller 公式讲到 reward smoke 和 SLiME 对照。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、reward parser、run lab、SLiME PPO utils 和 async train 主路径阅读。
4. 跑 notebook：[n10_rl_kl_reward.ipynb](../../notebooks/n10_rl_kl_reward.ipynb)。
5. 做 quiz：确认 KL controller、PPO 约束、reward parser 和 debug 边界。
6. 做 patch：实现 `AdaptiveKLController` 并通过测试。
7. 跑 smoke：验证 toy GSM8K、reward self-test 和 RL metrics artifact。
8. 填写 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md)，沉淀一次 RL smoke 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | RLHF and rollout systems |
| 核心风险 | KL 爆炸、KL 长期过低、reward parser 错误、reward hacking、rollout/update 证据不足 |
| 关键机制 | Adaptive KL coefficient、target KL、horizon、clip、reward self-test、rollout/update metrics |
| 源码落点 | patch controller、reward_math、run_verl_lab、SLiME PPO utils、SLiME async train loop |
| lab 检验 | 初始系数、KL 过高增罚、KL 过低减罚、目标处不变、极端 KL clip |

## Patch 闭环

```bash
cat labs/l29_verl_rl_baseline/patch/task.md
$EDITOR labs/l29_verl_rl_baseline/patch/starter/kl_controller.py
make patch-test M=l29_verl_rl_baseline
```

Smoke：

```bash
python labs/l29_verl_rl_baseline/scripts/run_verl_lab.py --run-id l31_smoke
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 KL 爆炸、reward parse 错误、rollout 慢和指标缺失 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 controller、reward smoke、SLiME PPO utils 和 train loop |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录一次 RL rollout 或 smoke 的配置、指标、判断和下一步 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L31 后进入 Train-Infer Mismatch 修正算子。下一讲会继续处理 rollout engine 与 training engine 的 logprob 不一致问题。
