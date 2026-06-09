# L33 · DPO Loss：从偏好对到稳定的 PyTorch 实现

<!-- LECTURE_FIRST_START -->

本讲处理 SFT 之后的偏好优化问题：给定同一个 prompt 下的 chosen/rejected response，怎样不用训练显式 reward model，也能让 policy 更偏向 chosen，同时受 reference policy 约束。L33 用一个 CPU-safe patch 讲清 completion log-prob 抽取、reference log-ratio、reward margin 和 `logsigmoid` 数值稳定性，再用合成偏好数据跑一个 DPO smoke。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L33 在 SFT、偏好优化与 RL 主线中的位置。
2. 读 [lecture.md](lecture.md)：从偏好样本、Bradley-Terry 目标、reference anchor 讲到实现边界。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、pytest、DPO smoke 和 RL logprob 对照阅读。
4. 做 quiz：确认公式方向、prompt mask、beta、reference 和数值稳定性。
5. 做 patch：实现 `compute_logps_for_completions` 和 `dpo_loss`。
6. 跑 smoke：在合成数据上观察 loss、reward margin 和 artifact。
7. 填写 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md)，沉淀一次偏好优化复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | SFT、偏好优化与 RL |
| 核心风险 | prompt token 混入 logprob、chosen/rejected 方向写反、reference 梯度泄漏、`log(sigmoid)` 下溢 |
| 关键机制 | completion-only logprob、reference log-ratio、implicit reward、reward margin、`F.logsigmoid` |
| 源码落点 | patch starter/reference/tests，DPO synthetic smoke，SLiME PPO logprob 对照 |
| lab 检验 | 9 个 CPU 测试覆盖 shape、mask、公式、单调性和大 beta 数值稳定 |

## Patch 闭环

```bash
cat labs/l30_dpo_loss/patch/task.md
$EDITOR labs/l30_dpo_loss/patch/starter/dpo.py
make patch-test M=l30_dpo_loss
```

Smoke：

```bash
bash labs/l30_dpo_loss/scripts/run_dpo_smoke.sh l33_smoke
```

参考实现验收可使用：

```bash
IMPL=reference make patch-test M=l30_dpo_loss
IMPL=reference bash labs/l30_dpo_loss/scripts/run_dpo_smoke.sh l33_reference
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 DPO logprob、reference、beta、reward margin 和 smoke artifact |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、pytest、smoke 和 RL logprob 对照源码 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录一次 DPO 运行的配置、指标、源码判断和下一步 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L33 后进入 CUDA Graph Cache + Memory Savor。下一讲会处理 co-locate RL 场景下的显存占用和 graph replay 边界。
