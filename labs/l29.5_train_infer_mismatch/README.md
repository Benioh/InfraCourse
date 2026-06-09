# L32 · Train-Infer Mismatch：K3、TIS/MIS 与序列级修正

<!-- LECTURE_FIRST_START -->

本讲处理 RL 训练里很容易被吞吐优化掩盖的问题：rollout engine 负责采样，training engine 负责算损失；即使用同一份权重，对同一段 token 序列算出的 log probability 也可能出现差异。这个差异进入 PPO/GRPO 后，会把同策略更新推成带偏差的异策略更新，常见现象是 K3 KL 上升、ratio 分布拉长、grad norm 异常和 reward 下跌。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L32 在 RLHF / rollout systems 主线里的位置。
2. 读 [lecture.md](lecture.md)：从 mismatch 来源、K3 KL、TIS/MIS、Geometric IS、Veto 和 Batch Normalize 讲到生产排查。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch reference、测试合同和 SLiME `mis.py` 主路径读源码。
4. 跑 notebook：[n19_train_infer_mismatch.ipynb](../../notebooks/n19_train_infer_mismatch.ipynb)。
5. 做 quiz：确认公式方向、shape、mask 边界和排查顺序。
6. 做 patch：实现六个 PyTorch 算子并通过测试。
7. 填写 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md)，沉淀一次 mismatch 排查复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | RLHF and rollout systems |
| 核心风险 | rollout logprob 与 training logprob 不一致，导致 ratio 长尾、loss 权重偏移和训练不稳定 |
| 关键机制 | K3 KL 监控、Token-level TIS、Token-level MIS、Geometric Sequence IS、Veto、Batch Normalize |
| 源码落点 | patch starter/reference/tests，SLiME train-infer mismatch helper |
| lab 检验 | 六个算子的公式、shape、padding、极值和归一化边界 |

## Patch 闭环

```bash
cat labs/l29.5_train_infer_mismatch/patch/task.md
$EDITOR labs/l29.5_train_infer_mismatch/patch/starter/mismatch.py
make patch-test M=l29.5_train_infer_mismatch
```

本讲没有独立 smoke target。系统层现象由 notebook 和讲义中的源码对照承担；patch 只验收算子合同。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 K3 KL 上升、ratio 长尾、veto 命中和 batch normalize 漂移 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch 与 SLiME `mis.py` 的源码主路径 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录一次 mismatch 复盘的输入、指标、源码判断和下一步动作 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L32 后进入 DPO Loss。下一讲会把偏好样本中的 chosen/rejected logprob 变成可训练的 preference objective。
