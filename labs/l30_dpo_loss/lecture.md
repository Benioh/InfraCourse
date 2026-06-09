# L33：DPO Loss

DPO 解决的是 SFT 之后的偏好优化问题。给定同一个 prompt 下两条 response，标注告诉我们哪条更好，哪条更差。传统 RLHF 通常先训练 reward model，再用 PPO 根据 reward 更新 policy。DPO 走另一条更短的路径：直接把 chosen/rejected pair 转成一个 pairwise loss，让 policy 相对 reference 更偏向 chosen。

本讲只实现最小 DPO 数学合同。你要写两个函数：`compute_logps_for_completions` 负责从 logits 和 labels 中抽取 completion token 的 log-prob 之和；`dpo_loss` 负责把 policy/ref 的 chosen/rejected logprob 转成 reward margin 和稳定 loss。学完后，你应该能判断一个 DPO 训练问题是公式方向错、mask 错、reference 错，还是数值稳定性错。

## 1. 本讲目标

- 解释 DPO 如何从偏好对构造 policy 优化目标。
- 区分 prompt token 和 completion token 在 logprob 统计中的角色。
- 写出 chosen/rejected 相对 reference 的 implicit reward。
- 使用 `F.logsigmoid` 实现数值稳定的 pairwise loss。
- 读懂 patch、pytest、synthetic smoke 和 RL logprob 对照源码。

## 2. 偏好样本和系统位置

一条 DPO 样本包含三部分：prompt、chosen response、rejected response。prompt 是条件，chosen/rejected 是要比较的 completion。模型要学的是：在同一个 prompt 下，chosen 的相对偏好应高于 rejected。

这类训练通常接在 SFT 后面。SFT 让模型学会基本指令格式，DPO 用偏好对调整回答风格、事实性、安全性或任务偏好。reference policy 通常是 SFT checkpoint 的冻结副本。它的作用是提供锚点：policy 可以偏向 chosen，但不应无约束地远离 SFT 分布。

本讲 patch 不处理 tokenizer、padding side、分布式训练、reference forward 性能优化或真实数据加载。它把核心机制缩到两个函数，先保证公式和 mask 不错。

## 3. Completion-only logprob

`compute_logps_for_completions(logits, labels)` 的输入是 `logits: [B, T, V]` 和 `labels: [B, T]`。labels 里 prompt 或 padding 位置用 `-100` 表示忽略。输出是 `[B]`，每条样本一个 completion log-prob 总和。

实现分四步。第一，用 `mask = labels != -100` 找出有效 completion token。第二，把 `-100` 临时替换成合法 token id，例如 0，否则 `gather` 会因负索引失败。第三，对 vocab 维度做 `F.log_softmax(logits, dim=-1)`，再用 `gather` 取出 label token 的 logprob。第四，把无效位置乘 0，并沿时间维求和。

这个函数的边界很重要。prompt token 是条件，不能被加到 `log π(y|x)` 里。若 prompt 混入，chosen/rejected 的比较会被共同前缀污染；在多轮数据中，tool response 或 padding 也可能混入训练目标。pytest 用 `labels == -100` 的 mask case 捕捉这类错误。

## 4. DPO loss 的状态链

DPO 的核心是比较 chosen 和 rejected 相对 reference 的变化，而非单独奖励 chosen。patch 接收四个 `[B]` 张量：

```text
policy_logp_chosen
policy_logp_rejected
ref_logp_chosen
ref_logp_rejected
```

先构造 implicit reward：

```text
chosen_reward = beta * (policy_logp_chosen - ref_logp_chosen)
rejected_reward = beta * (policy_logp_rejected - ref_logp_rejected)
reward_margin = chosen_reward - rejected_reward
```

如果 policy 相对 reference 更偏向 chosen，`reward_margin` 为正，loss 应下降。如果 policy 更偏向 rejected，margin 为负，loss 应上升。最终 loss 使用：

```python
loss = -F.logsigmoid(reward_margin).mean()
```

`F.logsigmoid` 是数值边界。大负 margin 下，`torch.sigmoid(margin)` 会接近 0，再取 log 可能得到 `-inf`；`F.logsigmoid` 使用稳定路径，pytest 用 `beta=20` 检查大 margin 下仍为有限值。

## 5. beta 和 reference 的工程含义

在 KL 正则化视角里，reference 是 policy 的锚点，beta 反映偏好目标与 reference anchor 之间的权衡。beta 进入 patch 时表现为 reward margin 的缩放因子。beta 太小，margin 变化弱，训练信号可能不足；beta 太大，margin 很快饱和，错误样本或方向错误会产生更强影响。

reference 的边界同样明确。它参与 logprob 计算，但真实训练中不更新参数。smoke 脚本会把 reference 初始化成 policy 的副本，并将 `requires_grad_(False)`。patch 函数本身只接收 logprob，不知道这些 logprob 是否来自冻结 reference；生产排查必须额外检查 reference 是否进入 optimizer。

## 6. 测试和 smoke 能证明什么

patch-test 覆盖 9 个行为合同。logprob 部分检查输出 shape、prompt mask 和手写 `log_softmax + gather` 一致性。DPO 部分检查 `beta=0` 时 loss 为 `log(2)`、policy 等于 reference 时 loss 为 `log(2)`、chosen 变好时 loss 下降、rejected 变好时 loss 上升、reward margin 公式正确，以及大 beta 下不会出现 inf/NaN。

synthetic smoke 使用一个 TinyLM、合成 chosen/rejected token 和冻结 reference 跑 50 步。它会写 `metrics.jsonl`、`artifacts/dpo_smoke.json` 和 `report.md`。这个 smoke 能证明 patch 可接入一个最小训练循环、metrics 能落盘、loss drop 达到配置阈值；它不能证明真实偏好数据质量、tokenizer mask、分布式 reference forward 或模型输出质量。

## 7. 生产排查顺序

排查 DPO 训练时，先看输入语义：chosen/rejected 是否被 swap，prompt 与 completion mask 是否正确，padding/tool token 是否进入 labels。再看 logprob：policy/ref 四组 logprob shape 是否一致，completion-only 求和是否符合预期，reference 是否冻结。

然后看 loss 证据：reward margin 的均值和分位数是否朝正方向移动，loss 是否下降，chosen_reward 是否高于 rejected_reward。若 loss 下降但输出变差，要回到数据质量和 reference anchor；若 loss 为 inf/NaN，先查 `logsigmoid`、beta 和 logprob 极值。

最后记录 artifact。一次可复盘的 DPO 实验至少应包含命令、配置、数据样本、mask 规则、beta、head/tail loss、reward margin、reference 冻结证据和失败样本。缺少这些证据时，无法判断问题来自公式、数据还是训练系统。

## Lab 验收边界

本讲 patch 命令：`make patch-test M=l30_dpo_loss`。

patch 验收的是 completion logprob 和 DPO pairwise loss 的最小合同。它不覆盖 tokenizer、真实偏好数据、reference forward 性能、分布式训练、checkpoint 或多轮 DPO 迭代。

课后使用 `outputs/rl_rollout_template.md` 记录一次完整 DPO 复盘。

---

## 补充：DPO 理论基础与实践陷阱

### Bradley-Terry 模型

DPO 的理论基础是 Bradley-Terry 偏好模型。给定两个 response y_w (chosen) 和 y_l (rejected)：

```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

其中 σ 是 sigmoid，r 是 reward function。DPO 的关键洞察是：最优 policy 和 reward 之间有解析关系：

```
r(x, y) = β · log(π(y|x) / π_ref(y|x)) + constant
```

把这个代入 Bradley-Terry 就得到 DPO loss：

```
L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

这就是 patch 中 `reward_margin` 的来源。

### DPO vs PPO/RLHF 对比

| 方面 | DPO | PPO/RLHF |
|---|---|---|
| 需要 reward model | 不需要（直接从偏好对学） | 需要先训练 RM |
| 训练复杂度 | 类似 SFT（一个 forward/backward） | 需要 4 个模型（actor/critic/ref/rm） |
| 数据需求 | 偏好对 (prompt, chosen, rejected) | Prompt + rollout + reward |
| Reference model | 必须（计算 log-ratio） | 用于 KL 约束 |
| 在线采样 | 不需要（离线数据） | 需要（在线 rollout） |
| 探索能力 | 弱（受限于离线数据分布） | 强（在线采样新 response） |

### 常见 DPO 失败模式

1. **Chosen/Rejected 标签反了**：loss 会上升而不是下降。检查方法：初始时 reward_margin 应接近 0（policy ≈ reference），训练后应为正。

2. **Reference 没有冻结**：如果 reference 也在更新，reward_margin 始终接近 0，训练没有方向。

3. **Prompt 进入 logprob**：如果 prompt token 没被 mask，chosen 和 rejected 的 logprob 差异被共同前缀稀释。

4. **Beta 太大**：reward_margin 快速饱和到 ±∞，logsigmoid 的梯度消失。表现：loss 很快降到接近 0 但输出质量没提升。

5. **Beta 太小**：信号太弱，训练进展极慢。

6. **数据中 chosen 和 rejected 质量差距太小**：模型难以区分，loss 在 log(2) 附近震荡。

### 实践建议

- beta 常用值：0.1-0.5（具体取决于 logprob 的量级）
- 初始 loss 应该接近 log(2) ≈ 0.693（因为 policy = reference 时 margin = 0）
- 监控 chosen_reward 和 rejected_reward 分别的趋势（不只看差值）
- 如果 chosen_reward 下降但 margin 上升 → reward hacking（policy 在两个都变差，但 rejected 变得更差）
