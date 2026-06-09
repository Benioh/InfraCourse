# 源码带读：L33 DPO Loss

这份带读按“patch 公式 -> pytest 合同 -> synthetic smoke -> RL logprob 对照”的顺序走。读源码时先抓住两个状态：completion logprob 和 reward margin。

## 0. 源码地图

```text
labs/l30_dpo_loss/patch/starter/dpo.py
labs/l30_dpo_loss/patch/reference/dpo.py
labs/l30_dpo_loss/patch/tests/test_patch.py
labs/l30_dpo_loss/scripts/run_dpo_smoke.py

mini_infra/rl/reward.py
github_repo/slime/slime/utils/ppo_utils.py
```

## 1. Patch starter：先看接口和 TODO

文件：[patch/starter/dpo.py](patch/starter/dpo.py)

第 8-23 行是 `compute_logps_for_completions`。读完要能说出 mask、safe labels、`log_softmax`、`gather` 和 sum 的顺序。

第 26-42 行是 `dpo_loss`。函数签名说明输入已经是四组 per-sequence logprob；TODO 把 chosen reward、rejected reward、reward margin 和 loss 分开写出。

## 2. Patch reference：最小正确实现

文件：[patch/reference/dpo.py](patch/reference/dpo.py)

第 9-15 行展示 completion logprob 抽取。重点看 `safe_labels[labels == -100] = 0`：它不改变结果，只是让 `gather` 有合法索引，真正的屏蔽由 mask 完成。

第 25-34 行展示 DPO loss。`chosen_reward` 和 `rejected_reward` 都是 policy 相对 reference 的 logprob 差乘 beta；`reward_margin` 决定 `-F.logsigmoid` 的输入。

## 3. Patch tests：每个断言保护什么

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

第 23-56 行覆盖 logprob 抽取：输出 shape、prompt mask、手写 `log_softmax + gather` 一致性。第 59-76 行覆盖两个基准点：`beta=0` 和 policy 等于 reference 时 loss 都应为 `log(2)`。

第 79-113 行覆盖 DPO 方向和公式：chosen 变好时 loss 下降，rejected 变好时 loss 上升，reward margin 等于手写公式。第 116-125 行覆盖数值稳定性：大 beta 下 loss 必须有限。

## 4. DPO smoke：最小训练循环

文件：[scripts/run_dpo_smoke.py](scripts/run_dpo_smoke.py)

第 32-47 行选择 starter 或 reference。验收时可用 `IMPL=reference` 强制走参考实现。第 50-64 行读取配置、准备 run 目录、写命令快照和 resolved config。

第 86-109 行构造 TinyLM、冻结 reference、生成 chosen/rejected token，并把前半段 label 设为 `-100` 模拟 prompt mask。第 111-127 行跑 50 步训练：分别计算 policy/ref 的 chosen/rejected logprob，再调用 `dpo_loss`。

第 128-147 行写 metrics、artifact 和 report。读完要能回答这个 smoke 能证明什么：它证明最小训练循环和 artifact 闭合，不证明真实数据质量。

## 5. RL logprob 对照

文件：[github_repo/slime/slime/utils/ppo_utils.py](../../github_repo/slime/slime/utils/ppo_utils.py)

第 151-158 行展示真实 RL 框架里 token logprob 的抽取路径。它使用 fused vocab parallel cross entropy，把 logits 和 tokens 转成 token-level logprob。DPO patch 用 `log_softmax + gather` 做 CPU-safe 版本，主合同相同：得到目标 token 的 logprob。

第 124-148 行展示 PPO policy loss 如何使用 ratio 和 advantage。它不是 DPO，但帮助你对比两类目标：PPO 依赖 rollout/advantage，DPO 依赖 preference pair 和 reference log-ratio。

文件：[mini_infra/rl/reward.py](../../mini_infra/rl/reward.py)

第 12-18 行和第 21-33 行展示 scalar reward parser。DPO 本讲不训练 reward model，但这个文件帮助你对比两条路线：reward parser 产生标量奖励，DPO 直接使用偏好对。

## 可以先跳过的内容

- 真实 TRL/Hugging Face 训练器的参数编排。
- 分布式 reference forward、gradient checkpointing 和 optimizer state 管理。
- 多轮 DPO iterative 数据刷新策略。

## 读完后的自检问题

1. `labels == -100` 在 logprob 抽取中起什么作用？
2. policy 等于 reference 时，为什么 DPO loss 是 `log(2)`？
3. smoke 中 reference 是怎样冻结的？
4. DPO 和 PPO 在使用 logprob 的方式上有什么差异？
