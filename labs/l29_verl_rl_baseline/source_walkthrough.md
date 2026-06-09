# 源码带读：L31 Adaptive KL Controller

这份带读按“patch controller -> reward smoke -> SLiME 对照”的顺序走。重点放在四个局部机制：KL 系数如何更新、reward parser 如何自测、smoke 如何落盘，以及真实框架如何使用 KL 和 PPO clip。

## 0. 源码地图

```text
labs/l29_verl_rl_baseline/patch/starter/kl_controller.py
labs/l29_verl_rl_baseline/patch/reference/kl_controller.py
labs/l29_verl_rl_baseline/patch/tests/test_patch.py

labs/l29_verl_rl_baseline/scripts/download_gsm8k.py
labs/l29_verl_rl_baseline/scripts/prepare_gsm8k_prompts.py
labs/l29_verl_rl_baseline/scripts/reward_math.py
labs/l29_verl_rl_baseline/scripts/run_verl_lab.py

github_repo/slime/slime/utils/ppo_utils.py
github_repo/slime/train_async.py
github_repo/slime/slime/utils/arguments.py
```

## 1. Patch controller

文件：[patch/starter/kl_controller.py](patch/starter/kl_controller.py)

先看第 16-22 行。docstring 已经给出公式：先算 proportional error，再 clip 到 `[-0.2, 0.2]`，最后按 `n_steps / horizon` 缩放后乘到 `kl_coef` 上。

再看第 24-41 行。starter 只留下三个 TODO：初始化状态、更新系数、读取系数。写 patch 前先确认 `target_kl` 和 `horizon` 的方向：target 是比较基准，horizon 越大更新越慢。

文件：[patch/reference/kl_controller.py](patch/reference/kl_controller.py)

重点看第 6-18 行。reference 是公式的直接翻译：`self.value` 保存当前系数，`update` 做相对误差、clip 和乘法更新，`get_coef` 只返回当前值。

## 2. Patch tests

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

按顺序读：

- 第 21-24 行：初始系数必须等于 `init_kl_coef`。
- 第 27-31 行：KL 高于目标时系数增大。
- 第 34-38 行：KL 低于目标时系数减小。
- 第 41-45 行：KL 等于目标时系数不变。
- 第 48-59 行：极端 KL 经 clip 后，horizon 为 10 时只增长 2%。

这五个测试只验证控制器数学，不验证 rollout、reward、reference logprob 或 advantage。

## 3. Reward smoke

文件：[scripts/download_gsm8k.py](scripts/download_gsm8k.py)

第 5-17 行写入 toy GSM8K JSONL。它让本地 smoke 不依赖网络下载。

文件：[scripts/prepare_gsm8k_prompts.py](scripts/prepare_gsm8k_prompts.py)

第 5-20 行把 toy 数据转成 prompt/target JSONL。后续 reward parser 用 target 做数值比较。

文件：[scripts/reward_math.py](scripts/reward_math.py)

第 13-38 行负责数字归一化、最终数字提取和 reward 计算。第 59-90 行定义 self-test cases，并输出每个 case 的通过状态和失败原因。先跑 self-test，再讨论 RL 曲线。

## 4. Run lab

文件：[scripts/run_verl_lab.py](scripts/run_verl_lab.py)

第 31-55 行准备 run 目录、写 command/config/prediction、生成 toy 数据、生成 prompts，并运行 reward self-test。第 56-74 行向 `metrics.jsonl` 写入模拟 RL 指标：reward、KL、entropy、response length、rollout/update time 和 samples/sec。第 75-120 行写 log、report 和 run 路径。

读完这一段要能回答：本地 smoke 能证明什么，不能证明什么。它能证明数据、reward parser 和指标落盘闭合；它不能证明真实 PPO 已经稳定。

## 5. SLiME 对照

文件：[github_repo/slime/slime/utils/ppo_utils.py](../../github_repo/slime/slime/utils/ppo_utils.py)

先看第 12-51 行。`compute_approx_kl` 支持 k1、k2、k3 和 low_var_kl，说明 KL estimator 是 controller 前面的观测来源。

再看第 125-148 行。`compute_policy_loss` 用 ratio 和 clip 构造 PPO policy loss。它和 KL penalty 是两层不同约束。

最后分两段看 KL penalty 的落点。第 217-229 行说明 returns 函数接收 `kl_coef`；第 253-258 行把 per-token KL 乘以负系数后并入 token-level reward。第 281-286 行说明 baseline advantage 路径也接收 `kl_coef`；第 302-308 行把 reward 和 KL penalty 组合成 unwhitened advantages。读完后要能说清 controller 输出最终如何影响 actor loss。

文件：[github_repo/slime/train_async.py](../../github_repo/slime/train_async.py)

第 10-29 行创建 rollout manager、actor/critic 并同步初始权重。第 34-53 行展示 async rollout 与 actor/critic train 的交替。第 69-79 行展示周期性权重同步、eval 和清理。

## 自检问题

1. `current_kl == target_kl` 时，controller 为什么不应改变系数？
2. clip 限制的是 error 还是 `kl_coef`？
3. reward self-test 能排除哪类 RL debug 噪声？
4. `compute_approx_kl` 和 `AdaptiveKLController` 在链路中的先后关系是什么？
5. ratio clip 和 KL penalty 分别约束什么？
