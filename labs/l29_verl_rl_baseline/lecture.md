# L31 · Adaptive KL Controller (PPO)

RLHF 训练最容易误读的一条曲线是 reward。reward 上升时，policy 可能真的学会了更好的回答，也可能只是离 reference model 越来越远，开始输出重复片段、格式投机、拒答模板，或者钻 reward parser 的漏洞。L31 讲一个很小的控制器：根据实际 KL 和目标 KL 的偏差，动态调节 KL penalty 系数。

这节课的重点不是跑完整 verl 训练。我们先把 `AdaptiveKLController` 的数学和状态机写清楚，再用 toy GSM8K reward smoke 检查数据、奖励和指标落盘，最后对照 SLiME 的 PPO utilities 看真实框架把 KL、ratio clip、reward 和 rollout loop 放在什么位置。

## 1. 本讲目标

- 解释 policy、reference、reward、KL penalty 和 ratio clip 在 PPO/RLHF 中的分工。
- 手写 InstructGPT App C 风格的 adaptive KL coefficient controller。
- 读懂 `current_kl / target_kl - 1`、`clip(-0.2, 0.2)`、`horizon` 和 `n_steps` 对更新速度的影响。
- 用 patch tests 判断控制器公式是否正确。
- 用 reward self-test、`metrics.jsonl` 和 debug checklist 区分 KL 爆炸、reward parse 错误和 rollout 慢。

## 2. 系统位置

在 RLHF/PPO 链路里，一次训练更新通常包含这些信号：

```text
prompt
  -> policy rollout
  -> policy logprob
  -> reference logprob
  -> reward parser / reward model
  -> KL estimate + reward
  -> advantage / returns
  -> PPO actor loss
```

KL controller 位于 KL estimate 和 actor loss 之间。它不生成 token，不计算 reward，也不做 optimizer step。它只保存一个状态：当前 KL penalty 系数。每次系统观测到 `current_kl` 后，controller 把它和 `target_kl` 比较，并把新的系数交给后续 loss 或 reward shaping 使用。

这个位置决定了排障顺序。KL 曲线异常时，先确认 reference logprob、KL estimator 和 mask 没错，再看 controller 是否按公式调系数。reward 曲线异常时，先确认 reward parser 自测通过，再讨论 PPO 参数。rollout 慢时，先拆 `rollout_time_sec` 和 `update_time_sec`，不要把生成路径问题归到 KL controller 上。

## 3. RLHF 里的几个角色

**Policy model** 是正在训练的模型。它负责生成 response，并在 actor update 时接收梯度。

**Reference model** 是锚点模型，通常来自 SFT checkpoint。它只 forward，给出 reference logprob。KL penalty 衡量当前 policy 相对它漂移了多少。

**Reward** 是训练方向。L31 使用规则 reward 和 toy GSM8K 数据，真实系统里可能是 reward model、规则、judge 或混合打分。

**Ratio clip** 是 PPO 局部约束。它限制新旧 policy 的 token-level importance ratio，避免一次 update 步子太大。

**KL penalty** 是相对 reference 的约束。它限制 policy 在多轮更新后偏离参考模型的程度。

这几个角色互相补位。reward 告诉 policy 哪些输出更好；ratio clip 限制单次更新；KL penalty 把 policy 拉在 reference 附近；reference model 提供漂移基准。只看其中一项，训练健康度判断会很脆。

## 4. Adaptive KL Controller 的输入、状态和输出

本讲 patch 的接口很小：

```python
ctrl = AdaptiveKLController(init_kl_coef=0.2, target_kl=0.05, horizon=10000)
ctrl.update(current_kl=0.08, n_steps=1)
coef = ctrl.get_coef()
```

输入有三个配置和一个观测值：

- `init_kl_coef`：初始 KL penalty 系数。
- `target_kl`：希望 KL 靠近的目标值。
- `horizon`：控制器反应时间尺度，越大反应越慢。
- `current_kl`：本轮 rollout 或训练统计得到的实际 KL。

内部状态是 `self.value`，也就是当前系数。输出是 `get_coef()` 返回的浮点数。这个数最终会参与 actor loss 或 reward shaping。L31 的最小实现不负责分布式同步；真实训练里如果多 worker 各自统计 KL，框架还要决定聚合方式和更新频率。

边界也要说清楚。这个 controller 假设 `target_kl` 和 `horizon` 为正；patch tests 聚焦公式合同，没有覆盖配置校验。生产代码通常会在配置解析层拦住非法值。

## 5. 公式怎么工作

L31 使用的公式可以拆成四步：

```text
raw_error = current_kl / target_kl - 1
proportional_error = clip(raw_error, -0.2, 0.2)
scale = 1 + proportional_error * n_steps / horizon
kl_coef = kl_coef * scale
```

第一步用相对误差。`current_kl` 是 `target_kl` 两倍时，`raw_error = 1`；`current_kl` 是目标值的 0.2 倍时，`raw_error = -0.8`。相对误差的好处是 target 改变量级后，controller 仍按偏离比例反应。

第二步做限幅。极端 KL 可能来自 reward outlier、mask bug 或 reference 错配。把误差限制在 `[-0.2, 0.2]` 可以避免一个异常 batch 让 `kl_coef` 剧烈跳变。这个 clip 限制的是误差，不是 `kl_coef` 的绝对值。

第三步用 `n_steps / horizon` 控制速度。`horizon` 越大，每次更新越小；`n_steps` 越大，一次统计覆盖的训练步越多，更新幅度也越大。patch test 里 `horizon=10`、极端正误差被 clip 到 `0.2`，所以单步 scale 是 `1 + 0.2 * 1 / 10 = 1.02`。

第四步用乘法更新。KL coefficient 是正的尺度参数，用乘法可以保持“按比例调节”的语义。只要 scale 为正，系数就不会变成负数。

## 6. 实现路径

starter 文件已经把任务拆成三个方法：

- `__init__`：保存 `self.value`、`self.target_kl`、`self.horizon`。
- `update`：读取 `current_kl`，计算误差、clip 和乘法更新。
- `get_coef`：返回当前 `self.value`。

实现时容易犯的错误有三类。第一，把 `self.value` 每次重置成初值，导致 controller 没有记忆。第二，把 `horizon / n_steps` 写进公式，导致 horizon 越大变化越快，方向反了。第三，用加法更新系数，让不同初始值下的控制行为不再按比例缩放。

参考实现刻意很短，因为这一讲要训练的是状态语义，不是框架 API 记忆。学生应该能从测试反推出公式，也能从公式解释每条测试为什么存在。

## 7. Patch Tests 验收什么

测试文件覆盖五个合同：

1. 初始系数等于 `init_kl_coef`。
2. `current_kl > target_kl` 时，系数增大。
3. `current_kl < target_kl` 时，系数减小。
4. `current_kl == target_kl` 时，系数不变。
5. 极端 KL 被 clip，`horizon=10` 时从 `1.0` 只增长到 `1.02`。

这些测试能证明 controller 的局部数学正确。它们不能证明真实 PPO 稳定，也不能证明 reward parser、reference checkpoint、KL estimator、advantage normalization 或 rollout engine 没问题。patch-test 通过后，下一步要跑本地 smoke，确认数据、reward 和 metrics 证据链能闭合。

## 8. Reward Parser 和本地 Smoke

L31 的本地 smoke 不启动大模型训练。它先把最小证据链跑通：

```text
download_gsm8k.py
  -> data/gsm8k_toy/train.jsonl
prepare_gsm8k_prompts.py
  -> data/gsm8k_toy/prompts.jsonl
reward_math.py --self-test
  -> artifacts/reward_self_test.json
run_verl_lab.py
  -> metrics.jsonl + rl.log + report.md
```

`reward_math.py` 的工作很具体：从 prediction 和 target 中提取最终数字，做数值比较，并在失败时给出 `missing_number` 或 `answer_mismatch`。这一步先跑，是为了避免把答案解析错误误判成 PPO 或 KL 问题。

`run_verl_lab.py` 写入五步模拟 RL 指标：`reward_mean`、`reward_std`、`kl_mean`、`entropy`、`response_len_mean`、`rollout_time_sec`、`update_time_sec` 和 `samples_per_sec`。这些指标的作用是建立 debug 证据格式。真实 GPU 训练还需要接入 verl/SLiME 的实际 rollout、reference logprob、actor update 和分布式日志。

## 9. 对照 SLiME 源码

本仓库没有本地 `github_repo/verl/`，所以 L31 用 SLiME 作为框架对照。它能帮助学生把小 controller 放回完整 RL infra。

`github_repo/slime/slime/utils/ppo_utils.py` 里，`compute_approx_kl` 接收当前 logprob 和 base/reference logprob，支持 `k1`、`k2`、`k3` 和 `low_var_kl` 等 KL estimator。这个函数在 controller 之前，负责提供 KL 观测。

同一文件的 `compute_policy_loss` 使用 ratio 和 clip 构造 PPO policy loss。这是单次 update 的约束，和 KL penalty 分层工作。后面的 return / advantage 代码会把 `kl_coef` 乘到 per-token KL 上，说明 controller 输出会进入训练信号。

`github_repo/slime/train_async.py` 展示 rollout manager、actor、critic 和权重同步的主循环。真实系统里，KL controller 只是其中一个小组件。rollout 并发、stop token、max new tokens、weight sync、reference 版本和指标聚合都可能影响最终曲线。

## 10. Debug 路线

遇到 KL 爆炸，先检查 step 0 KL。如果 policy 和 reference 初始相同，step 0 KL 应接近 0；如果一开始就很高，优先查 reference checkpoint、tokenizer、mask 和 logprob 对齐。随后看 KL 是否随 update 指数上涨，再检查 `kl_coef`、`target_kl`、learning rate、PPO epochs、ratio clip、advantage normalization 和 reward outlier。

遇到 reward 异常，先跑 reward parser self-test。自测失败时，继续调 PPO 参数没有意义，因为训练信号入口已经错了。自测通过后，再看 reward distribution、outlier、response length 和样本内容。

遇到 rollout 慢，先拆时间。`rollout_time_sec` 高，优先看 engine、并发、batch、`max_new_tokens`、stop token、prefix cache、tokenizer CPU 和权重同步；`update_time_sec` 高，再看 actor/critic 训练、通信和显存。L31 的 smoke 只给出记录格式，后续课程会继续拆 rollout/training logprob 一致性和 rollout 并发控制。

## 11. Lab 验收边界

patch 命令：

```bash
IMPL=reference make patch-test M=l29_verl_rl_baseline
```

学生实现时默认跑：

```bash
make patch-test M=l29_verl_rl_baseline
```

smoke 命令：

```bash
python labs/l29_verl_rl_baseline/scripts/run_verl_lab.py --run-id l31_smoke
```

本讲交付的能力是：能解释 KL controller 的公式、实现最小 patch、跑通 reward smoke，并能把 KL、reward、entropy、长度和时间指标放到同一张排查表里。完成 L31 后，再进入 Train-Infer Mismatch 修正算子，继续处理 rollout 与 training 之间的 logprob 对齐问题。
