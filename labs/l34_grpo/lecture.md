# L39：GRPO / RLOO Advantage and Loss

L39 进入 actor 更新公式。L38 解决 rollout 样本是否来自足够新的 policy；L39 解决这些样本拿到 reward 以后，怎样在没有 value critic 的情况下构造 advantage，并把它放进受约束的 policy loss。

在 PPO 中，advantage 常来自 reward-to-go 或 GAE，需要 value model 估计 baseline。GRPO/RLOO 走另一条路：对同一 prompt 采样多条 response，用这一组 response 的 reward 相对关系构造 baseline。这样可以省掉 value critic，但会把训练质量强依赖到 group 采样、reward parser、mask 和 KL 约束上。

## 1. 本讲目标

- 解释 GRPO/RLOO 为什么属于 critic-free advantage 估计。
- 写出 GRPO 组内归一化和 RLOO leave-one-out baseline。
- 实现带 completion mask、ratio clip 和 reference KL 的最小 loss。
- 读懂 SLiME PPO utils 中 KL estimator、OPSM 和 policy loss 的对应关系。
- 用 smoke 指标判断 loss 下降是否有足够证据支撑。

## 2. 问题背景

RLHF 或数学推理 RL 常会对同一 prompt 采样多条 response，然后用 rule-based reward 或 reward model 打分。对于一个 prompt，如果 response A 得分高、response B 得分低，我们可以不训练 value critic，直接用组内相对 reward 告诉 actor：更像高分 response，少走低分 response。

这个方法的代价也很明确。它要求每个 prompt 有足够的 group size；reward 需要在组内有区分度；mask 必须只覆盖 completion token；更新幅度要被 ratio clip 和 reference KL 控住。否则，advantage 可能退化成噪声，或者 policy 被少数高 reward 样本拉得过远。

## 3. GRPO Advantage

GRPO 的 patch 输入是一个 prompt 下 G 条 response 的 reward：

```text
rewards: [G]
```

输出也是 `[G]`，每个 response 一个 advantage：

```text
A_i = (r_i - mean(r)) / (std(r) + eps)
```

直观上，组内平均 reward 是 baseline，高于组均值的 response 得到正 advantage，低于组均值的 response 得到负 advantage。标准差归一化让不同 prompt 的 reward 尺度更接近。

边界要处理清楚。G=1 时没有组内比较，advantage 返回 0。所有 reward 相等时 std 为 0，也返回 0 或等价的 `rewards - mean`，不能产生 NaN。这个边界在真实训练里很常见，例如 rule-based reward 对一组 response 全给 0。

## 4. RLOO Advantage

RLOO 使用 leave-one-out baseline：

```text
A_i = r_i - mean(r_j for j != i)
```

它和 GRPO 的区别在 baseline。GRPO 对所有 response 使用同一个组均值；RLOO 对第 i 条 response 的 baseline 排除 `r_i` 本身。这样第 i 条 response 不会参与自己的 baseline。G=1 时同样没有可用 baseline，返回 0。

在 patch 中，RLOO 只需要处理一维 rewards。真实系统里要先按 prompt 分组，再分别计算每组 advantage，不能把不同 prompt 的 reward 混在一个 group 里。

## 5. GRPO Loss

`grpo_loss` 的输入包括当前 policy、behavior policy、reference policy 的 token log-probs：

```text
log_probs:      [G, T]
log_probs_old:  [G, T]
log_probs_ref:  [G, T]
advantages:     [G]
mask:           [G, T]
```

先把 `[G]` 的 advantage 扩展到 `[G, T]`。然后计算 ratio：

```text
ratio = exp(log_probs - log_probs_old)
```

policy 部分沿用 PPO 的 clipped surrogate：

```text
unclipped = ratio * A
clipped = clip(ratio, 1-eps, 1+eps) * A
policy_loss = -mean_masked(min(unclipped, clipped))
```

mask 很关键。prompt token、padding token 或不应训练的位置都不能参与 loss。patch 要用 `mask.sum().clamp(min=1)` 做归一化，避免全 0 mask 时除 0。

KL 部分使用 reference policy：

```text
diff = log_probs_ref - log_probs
kl = exp(diff) - diff - 1
loss = policy_loss + kl_beta * kl_loss
```

`ratio_mean` 和 `ratio_clipped_frac` 是排障指标。前者告诉你更新幅度平均多大；后者告诉你多少 token 被 clip 影响。它们也要按 mask 归一化。

## 6. 和 SLiME PPO Utils 对照

SLiME 的 `compute_approx_kl` 支持不同 KL estimator，其中 k3 / low_var_kl 使用 `exp(log_ratio) - 1 - log_ratio` 形式。patch 里的 reference KL 使用同一类低方差、非负倾向的近似，只是输入写成 `log_probs_ref - log_probs`。

SLiME 的 `compute_policy_loss` 使用 ratio 和 advantage，并在 ratio 超过 clip 区间时切到 clipped 分支。真实实现还支持 dual-clip 等扩展。L39 patch 只保留最小 clip 语义，方便先看清公式。

`compute_opsm_mask` 展示了另一个训练稳定性工具：当负 advantage 序列的 sequence-level KL 超过阈值时，可以把该序列 mask 掉。它不是 L39 patch 的必要实现，但说明真实 RL 训练会继续围绕 KL、advantage 和 mask 做保护。

## 7. Smoke 怎么读

`run_grpo_smoke.py` 构造一个很小的可训练参数 `policy`，用随机方向生成合成 reward，再用 patch 的 `grpo_advantage` 和 `grpo_loss` 跑 50 步。它写出 `metrics.jsonl` 和 `artifacts/grpo_smoke.json`。

这个 smoke 的作用是检查证据格式和数值闭环：loss 是否有下降，KL 是否非负，ratio_mean 和 clipped fraction 是否被记录。它不是训练真实 Qwen，也不能证明长训稳定。H200 配置只提供 verl 命令模板，真实验证还需要 GPU、数据、reward 和 rollout 配置。

## 8. 排障顺序

GRPO/RLOO 失败时先看数值，再看模型。

1. 检查 reward 是否按 prompt 分组，group size 是否符合配置。
2. 检查 GRPO advantage 是否均值接近 0、std 接近 1；常数 reward 是否返回 0。
3. 检查 RLOO 是否排除了自己的 reward。
4. 检查 mask 是否只覆盖 completion token，loss 是否除以 mask.sum。
5. 检查 ratio_mean、ratio_clipped_frac、KL 和 loss 是否同时记录。
6. 如果 KL 或 ratio 爆掉，再回头看 rollout freshness、old_log_probs、reference log_probs 和学习率。

## Lab 验收边界

patch 命令：

```bash
make patch-test M=l34_grpo
```

参考实现验证：

```bash
IMPL=reference make patch-test M=l34_grpo
IMPL=reference python labs/l34_grpo/scripts/run_grpo_smoke.py --run-id l39_local
```

patch 验收的是最小数值合同：advantage、RLOO、ratio clip、k3 KL、mask 和指标返回。测试通过后，还要能把这些字段对应到 smoke 产物和真实 PPO/GRPO 训练日志。

---

## 补充：GRPO/RLOO 与 Value-Critic PPO 的对比

### 为什么去掉 Value Critic？

Value-critic PPO 的问题：
1. **额外模型成本**：Critic 和 Actor 通常同等大小，显存翻倍。
2. **训练不稳定**：Critic 和 Actor 互相依赖——Critic 不准则 advantage 不准，advantage 不准则 Actor 更新方向错误。
3. **Value head 初始化**：从 SFT checkpoint 加 value head，初始 value 预测很差，前期 advantage 全是噪声。

GRPO/RLOO 用组内相对 reward 代替 learned value baseline，省掉了整个 critic。

### 适用场景对比

| 场景 | 推荐方法 | 原因 |
|---|---|---|
| 数学推理（rule-based reward） | GRPO/RLOO | Reward 精确，group 内区分度高 |
| 通用对话（reward model） | PPO with critic | Reward model 本身有噪声，value baseline 能降低方差 |
| Group size 受限（如 G=2） | PPO with critic | 太少样本无法构建稳定 baseline |
| 显存紧张 | GRPO/RLOO | 省掉 critic 模型 |
| 长期 reward 结构 | PPO + GAE | GAE 能处理 token-level credit assignment |
| 稀疏末尾 reward | GRPO/RLOO | 反正只有一个 reward，不需要 temporal decomposition |

### Group Size 的影响

```
G=1:  无法计算 advantage（返回 0），等于没训练
G=2:  baseline 方差极大（只看另一个样本）
G=4:  最小可用，但 advantage 估计仍有较大方差
G=8:  标准配置，平衡采样成本和估计质量
G=16: 更稳定但采样成本高（每个 prompt 16 次 rollout）
```

### 超参数建议

| 参数 | 常用值 | 作用 |
|---|---|---|
| Group size G | 4-16 | 越大估计越稳，但采样成本线性增长 |
| Clip eps | 0.2 | 限制单步更新幅度 |
| KL beta | 0.01-0.1 | Reference KL 惩罚强度 |
| num_epochs per batch | 1-4 | 每批数据用几次（重用会偏离 on-policy） |

### GRPO 的失败信号

| 指标 | 健康范围 | 异常信号 |
|---|---|---|
| reward_mean | 随训练上升 | 持续不变 → reward parser 可能有 bug |
| advantage_std | 接近 1.0 | >> 1 → 组内 reward 差异过大；≈ 0 → reward 无区分度 |
| ratio_mean | 接近 1.0 | >> 1.5 → policy 偏离 old policy 太多 |
| clipped_fraction | < 0.3 | > 0.5 → clip 发生太频繁，学习率或 KL 有问题 |
| kl | 缓慢上升 | 快速上升 → policy diverging from reference |
