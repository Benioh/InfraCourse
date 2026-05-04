# L11.8 · GRPO / RLOO：DeepSeek-R1 同款 advantage 估计

> 本关只做一件事：**手写 GRPO（Group Relative Policy Optimization）和 RLOO
> （Leave-One-Out）的 advantage + loss**——这是 DeepSeek-R1、Qwen2.5-Math、verl
> 等开源 RL 主流的核心 60 行代码。

之前 RL 段只学了 vanilla PPO 的 KL controller（L10）。L11.8 把现代 RL 的
critic-free advantage 估计补上，与 L10.3 DPO 形成完整"DPO / PPO / GRPO / RLOO"对照。

## 闭环

```bash
cat labs/l34_grpo/patch/task.md
$EDITOR labs/l34_grpo/patch/starter/grpo.py
make patch-test M=l34_grpo
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_group_advantage_zero_mean` | 同 group 内 advantage 均值 ≈ 0 |
| `test_group_advantage_unit_std` | 同 group 内 std ≈ 1（除常数偏移） |
| `test_single_response_group_returns_zero` | G=1 时 advantage 全为 0（不能 NaN） |
| `test_rloo_advantage_uses_other_responses` | A_i 与 r_{j≠i} 相关，不与 r_i 相关 |
| `test_rloo_zero_when_all_equal` | 所有 reward 相等时 A==0 |
| `test_grpo_loss_with_kl_penalty` | β=0.1 时 loss 包含 KL 项 |
| `test_grpo_loss_clip_caps_ratio` | ratio 超出 [1-ε, 1+ε] 时被截断 |
| `test_grpo_loss_per_token_mask` | mask=0 的位置不计入 |

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 合成 reward + log-prob，跑 50 步看 loss 下降 |
| `configs/h200_qwen.yaml` | 真实 Qwen2.5-7B + GSM8K rule-based reward |

## 进入下一关

通过后回到 [L11.5 SLiME rollout freshness](../l33_rl_rollout_freshness/README.md) 或
进入 [L12 capstone](../l35_multimodal_capstone/README.md)。
