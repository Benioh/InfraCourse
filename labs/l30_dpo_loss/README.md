# L10.3 · DPO Loss：从 Bradley-Terry 推导到 PyTorch 实现

> 本关只做一件事：**实现 DPO（Direct Preference Optimization）loss + 完成
> log-prob 抽取**——这是 DeepSeek-R1、Qwen-Chat、Tülu-2 等开源 RLHF 主流方案
> 的核心 50 行代码。

之前 RL 段直接跳到 PPO（L10/L11）。L10.3 补上工业现已默认的 DPO，并和 L09.8 SFT
形成"SFT → DPO"完整对齐链路。

## 闭环

```bash
cat labs/l30_dpo_loss/patch/task.md
$EDITOR labs/l30_dpo_loss/patch/starter/dpo.py
make patch-test M=l30_dpo_loss
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_compute_logps_shape` | 输出形状 [B] |
| `test_compute_logps_masks_prompt` | prompt 部分（label==-100）不计入 logp |
| `test_compute_logps_matches_manual` | 与手写 gather + log_softmax 一致 |
| `test_dpo_zero_beta_returns_log2` | β=0 时 loss == log(2) |
| `test_dpo_policy_equals_ref_returns_log2` | policy==ref 时 loss == log(2) |
| `test_dpo_chosen_better_lowers_loss` | chosen logp 抬升后 loss 单调下降 |
| `test_dpo_rejected_better_raises_loss` | rejected 更强时 loss > log(2) |
| `test_dpo_reward_margin_formula` | reward_margin == β·(logp_chosen - logp_rejected - logp_ref_chosen + logp_ref_rejected) |
| `test_dpo_numerically_stable_with_large_beta` | β=20 不出现 inf/NaN（用 logsigmoid，不要 log(sigmoid(...)) ） |

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | tiny preference 数据集，CPU 跑 50 步看 loss 下降 |
| `configs/4090_qwen.yaml` | Qwen2.5-0.5B + UltraFeedback 5k preference pairs |

## 调试工单

见 `tickets/INDEX.md`。

## 进入下一关

通过后进入 [L10.5 rollout pool](../l31_rollout_only_smoke/README.md) → [L11 SLiME RL core](../l32_slime_rl_core/README.md)。
