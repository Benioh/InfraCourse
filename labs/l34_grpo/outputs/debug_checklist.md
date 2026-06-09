# L39 Debug Checklist：GRPO / RLOO

## 1. 固定现场

- 记录命令、配置、run id、git commit、PyTorch 版本和随机种子。
- 记录 group size、prompts、seq_length、clip_eps、kl_beta 和 mask 规则。
- 保存 `metrics.jsonl`、`artifacts/grpo_smoke.json`、`report.md` 和 patch-test 输出。

## 2. 先看数值边界

| 问题 | 要看什么 | 可能结论 |
|---|---|---|
| advantage NaN | group size、std、常数 reward | G=1 或 std=0 没处理 |
| advantage 信号弱 | reward 分布、组内方差 | reward parser 区分度不足 |
| RLOO 结果偏 | baseline 是否排除当前 response | 把 GRPO 均值当成 RLOO |
| loss 不随 mask 变化 | mask.sum、prompt token mask | mask 没参与归一化 |
| KL 或 ratio 爆 | log_probs 差值、kl_beta、clip_eps | 更新幅度过大或 reference 对齐异常 |

## 3. 沿源码主路径复查

1. `labs/l34_grpo/patch/reference/grpo.py`：确认 advantage、ratio、clip、KL 和 mask 归一化。
2. `labs/l34_grpo/patch/tests/test_patch.py`：确认失败的是哪个数值合同。
3. `mini_infra/rl/reward.py`：确认 reward 是否能给同组 responses 拉开差距。
4. `github_repo/slime/slime/utils/ppo_utils.py`：对照 KL estimator、OPSM 和 policy loss。
5. `labs/l34_grpo/scripts/run_grpo_smoke.py`：确认 metrics 和 artifact 是否完整。

## 4. 常见错误判断

- 把不同 prompt 的 rewards 放进同一个 group。
- 对 prompt token 计算 policy loss。
- 全 0 mask 时直接除以 0。
- ratio_mean 记录了未 mask 的所有 token。
- smoke loss 下降就判断真实模型长训稳定。

## 5. 结束条件

- patch-test 能定位到一个明确合同。
- smoke 产物包含 loss、policy_loss、kl_loss、ratio_mean 和 ratio_clipped_frac。
- 结论写进 `rl_rollout_template.md`，并说明不能证明的范围。
