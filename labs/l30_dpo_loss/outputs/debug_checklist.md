# Debug Checklist：L33 DPO Loss

## 1. 固定现场

- 记录命令、配置、git commit、Python/PyTorch 版本、数据来源、beta 和随机种子。
- 保存一两条 prompt、chosen、rejected、labels、attention mask 和 tokenizer 输出。
- 保存 `metrics.jsonl`、`artifacts/dpo_smoke.json`、report、stdout/stderr 和 checkpoint 路径。

## 2. 检查输入语义

| 检查项 | 证据 | 失败时的判断 |
|---|---|---|
| chosen/rejected 方向 | 抽样人工检查偏好对 | 数据 swap 会让训练朝反方向优化 |
| prompt mask | prompt/pad/tool 位置为 `-100` | prompt 混入会污染 completion logprob |
| shape | policy/ref/chosen/rejected logprob shape 一致 | loss 公式输入不可靠 |
| reference | ref 参数冻结或 no_grad | reference 跟着 policy 漂移，anchor 失效 |

## 3. 检查 loss 证据

- `loss`：head/tail 是否下降，下降是否来自少数 batch。
- `reward_margin_mean`：是否向正方向移动。
- `chosen_reward` 与 `rejected_reward`：是否符合偏好方向。
- beta：是否过大导致 margin 饱和，或过小导致信号弱。
- 数值：是否出现 inf/NaN，是否使用 `F.logsigmoid`。

## 4. 沿源码主路径复查

- `labs/l30_dpo_loss/patch/starter/dpo.py`：学生实现的 logprob 抽取和 DPO loss。
- `labs/l30_dpo_loss/patch/reference/dpo.py`：最小正确公式。
- `labs/l30_dpo_loss/patch/tests/test_patch.py`：9 个行为合同。
- `labs/l30_dpo_loss/scripts/run_dpo_smoke.py`：synthetic smoke 的训练循环和 artifact。
- `github_repo/slime/slime/utils/ppo_utils.py`：真实 RL 框架中的 logprob 与 PPO 对照。

## 5. 常见错误判断

- 只看 loss 下降，没有检查 chosen/rejected 是否被反向标注。
- 把 prompt token 加进 completion logprob。
- reference 参数进入 optimizer。
- 使用 `torch.log(torch.sigmoid(x))`，大 margin 下出现下溢。
- smoke 通过后直接推断真实数据训练稳定。

## 6. 结束条件

- 问题能被最小 batch 或一条 smoke 命令复现。
- 数据样本、mask、logprob、loss 和 artifact 都已保存。
- 能指出源码中 logprob、reward margin 或 loss 的状态变化位置。
- 结论写入 `rl_rollout_template.md`，并包含下一步动作。
