# L39 Source Reading Card：GRPO / RLOO

## 主路径

1. `labs/l34_grpo/patch/starter/grpo.py`：学生要补的三个函数。
2. `labs/l34_grpo/patch/reference/grpo.py`：GRPO/RLOO advantage 与 loss 参考实现。
3. `labs/l34_grpo/patch/tests/test_patch.py`：9 个数值边界测试。
4. `mini_infra/rl/reward.py`：rule-based reward 输出。
5. `github_repo/slime/slime/utils/ppo_utils.py`：KL estimator、OPSM 和 policy loss。
6. `labs/l34_grpo/scripts/run_grpo_smoke.py`：合成 smoke 和产物写出。

## 阅读方法

1. 先确认每个张量 shape。
2. 再看 group 边界和 baseline。
3. 接着看 mask 参与了哪些平均。
4. 最后看 metrics 是否能支撑结论。

## 自检

- 我能否解释 GRPO、RLOO 和 PPO value baseline 的差异？
- 我能否写出 ratio clip 的两个分支？
- 我能否指出真实 SLiME 比 patch 多出的稳定性保护？
