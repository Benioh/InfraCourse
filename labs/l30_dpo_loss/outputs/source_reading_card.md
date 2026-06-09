# Source Reading Card：L33 DPO Loss

## 主路径

1. `labs/l30_dpo_loss/patch/starter/dpo.py`：学生要实现的 completion logprob 和 DPO loss。
2. `labs/l30_dpo_loss/patch/reference/dpo.py`：最小正确公式。
3. `labs/l30_dpo_loss/patch/tests/test_patch.py`：9 个数值、shape、mask 和稳定性合同。
4. `labs/l30_dpo_loss/scripts/run_dpo_smoke.py`：合成偏好数据上的最小训练循环。
5. `github_repo/slime/slime/utils/ppo_utils.py`：真实 RL 框架的 logprob 与 PPO 对照。

## 阅读顺序

1. 先看 reference 第 9-15 行，确认 completion-only logprob。
2. 再看 reference 第 25-34 行，确认 DPO reward margin 和 loss。
3. 然后看 tests，把每个断言对应到一个行为合同。
4. 最后看 smoke 的 labels mask、reference freeze、loss 调用和 artifact 写入。

## 自检

- 我能否解释 safe labels 为什么不会改变 masked 位置的结果？
- 我能否手写 reward margin 公式？
- 我能否指出 DPO smoke 能证明和不能证明的内容？
- 我能否说明 DPO 与 PPO 对 logprob 的使用差异？
