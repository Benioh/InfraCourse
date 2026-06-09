# Debug Checklist：L31 Adaptive KL Controller

## 1. 固定现场

- 记录命令、配置、git commit、Python 环境、硬件、数据路径和随机种子。
- 保存 `command.sh`、`config.resolved.yaml`、`metrics.jsonl`、`rl.log`、`report.md` 和 `artifacts/reward_self_test.json`。
- 标记运行类型：patch-test、notebook、CPU smoke、真实 verl/SLiME 训练或线上回放。

## 2. 先验证 reward 入口

- 运行 `python labs/l29_verl_rl_baseline/scripts/reward_math.py --self-test`。
- 如果 self-test 失败，先修 `extract_final_number`、target 格式或 tolerance。
- 抽样检查 prediction 和 target，确认最终数字提取符合预期。
- 不要在 reward parser 失败时调 `kl_coef`、learning rate 或 PPO epochs。

## 3. 再看 KL 控制

| 现象 | 优先检查 | 可能动作 |
|---|---|---|
| step 0 KL 很高 | reference checkpoint、tokenizer、mask、logprob 对齐 | 先修输入和 reference |
| KL 随 update 快速上涨 | `kl_coef`、`target_kl`、horizon、learning rate、PPO epochs、ratio clip | 调慢更新或增大惩罚 |
| KL 长期接近 0 | reward 太弱、KL 惩罚过强、learning rate 太低、advantage 被压平 | 降低惩罚或检查 reward |
| entropy 很快塌陷 | reward outlier、response 重复、clip 过宽、采样温度 | 看样本内容和长度分布 |

## 4. 拆 rollout 和 update 时间

- `rollout_time_sec` 高：看 engine、并发上限、batch、`max_new_tokens`、stop token、prefix cache、tokenizer CPU 和 weight sync。
- `update_time_sec` 高：看 actor/critic 训练、通信、显存、gradient accumulation 和 checkpoint。
- `response_len_mean` 贴近上限时，先检查 stop token 和 prompt 模板。

## 5. 对照源码主路径

- `patch/starter/kl_controller.py`：学生实现的状态机入口。
- `patch/reference/kl_controller.py`：公式的最小参考实现。
- `patch/tests/test_patch.py`：五个数值合同。
- `scripts/reward_math.py`：最终数字提取、reward 和失败原因。
- `scripts/run_verl_lab.py`：run 目录、reward self-test 和模拟 RL metrics。
- `github_repo/slime/slime/utils/ppo_utils.py`：KL estimator、ratio clip 和 KL penalty 进入 returns/advantages 的位置。
- `github_repo/slime/train_async.py`：rollout、actor/critic train 和 weight sync 主循环。

## 6. 结束条件

- 问题可以用一个最小命令复现。
- reward self-test 状态明确。
- KL、reward、entropy、response length、rollout/update time 都有记录。
- 能指出异常来自 reward 入口、KL estimator/controller、PPO update、rollout engine 还是指标缺失。
- 结论写入 `rl_rollout_template.md`，并列出下一步要改的配置或代码。
