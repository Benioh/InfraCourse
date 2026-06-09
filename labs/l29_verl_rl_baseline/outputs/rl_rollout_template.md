# L31 RL Smoke 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- run 目录：
- git commit：
- 数据或 workload：

## 预期

- 这次验证的机制：
- 比较对象：
- 成功标准：
- 已知边界：

## Reward 入口

| 检查项 | 结果 / 路径 | 判断 |
|---|---|---|
| reward self-test |  |  |
| prediction 抽样 |  |  |
| target 抽样 |  |  |
| 失败原因分布 |  |  |

## RL 指标

| 指标 | 起始值 | 结束值 | 判断 |
|---|---:|---:|---|
| reward_mean |  |  |  |
| kl_mean |  |  |  |
| entropy |  |  |  |
| response_len_mean |  |  |  |
| rollout_time_sec |  |  |  |
| update_time_sec |  |  |  |
| samples_per_sec |  |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| KL 系数变化 | `patch/reference/kl_controller.py` |  |
| reward 解析 | `scripts/reward_math.py` |  |
| metrics 落盘 | `scripts/run_verl_lab.py` |  |
| KL estimator / PPO clip | `github_repo/slime/slime/utils/ppo_utils.py` |  |
| rollout / train loop | `github_repo/slime/train_async.py` |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 如果 KL 异常，下一步检查：
- 如果 reward 异常，下一步检查：
- 如果 rollout 慢，下一步检查：
- 需要修改的配置、代码或实验：
