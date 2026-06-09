# L36 Weight Sync 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- run 目录：
- git commit：
- actor_gpus：
- rollout_gpus：
- sync interval：

## 预期

- 这次验证的机制：
- 只改变的变量：
- 成功标准：
- 已知边界：

## 同步合同

| 检查项 | 结果 / 路径 | 判断 |
|---|---|---|
| num_tensors |  |  |
| bytes_synced |  |  |
| mismatched_keys |  |  |
| dtype 策略 |  |  |
| shape/name 转换 |  |  |

## RL 指标

| 指标 | 数值 | 判断 |
|---|---:|---|
| rollout_time_sec |  |  |
| actor_update_time_sec |  |  |
| weight_sync_time_sec |  |  |
| sglang_generation_tokens_per_sec |  |  |
| weight_version |  |  |
| effective_bandwidth |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| 本地 shape/dtype gate | `patch/reference/weight_sync.py` |  |
| MiniInfra version | `mini_infra/slime/ray/rollout.py` |  |
| SLiME train loop | `github_repo/slime/train.py` |  |
| updater choice | `github_repo/slime/slime/backends/megatron_utils/actor.py` |  |
| distributed sync | `update_weight_from_distributed.py` |  |
| SGLang engine update | `sglang_engine.py` |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 如果 sync 慢，下一步检查：
- 如果 mismatch 多，下一步检查：
- 如果 version 不推进，下一步检查：
- 下一次实验只改的变量：
