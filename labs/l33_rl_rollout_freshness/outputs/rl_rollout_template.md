# L38 Rollout Freshness 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- git commit：
- 数据或 prompts：
- server 列表：
- `max_staleness`：
- `update_rate` / `update_subset_size`：

## 预期

- 本次要验证的 freshness 合同：
- 允许的最大 staleness：
- no-update 场景是否应该失败：
- 成功标准：

## 版本证据

| server_id | actor_version | weight_version | staleness | 是否 fresh | 备注 |
|---|---:|---:|---:|---|---|
|  |  |  |  |  |  |

## 指标和产物

| 指标或 artifact | 数值 / 路径 | 解释 |
|---|---|---|
| failures |  |  |
| successful_rollouts |  |  |
| staleness_p50 |  |  |
| staleness_p99 |  |  |
| reward_mean |  |  |
| kl_mean |  |  |
| metrics.jsonl |  |  |
| artifacts/rl_drill.json |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| stale server 被跳过 | `patch/reference/rollout_manager.py` |  |
| 局部更新只改指定 server | `patch/reference/rollout_manager.py` |  |
| Sample 收集 weight_version | `github_repo/slime/slime/utils/types.py` |  |
| SGLang 查询 weight_version | `github_repo/slime/slime/backends/sglang_utils/sglang_engine.py` |  |

## 结论

- 本次能证明什么：
- 不能证明什么：
- 下一步要改的配置、代码或实验：
