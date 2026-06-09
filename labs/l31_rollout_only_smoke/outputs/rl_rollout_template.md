# L35 Rollout-only 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- endpoint / mock：
- 命令：
- 配置文件：
- run 目录：
- git commit：
- prompt 集：

## 预期

- 这次验证的机制：
- 只改变的变量：
- 成功标准：
- 已知边界：

## 样本检查

| 检查项 | 结果 / 路径 | 判断 |
|---|---|---|
| rollouts 数量 |  |  |
| prompt/response 是否对齐 |  |  |
| reward_input_ready |  |  |
| response_len 抽样 |  |  |
| 异常或 timeout |  |  |

## 指标

| 指标 | 数值 | 判断 |
|---|---:|---|
| request_count |  |  |
| max_concurrency |  |  |
| rollouts_per_sec |  |  |
| avg_ttft_ms |  |  |
| avg_output_tokens |  |  |
| server queue / TTFT |  |  |
| weight_sync_sec |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| 客户端限流 | `patch/reference/rollout_pool.py` |  |
| 本地 artifact | `scripts/run_rollout_only.py` |  |
| vLLM 请求返回 | `github_repo/vllm/vllm/v1/engine/async_llm.py` |  |
| SGLang batch 状态 | `github_repo/sglang/python/sglang/srt/managers/scheduler.py` |  |
| SLiME rollout | `github_repo/slime/slime/rollout/sglang_rollout.py` |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 如果 rollout 慢，下一步检查：
- 如果 reward-ready 缺失，下一步检查：
- 如果样本顺序异常，下一步检查：
- 下一次实验只改的变量：
