# L28 PD Cache / Transfer 复盘模板

## 1. Run 信息

- 日期：
- run id：
- git commit：
- 机器 / GPU：
- 模型：
- workload：
- prefill workers：
- decode workers：
- request 数：
- prompt token 分布：
- cache hit rate：
- cached prefix ratio：

## 2. 命令与配置

- patch-test 命令：
- drill 命令：
- 配置文件：
- IMPL：
- baseline run：
- current run：
- 只改变的变量：

## 3. 预期

- 这次要验证的机制：
- 成功标准：
- 可能代价：

## 4. 关键指标

| 指标 | cache off | cache on | 判断 |
|---|---:|---:|---|
| prompt_tokens total |  |  |  |
| cached_prefix_tokens |  |  |  |
| prefill_tokens |  |  |  |
| saving ratio |  |  |  |
| kv_transfers |  |  |  |
| transfers/request min-max |  |  |  |
| tokens_transferred |  |  |  |
| active_requests after complete |  |  |  |
| worker_loads after complete |  |  |  |

## 5. Worker 分布

| Worker | role | load before complete | load after complete | 判断 |
|---|---|---:|---:|---|
|  |  |  |  |  |

## 6. Artifact

| 文件 | 路径 | 说明 |
|---|---|---|
| command |  |  |
| resolved config |  |  |
| metrics.jsonl |  |  |
| pd_drill.json |  |  |
| report.md |  |  |
| sglang_pd_command.sh |  |  |

## 7. 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
|  |  |  |

## 8. 结论

- 本次能证明什么：
- 不能证明什么：
- 主要瓶颈或风险：
- 下一步只改哪一个变量：
