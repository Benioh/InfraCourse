# Risk Register

| 风险 | 阶段 | 触发信号 | 影响 | 当前证据 | 缓解方案 | 下一步验证 |
|---|---|---|---|---|---|---|
| 环境漂移 | 环境 | Python/CUDA/NCCL 版本变化 | 运行不可复现 |  | 固定 env 与 collect_env | 重跑 L00 |
| 数据 schema 漂移 | 数据 | loader shape/key 不一致 | 训练或 reward 错误 |  | schema check + 样本 ID | 重跑 L06 |
| Checkpoint 不兼容 | 训练/Serving/RL | TP/PP/tokenizer 不匹配 | resume/load 失败 |  | 版本化 checkpoint metadata | 小模型加载验证 |
| Serving benchmark 不可比 | Serving | prompt/output 分布变化 | 性能结论无效 |  | 固定 workload artifact | 重跑 L07/L08 |
| Reward/parser 错误 | RL | reward 全 0 或异常高 | RL 方向错误 |  | reward self-test | 重跑 L10/L10.5 |

## 使用要求

每个风险必须连接到一个 evidence path，例如 `metrics.jsonl`、`serve.log`、`rl.log`、`artifacts/*.jsonl` 或具体 ticket。
