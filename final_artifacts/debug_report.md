# Debug Report

本报告汇总至少 5 个 debug ticket，覆盖环境、训练、数据、serving、RL。每个 ticket 都要有最小复现、最小修复和 evidence path；不要只贴错误日志。

| Ticket ID | 阶段 | 失败签名 | 最小检查 | 修复动作 | 验证证据 |
|---|---|---|---|---|---|
|  | 环境 |  |  |  |  |
|  | 训练 |  |  |  |  |
|  | 数据 |  |  |  |  |
|  | Serving |  |  |  |  |
|  | RL |  |  |  |  |

## 共同模式

- 哪些失败来自配置语义？
- 哪些失败来自硬件/环境？
- 哪些失败来自数据 schema 或 tokenizer/processor？
- 哪些失败只能在 8×H200 上验证？
