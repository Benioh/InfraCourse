# Final Infra Delivery README

本文件是 L12 Capstone 的总入口。目标不是展示一次 demo，而是让另一个工程师能根据证据复现你的环境、数据、训练、serving、RL 与 debug 结论。

## 项目摘要

- 项目目标：
- 覆盖阶段：环境 / 数据 / 训练 / Serving / RL / Debug
- 运行边界：0 GPU / 1×4090 / 8×H200 / validation-only
- 最终结论：

## 证据索引

| 阶段 | 关键 run_id | command | config | metrics | report | 真实运行边界 |
|---|---|---|---|---|---|---|
| 环境 |  |  |  |  |  |  |
| 数据 |  |  |  |  |  |  |
| 训练 |  |  |  |  |  |  |
| Serving |  |  |  |  |  |  |
| RL |  |  |  |  |  |  |

## 架构说明

请引用 `architecture.mmd`，说明数据如何进入训练，checkpoint 如何进入 serving/RL，metrics 如何回到报告。

## 迁移判断

- 可以直接迁移的配置：
- 必须在 8×H200 重新验证的假设：
- 仍未覆盖的 failure mode：

## 下一步

列出 3 个最小后续实验，每个实验都必须说明只改变哪个变量、预期指标和回滚方案。
