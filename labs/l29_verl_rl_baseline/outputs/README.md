# 课后产物：L31 Adaptive KL Controller

本目录保存三份可复用材料，用来复盘 KL controller、reward parser 和 RL smoke 证据链。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 排查 KL 爆炸、KL 长期过低、reward parse 错误、rollout 慢和指标缺失 |
| `source_reading_card.md` | 快速回忆 controller、reward smoke、SLiME PPO utils 和 async train loop 的阅读顺序 |
| `rl_rollout_template.md` | 记录一次 RL smoke、rollout 或训练 run 的配置、指标、判断和下一步 |

使用这些材料时，先写清命令、配置、输入数据、run 目录和关键指标。没有 reward self-test、KL/entropy/length 曲线和 rollout/update 时间拆分时，不要给训练健康度下结论。
