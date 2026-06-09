# L09 课后产物说明

本目录保存 L09 可复用材料。它们面向真实训练排查，不依赖课堂上下文。

| 文件 | 用途 |
|---|---|
| [debug_checklist.md](debug_checklist.md) | 排查训练启动、LR 曲线、loss 抖动、resume 和日志证据 |
| [source_reading_card.md](source_reading_card.md) | 复习 MiniInfra、Megatron、patch 和 drill 的源码主路径 |
| [training_step_template.md](training_step_template.md) | 记录一次训练运行的配置、artifact、指标、源码判断和结论 |

建议在完成 patch 后，至少用一次 `scripts/train_4090.sh` 或 `parse_megatron_log.py --self-test` 填写模板。若本地只能走 fallback，也要记录缺失条件和可证明的边界。
