# L02 课后产物

本目录存放 PyTorch 显存账本和训练 step 排查材料。它们用于后续训练系统问题复盘，不是作业提交格式。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序排查 OOM、step 变慢、optimizer state 异常和 profiler 误读 |
| `source_reading_card.md` | 快速复习 patch、tiny trainer、MiniInfra trainer 和 Megatron-shaped `train_step` |
| `training_step_template.md` | 跑完 tiny transformer smoke 或真实训练后记录条件、指标和结论 |

建议每次修改 batch size、sequence length、dtype、activation checkpointing、optimizer 或模型层数时，都填一次 `training_step_template.md`。
