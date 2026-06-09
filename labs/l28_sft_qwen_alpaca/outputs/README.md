# L29 课后产物

本目录保存 SFT 数据处理和训练排查时可以复用的材料。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按 role、tokenizer、labels、pad 和 loss 顺序定位问题 |
| `source_reading_card.md` | 快速回忆 L29 源码主路径 |
| `sft_training_template.md` | 记录一次 SFT 数据处理或训练复盘 |

每次跑 patch、smoke 或真实训练后，至少记录命令、配置、模型、tokenizer、数据量、max length、关键指标和 artifacts。没有这些条件，loss 下降或样本输出都不能支撑可复查结论。
