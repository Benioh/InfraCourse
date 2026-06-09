# L18 课后产物

本目录放 L18 的复盘材料。它们面向真实多模态数据问题：样本清单、batch 合同、shard 组织和 dataloader 状态。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按层定位 manifest、collator、shard、worker 或 forward 问题 |
| `source_reading_card.md` | 快速回顾 patch、MiniInfra 和 Megatron 源码主路径 |
| `data_pipeline_template.md` | 记录一次 smoke、训练或排查的命令、指标和结论 |

跑完 patch-test、smoke 或真实训练后，把配置、输入规模、关键张量 shape、artifact 路径和判断写进模板。缺少这些证据时，结论只能算临时判断。
