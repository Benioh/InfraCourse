# L19 课后产物

本目录放数据工程复盘材料，用于记录 MinHash 去重、WebDataset-style pipeline 和 shard recovery 的证据。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按层定位 dedup、shard、worker、resume cursor 问题 |
| `source_reading_card.md` | 快速回顾 patch 和 MiniInfra 数据工程源码 |
| `data_pipeline_template.md` | 记录一次 smoke、benchmark、训练或排查 |

跑完 patch-test、notebook、smoke 或真实数据任务后，把命令、输入规模、阈值、`num_perm`、dedup ratio、throughput 和 recovery artifact 写进模板。
