# L08 课后产物使用说明

本目录保存 Megatron 数据预处理的复用材料。patch 证明 `.bin/.idx` 最小格式能写能读，outputs 用来把这条数据链路迁移到真实预训练项目。

| 文件 | 用途 |
|---|---|
| [debug_checklist.md](debug_checklist.md) | 排查样本数异常、token 数异常、idx/bin 不匹配、dtype 越界和 Megatron 读取失败 |
| [source_reading_card.md](source_reading_card.md) | 复习 patch、drill、MiniInfra 和 Megatron 源码主路径 |
| [data_pipeline_template.md](data_pipeline_template.md) | 记录一次预处理运行的输入、输出、指标、命令和判断 |

建议每次更换 tokenizer、清洗规则、json key、dtype、数据 split 或 output prefix 时，都填一次 `data_pipeline_template.md`。数据 artifact 是后续训练复现的入口。
