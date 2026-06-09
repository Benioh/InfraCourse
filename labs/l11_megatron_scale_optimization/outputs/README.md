# L12 课后产物：Megatron Scale Optimization

本目录存放 L12 课后可以继续复用的材料。它们用于后续排查 data parallel 通信、低 MFU、并行配置和 checkpoint 切片问题，也可以直接作为记录模板。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序定位 bucketed grad sync、scale 配置和 checkpoint layout 问题 |
| `source_reading_card.md` | 快速回忆 BucketedManualDDP、Megatron DDP 和 TorchTitan mesh 主路径 |
| `training_step_template.md` | 记录一次 patch、drill、benchmark 或训练扩展评审 |

建议每次跑完 patch、drill 或真实任务后，把命令、配置、world size、bucket size、并行度、关键指标和结论写进模板。缺少这些证据时，不要把局部现象写成生产结论。
