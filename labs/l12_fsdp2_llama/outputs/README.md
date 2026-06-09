# L13 课后产物：FSDP2

本目录存放 L13 课后可以继续复用的材料。它们面向 FSDP2 wrap、mixed precision、reshard、OOM 和 checkpoint 恢复问题，不是一次性作业模板。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序定位 FSDP2 wrap、显存、dtype 和 checkpoint 问题 |
| `source_reading_card.md` | 快速回忆 patch、smoke、TorchTitan parallelize 和 checkpoint 主路径 |
| `fsdp2_wrap_template.md` | 跑 dryrun、GPU smoke 或真实 benchmark 后填写的复盘模板 |

建议每次跑完 patch、drill 或真实训练后，把命令、配置、world size、wrap 粒度、reshard policy、dtype、loss、peak memory 和 checkpoint 证据写进模板。缺少这些条件时，不要把局部现象写成生产结论。
