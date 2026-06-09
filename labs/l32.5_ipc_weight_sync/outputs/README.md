# 课后产物：L37 CUDA IPC Weight Sync

本目录保存三份可复用材料，用来复盘 co-locate weight sync 的 handle 大小、共享 storage、rank gather、LocalSerializedTensor 和 flush 时序。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 排查 handle/data 比例、rank gather、storage 共享、cache flush 和 IPC 生命周期 |
| `source_reading_card.md` | 快速回忆 patch、SGLang update、SLiME bucket 和 serializer 主路径 |
| `rl_rollout_template.md` | 记录一次 CUDA IPC-shaped weight sync 复盘 |

使用这些材料时，先保存 patch-test 输出、handle bytes、tensor bytes、data_ptr 检查、rank gather 结果和 flush 时序。没有这些证据时，不要把 CPU 模拟结论写成真实 CUDA IPC 结论。
