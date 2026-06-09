# L03 课后产物使用说明

本目录保存 Memory Snapshot 泄露归因的复用材料。patch 证明最小 tracker 合同成立，outputs 用来把这个合同迁移到真实 OOM 排查。

| 文件 | 用途 |
|---|---|
| [debug_checklist.md](debug_checklist.md) | 长期 OOM、snapshot dump、rank/worker 选择和 stack 聚合排查顺序 |
| [source_reading_card.md](source_reading_card.md) | 快速复习 starter、reference 和 tests 的主路径 |
| [performance_metrics_template.md](performance_metrics_template.md) | 记录一次显存泄露事件的命令、时间线、top stack、判断和修复 |

建议在完成 `make patch-test M=l02.5_memory_snapshot` 后，用 README 里的 CPU drill 制造一次小泄露，再把结果填进模板。真实训练中使用时，把 `total_leaked_bytes`、top stack、rank、step 区间和修复动作一起记录。
