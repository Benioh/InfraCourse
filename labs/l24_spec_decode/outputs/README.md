# L25 课后产物

这个目录用于保存 spec decode 的复盘材料。每次跑完 patch、smoke、bench 或真实服务实验后，把命令、输入规模、baseline、关键指标和结论写进模板，避免只留下零散截图或单个吞吐数字。

| 文件 | 用途 |
|---|---|
| `debug_checklist.md` | 按顺序排查低接受率、draft OOM、KV 回滚浪费和 p99 抖动 |
| `source_reading_card.md` | 快速回忆 patch、MiniInfra、vLLM 和 SGLang 主路径 |
| `serving_metrics_template.md` | 记录一次 spec decode serving 的 baseline、acceptance、latency、显存和质量边界 |

建议把 CPU smoke 和真实 GPU 结果分开写。CPU 结果只能证明字段和合同，GPU 结果才适合讨论速度。
