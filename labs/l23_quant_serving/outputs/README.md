# 课后产物：L24 AWQ-lite W8 Per-Channel 量化 Serving

本目录存放 L24 课后可以继续复用的材料。它们用于复盘量化 serving、排查 AWQ/FP8/KV scale 问题，并整理源码阅读结论。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序排查 scale shape、量化误差、checkpoint 加载、accuracy drift 和 KV cache 漂移 |
| `source_reading_card.md` | 快速回忆 patch、MiniInfra 和真实 vLLM quantization 主路径 |
| `serving_metrics_template.md` | 记录一次量化 serving 的显存、速度、准确率和硬件边界 |

每次跑完 patch、notebook、smoke 或真实 benchmark 后，至少记录 quant method、bits、group_size、activation dtype、校准数据、硬件、workload、TTFT、ITL、tokens/s、peak memory 和 acc_drop_pp。缺少这些条件时，量化结论不能外推。
