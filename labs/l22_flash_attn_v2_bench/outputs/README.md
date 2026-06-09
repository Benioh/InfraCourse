# 课后产物：L23 PyTorch SDPA 与 FlashAttention benchmark

本目录存放 L23 课后可以继续复用的材料。它们用于复盘 attention kernel benchmark、排查 SDPA fallback 和整理源码阅读结论。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序排查数值不一致、backend fallback、计时失真和显存统计异常 |
| `source_reading_card.md` | 快速回忆 eager baseline、SDPA 调用、online softmax 和真实 wrapper |
| `serving_metrics_template.md` | 记录一次 attention benchmark 的条件、指标、源码对应和结论 |

每次跑完 patch、CPU smoke 或 GPU benchmark 后，至少记录命令、配置、device、dtype、shape、iters、causal、max_abs_diff、time、speedup 和 peak memory。缺少这些字段时，性能结论不能复用到其他模型或硬件。
