# L27 课后产物：SGLang PD Observability

本目录存放 L27 之后可复用的观测材料。它们用于排查 serving 指标缺失、PD 分离配置不匹配、decode 饥饿、KV transfer 慢和 label 不稳定等问题。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按阶段检查 metrics endpoint、scrape、queue、cache、KV transfer 和证据链 |
| `source_reading_card.md` | 复习 patch、PD lab、SGLang metrics collector 和 scheduler 写指标路径 |
| `serving_metrics_template.md` | 记录一次 SGLang 或 MiniInfra serving 观测复盘 |

使用这些材料时，先固定命令、配置、模型、workload、硬件和 run id。没有这些上下文，单个 p99、TTFT 或 token/s 数值无法支持工程判断。
