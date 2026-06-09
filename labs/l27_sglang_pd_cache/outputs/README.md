# L28 课后产物：SGLang PD Disaggregation

本目录存放 L28 之后可复用的 PD 分离排查材料。它们用于检查 prefix cache saving、KV transfer、worker load、complete 释放和真实 SGLang PD queue。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按 route、cache、transfer、decode、complete 和 artifact 顺序排查 |
| `source_reading_card.md` | 快速回忆 patch、MiniInfra 和真实 SGLang 主路径 |
| `serving_metrics_template.md` | 记录一次 PD cache/transfer drill 或 benchmark 复盘 |

每次复盘都要保存命令、配置、run id、workload、cache off/on 指标、transfer/request 和 worker loads。没有这些证据时，不要把 drill 中的局部现象写成生产结论。
