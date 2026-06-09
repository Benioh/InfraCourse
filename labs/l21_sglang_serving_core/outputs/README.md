# L22 课后产物

本目录保存 prefix-cache 排查可以复用的材料。patch 只能证明 trie 合同；这些模板用于把 repeated-prefix workload、server 状态、metrics 和源码证据放到同一份复盘里。

| 文件 | 用途 |
|---|---|
| [debug_checklist.md](debug_checklist.md) | 排查 cache miss、TTFT 高、server 不可用和 validation-only |
| [source_reading_card.md](source_reading_card.md) | 复习 patch、MiniInfra 和真实 SGLang 的源码主路径 |
| [serving_metrics_template.md](serving_metrics_template.md) | 记录 repeated-prefix benchmark 的 workload、指标、证据和结论边界 |

建议在完成 `make patch-test M=l21_sglang_serving_core`、`run_server.py` 和 `bench_repeated_prefix.py` 后，把结果填进模板。没有真实 server 时，模板应明确写出 validation-only。
