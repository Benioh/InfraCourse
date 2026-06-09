# L21 课后产物

这一目录放完成本讲后可以继续复用的材料。patch 证明你实现了最小合同，这些产物帮助你把知识带到真实 serving 排查里。

| 文件 | 用途 |
|---|---|
| [debug_checklist.md](debug_checklist.md) | 排查 TTFT、ITL、waiting queue、KV pressure 时按顺序使用 |
| [source_reading_card.md](source_reading_card.md) | 读 vLLM engine、scheduler、KV manager 时的源码卡 |
| [serving_metrics_template.md](serving_metrics_template.md) | 跑 drill 或真实 benchmark 后填写的指标复盘模板 |

建议在完成 `make patch-test M=l20_vllm_scheduler_kv` 和 `run_serving_drill.sh` 后，把 drill 结果填进 `serving_metrics_template.md`，再用 `debug_checklist.md` 做一次自查。
