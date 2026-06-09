# L20 课后产物

本目录保存 serving baseline 和采样过滤实验的复盘材料。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序定位端口、模型、请求、采样和指标问题 |
| `source_reading_card.md` | 复习 MiniInfra、vLLM sampler 和 patch 源码主路径 |
| `serving_metrics_template.md` | 记录 TTFT、ITL、吞吐、命令和判断 |

每次跑 serving smoke 或真实 benchmark 后，把命令、模型、端口、并发、prompt/output 长度、TTFT、ITL 和状态写进模板。
