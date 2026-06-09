# L26 课后产物

这个目录用于保存 serving eval 的复盘材料。每次跑完 patch、stub eval 或真实 OpenAI-compatible 服务评测后，把命令、配置、样本规模、完成率、质量指标、异常样本和结论写进模板。

| 文件 | 用途 |
|---|---|
| `debug_checklist.md` | 排查 exact 为 0、numeric 高但格式错、stop 缺失、timeout、endpoint 配置错误 |
| `source_reading_card.md` | 快速回忆 patch、run_eval、SGLang eval 和 OpenAI endpoint 主路径 |
| `serving_metrics_template.md` | 记录一次 serving eval 的配置、样本、质量指标、完成率和后续动作 |

stub eval 只能证明 harness 和 artifact 路径。真实质量结论必须绑定模型、endpoint、prompt 集、生成参数和完成率。
