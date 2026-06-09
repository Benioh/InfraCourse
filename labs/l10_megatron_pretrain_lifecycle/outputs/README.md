# L11 课后产物说明

本目录保存 L11 的训练生命周期复盘材料。

| 文件 | 用途 |
|---|---|
| [debug_checklist.md](debug_checklist.md) | 排查 train_step 调用顺序、loss、optimizer skip、scheduler、metrics 和 checkpoint |
| [source_reading_card.md](source_reading_card.md) | 快速复习 patch、MiniInfra lifecycle、checkpoint 和真实 Megatron 对照 |
| [training_step_template.md](training_step_template.md) | 记录一次 lifecycle run 的配置、指标、artifact 和结论 |

建议在完成 patch 后运行一次 `run_lifecycle.py --config configs/cpu_smoke.yaml`，再用模板记录 `metrics.jsonl` 和 `acceptance.json` 的关键字段。
