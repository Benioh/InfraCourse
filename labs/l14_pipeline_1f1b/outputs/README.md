# L15 课后产物：Pipeline Parallel 1F1B

本目录存放 L15 之后可以继续复用的排查材料。它们用于记录 PP schedule 的配置、现象、源码定位和判断，适合在 CPU smoke、Megatron dryrun 或真实训练 benchmark 后填写。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序排查 warmup、steady、cooldown、bubble ratio 和 P2P 等待 |
| `source_reading_card.md` | 快速回忆 patch、MiniInfra 和 Megatron non-interleaved schedule 主路径 |
| `training_step_template.md` | 记录一次 PP 调度实验的配置、指标、源码对应和结论 |

建议每次跑完 `run_pp_smoke.sh` 或 Megatron PP 训练命令后，至少保存命令、配置、stage 数、microbatch 数、bubble ratio、总 op 数和关键 artifact 路径。没有这些证据时，不要把一次局部输出当作 PP 配置已经合格。
