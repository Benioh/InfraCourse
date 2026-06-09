# L10 课后产物说明

本目录保存 L10 的长上下文复盘材料。它们用于排查 attention 分块、CP ring、RoPE/YaRN 和长上下文评估证据。

| 文件 | 用途 |
|---|---|
| [debug_checklist.md](debug_checklist.md) | 排查长上下文 OOM、online softmax 数值偏差、CP shape mismatch 和通信证据 |
| [source_reading_card.md](source_reading_card.md) | 快速复习 patch、MiniInfra、Megatron CP 和 TE attention 主路径 |
| [long_context_template.md](long_context_template.md) | 记录序列长度、shape、显存估算、通信字节、YaRN 参数和验证结论 |

建议在完成 patch 后，至少记录一次 `run_seqlen.py --seq 16384 --cp 4` 的输出，并用模板说明它只是一份教学估算，不能替代真实 GPU profile。
