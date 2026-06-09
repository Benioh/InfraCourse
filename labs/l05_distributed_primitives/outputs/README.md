# L06 课后产物使用说明

本目录保存 Tensor Parallel Linear 的复用材料。它们用于复盘一次 TP patch、collective hang 或 Row bias 数值问题。

## 文件说明

| 文件 | 用途 |
|---|---|
| `debug_checklist.md` | 排查初始化失败、collective hang、shape 错、grad 不等价和 Row bias 重复加 |
| `source_reading_card.md` | 复习 Column/Row、autograd primitive、tests 和 Megatron mapping 主路径 |
| `training_step_template.md` | 记录一次 TP Linear patch-test 或 smoke 复盘 |

## 建议使用顺序

1. 先用 `source_reading_card.md` 回忆切分维度和通信路径。
2. 运行 `make patch-test M=l05_distributed_primitives`。
3. 如果失败，按 `debug_checklist.md` 从初始化、hang、forward diff、backward diff、bias diff 分流。
4. 需要观察 artifact 链路时运行 `python labs/l05_distributed_primitives/scripts/run_lab.py --mode smoke`。
5. 用 `training_step_template.md` 写下测试结果、失败证据和下一步动作。

## 证据分级

| 证据 | 能说明什么 | 不能说明什么 |
|---|---|---|
| patch-test pass | Column/Row forward 与 backward 语义对齐单卡 | GPU/NCCL 性能、sequence parallel、checkpoint 转换 |
| toy_column max_error 接近 0 | 单进程 Column forward 切分数学正确 | Row、backward、autograd primitive 正确 |
| collectives_demo | all-reduce/all-gather/broadcast 字段可理解 | TP Linear patch 通过 |
| run_lab smoke | artifact 生成链路可用 | starter 实现正确 |
