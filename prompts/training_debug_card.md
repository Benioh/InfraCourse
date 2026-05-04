# Training Debug Card

适用范围：L01-L06 的 PyTorch、TorchTitan、Megatron 训练问题，尤其是 OOM、NaN、checkpoint resume、吞吐/MFU 异常。

## Observe

请先收集并贴出：

- `command.sh`、`config.resolved.yaml`、`metrics.jsonl` 中最近 20 行。
- 主日志中第一个 error 前后 80 行，而不是最后一个堆栈。
- GPU/CPU/数据路径/checkpoint 路径，以及是否 validation-only。
- 最近一次只改变了哪个变量。

## Ask

向 AI 提问时使用：

```text
你是训练 infra reviewer。请只基于我给出的 command/config/log/metrics 判断最可能的 3 个根因。
不要重写训练框架；先列最小检查，每次只改一个变量。
重点排查 OOM/NaN/checkpoint resume/数据加载/并行配置边界。
```

## Patch Plan

要求 AI 输出：

- 先验判断：属于数据、模型、优化器、并行、checkpoint、环境中的哪一类。
- 最小复现：最小 batch、seq_len、step、rank 数。
- 最小 patch：只改配置、日志、断言或 parser；不做大重构。
- 回滚方案：patch 失败后如何恢复 baseline。

## Human Check

人工必须确认：

- 是否隐藏了 `validation-only` 边界。
- 是否删除断言、吞异常或伪造 metrics。
- 是否破坏 checkpoint、tokenizer、TP/PP/DP 兼容性。
- 是否同时改了多个变量。

## Apply

只允许先做这些小改动：

- 增加配置打印、shape/NaN/inf 断言、metrics 字段。
- 降低 batch/seq_len 或关闭混精做对照。
- 修正 checkpoint/tokenizer/data prefix 路径。
- 加一个能复现 ticket 的 smoke。

## Test

验证顺序：

1. 最小 CPU/单卡 smoke。
2. 原硬件小 step 验证。
3. 对照实验只改一个变量。
4. `make grade M=<mission>` 与 `make self-check M=<mission>`。

## Explain

报告里写清：

- 根因证据来自哪个日志或 metrics 字段。
- 修复前后只改变了哪个变量。
- 仍未验证的 H200/多卡/长训边界。

## Commit

提交前检查：

- 不提交临时 checkpoint、大日志或私有数据。
- 保留 `command.sh`、`config.resolved.yaml`、`metrics.jsonl` 证据链。
- 在报告中写明 AI 参与了哪些推理，哪些由你人工验证。
