# L07 课后产物使用说明

本目录保存 selective activation checkpoint 的复用材料。它们用于复盘一次 policy 实验、patch-test 或 TorchTitan stub。

## 文件说明

| 文件 | 用途 |
|---|---|
| `debug_checklist.md` | 排查输出/梯度不等价、policy 计数错误、GPU memory skip、RNG 和 wrapper 副作用 |
| `source_reading_card.md` | 复习 wrapper、tests、TorchTitan `apply_ac` 和 parallelize 主路径 |
| `training_step_template.md` | 记录一次 activation checkpoint 或 TorchTitan stub 复盘 |

## 建议使用顺序

1. 先用 `source_reading_card.md` 回忆源码主路径。
2. 运行 `make patch-test M=l06_torchtitan_training`。
3. 如果 GPU memory test skip，把它写成未验证，不写成显存收益已证明。
4. 需要框架 artifact 时运行 `python labs/l06_torchtitan_training/scripts/run_torchtitan_stub.py --config configs/4090_debug.toml --mode smoke`。
5. 用 `training_step_template.md` 记录输出差异、梯度差异、wrapped child 数量、peak memory 和 step time。

## 证据分级

| 证据 | 能说明什么 | 不能说明什么 |
|---|---|---|
| CPU patch tests pass | wrapper policy、输出、梯度和 no-op 语义成立 | CUDA peak memory 下降 |
| GPU memory test pass | 测试模型上 checkpoint peak memory 更低 | 大模型真实收益、step time 可接受 |
| TorchTitan stub | 训练框架 artifact、checkpoint save/load 边界可用 | 真实 TorchTitan 集群训练通过 |
| notebook | 帮助解释 recompute 机制 | starter patch 已通过 |
