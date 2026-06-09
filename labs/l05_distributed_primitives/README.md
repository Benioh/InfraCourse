# L06 · 分布式原语：手写 Tensor Parallel Linear

这一讲解决层内并行的第一个核心问题：一个 `nn.Linear` 的矩阵太大时，怎样把权重切到多个 rank 上，同时保持 forward 输出、输入梯度和权重梯度与单卡 `nn.Linear` 等价。你会手写 `ColumnParallelLinear`、`RowParallelLinear` 和三个 autograd 通信 primitive。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L06 如何把 L04 的 collective 和 L05 的单算子直觉合成 Tensor Parallel Linear。
2. 读 [lecture.md](lecture.md)：理解 TP 与 DP 的边界、Column/Row 切分、bias 位置和 backward 通信。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 starter、reference、tests、MiniInfra 和 Megatron 映射读源码。
4. 跑 notebook：[n03_ddp_collectives.ipynb](../../notebooks/n03_ddp_collectives.ipynb) 和 [n04_tensor_parallel_linear.ipynb](../../notebooks/n04_tensor_parallel_linear.ipynb)。
5. 做 quiz：确认切分维度、collective 位置、bias 和 checkpoint 边界。
6. 做 patch：实现 `patch/starter/tp_linear.py`。
7. 跑 lab smoke：生成 collective、DDP、TP toy、pipeline toy 的 artifact。
8. 填写 [outputs/training_step_template.md](outputs/training_step_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Distributed training / tensor parallel |
| 它解决什么问题 | 把一个 Linear 的 weight 和 matmul 切到多个 rank 上，并保持单卡数值语义 |
| 它连接哪些指标 | world_size、rank、all_reduce/all_gather、forward max diff、grad_x diff、grad_W diff、bias 是否只加一次 |
| 它连接哪些源码 | `patch/reference/tp_linear.py`、`patch/tests/worker_cases.py`、`mini_infra/megatron/core/tensor_parallel/layers.py`、Megatron `mappings.py` 和 `layers.py` |
| lab 检验什么 | Column/Row forward 对齐单卡；Column/Row backward 对齐单卡；Row bias 只加一次 |

## 你会学到什么

- `nn.Linear` 的 weight shape 为什么是 `(out_features, in_features)`。
- Column Parallel 为什么切输出维度，Row Parallel 为什么切输入维度。
- `_CopyToParallelRegion`、`_ReduceFromParallelRegion`、`_GatherAlongLastDim` 的 forward/backward 规则。
- Row bias 为什么必须在 all-reduce 之后加一次。
- CPU/gloo patch-test 能证明什么，不能证明 GPU/NCCL 性能什么。
- TP checkpoint 改 world size 时为什么需要 gather 后重新切分。

## Patch 闭环

```bash
cat labs/l05_distributed_primitives/patch/task.md
$EDITOR labs/l05_distributed_primitives/patch/starter/tp_linear.py
make patch-test M=l05_distributed_primitives
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_column_parallel_matches_single_gpu` | Column forward 输出与单卡 `nn.Linear` 对齐 |
| `test_row_parallel_matches_single_gpu` | Row forward 输出与单卡 `nn.Linear` 对齐 |
| `test_column_grad_matches_single_gpu` | Column 的 `grad_x` 和本地 `grad_W` 切片对齐 |
| `test_row_grad_matches_single_gpu` | Row 的输入梯度切片和本地 `grad_W` 对齐 |
| `test_row_bias_added_once` | Row bias 在 reduce 后只加一次 |

patch-test 默认使用 CPU/gloo 和 2 个 worker，不需要 GPU。它验证语义，不验证 NCCL throughput。

## Lab Smoke

```bash
python labs/l05_distributed_primitives/scripts/run_lab.py --mode smoke
```

smoke 会运行 collective demo、DDP toy train、Column TP toy 和 pipeline toy，并写出 `metrics.jsonl`、`train.log` 和 `report.md`。这些 toy 帮助建立系统位置，不能替代 patch-test 对 `ColumnParallelLinear` 和 `RowParallelLinear` 的验收。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 collective hang、shape 错、grad 不等价、Row bias 重复加 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 TP Linear 和 Megatron 映射主路径 |
| [outputs/training_step_template.md](outputs/training_step_template.md) | 记录一次 TP Linear patch 或 smoke 复盘 |

## 进入下一讲

`make patch-test M=l05_distributed_primitives` 通过，并能解释 Column/Row 的 forward/backward 通信位置后，进入 [L07 TorchTitan 训练入口](../l06_torchtitan_training/README.md)。
