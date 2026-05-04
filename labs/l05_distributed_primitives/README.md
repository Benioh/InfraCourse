# L02 · 分布式原语：手写 Tensor Parallel Linear

> 本关的目标只有一个：**用 `torch.distributed` + `autograd.Function` 自己实现 `ColumnParallelLinear` 和 `RowParallelLinear`，并通过 7 个测试。**

写完之后你能解释 Megatron 的 `tensor_parallel/layers.py` 每一行，并在 L05.5 (MoE) 和 Capstone (多模态 projector) 里复用。

## 闭环（学完只需做这一件事）

```bash
# 1. 读任务说明
cat labs/l05_distributed_primitives/patch/task.md

# 2. 改 patch/starter/tp_linear.py（不许改其它文件）

# 3. 跑测试（CPU gloo, world=2，不需要 GPU）
make patch-test M=l05_distributed_primitives

# 4. 7 个 pytest 全过 → 本关 PASS。
```

pytest 全绿代表 TP patch 的代码契约通过；之后还要对照 `source_reading` 和 AI 框架理解口试，确认能把它放回 Megatron TP/PP 主线。

## 你要改的文件

```
labs/l05_distributed_primitives/patch/
├── task.md                       # 任务详细说明（必读）
├── starter/tp_linear.py          # ★ 唯一要改的文件
├── reference/tp_linear.py        # 参考解（卡住再看）
└── tests/                        # 自动测试（不要改）
```

## 测试覆盖

| 类别 | 测试 | 通过条件 |
|---|---|---|
| 结果 | `test_column_parallel_matches_single_gpu` | TP=2 forward 输出 `allclose(atol=1e-10)` 单卡 nn.Linear |
| 结果 | `test_row_parallel_matches_single_gpu` | 同上 |
| 结果 | `test_column_grad_matches_single_gpu` | backward 之后 grad_x / grad_W 都与单卡相等 |
| 结果 | `test_row_grad_matches_single_gpu` | 同上 |
| 结果 | `test_row_bias_added_once` | 输出等于单卡 nn.Linear（bias 加错位置时差 ≈ bias × ws）|

5 个测试**全部按结果判定**——不规定你的实现方式，输出 / 梯度对就过。`make patch-test M=l05_distributed_primitives` 全部约 15–20 秒。

## 卡住怎么办

1. 先打开 `notebooks/n04_tensor_parallel_linear.ipynb` 把矩阵切分图画一遍。
2. `make patch-hint M=l05_distributed_primitives` —— 看 TODO 列表 + 通信路径速查表。
3. 还卡住，`make patch-show-solution M=l05_distributed_primitives` —— 打开完整参考解。

## 配套源码研读（可选，30min）

写完 patch 之后，对照：

- `github_repo/Megatron-LM/megatron/core/tensor_parallel/layers.py` —— 看 Megatron 的工程级实现（async tensor parallelism、weight 初始化与 ckpt 切片、sequence parallel）有哪些边界你没考虑。
- `notebooks/n03_ddp_collectives.ipynb` —— 顺手把 DDP 的 collective 顺序也理清。

读完你会发现：你已经掌握了 Megatron TP 的核心 90%，剩下 10% 是工程细节。

## Debug Tickets（任选一个，可选）

只有真的对死锁/形状错配感兴趣再做：

- `tickets/dist_wrong_world_size_001.yaml`
- `tickets/dist_rank_hang_002.yaml`
- `tickets/dist_device_mismatch_003.yaml`

## 进入下一关的前置

`make patch-test M=l05_distributed_primitives` 全绿后，继续用 AI 框架理解口试检查源码主线。下一关 [L03 TorchTitan](../l06_torchtitan_training/README.md) 会让你给 TorchTitan 加一个 selective activation checkpoint policy。
