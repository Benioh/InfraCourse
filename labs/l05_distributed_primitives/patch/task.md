# L02 Patch · 手写 Tensor Parallel Linear

## 你要交付什么

在 `starter/tp_linear.py` 里把两个类填完整：

- `ColumnParallelLinear`：把权重沿 **输出维度** 切到 N 张卡上，forward 是 broadcast input + 各卡算各自分片，backward 是 all-reduce grad-input。
- `RowParallelLinear`：把权重沿 **输入维度** 切，forward 各卡算分片输出再 all-reduce 求和，backward 把 grad-output 直接广播给每张卡。

**禁止使用 `megatron.core.tensor_parallel`** 或 `torch.nn.parallel.DistributedDataParallel`，因为本关的目的就是让你自己用 `autograd.Function` 写一遍切分 + 通信。允许使用 `torch.distributed` 提供的 `all_reduce` / `all_gather`。

补丁规模目标：60–120 行 Python（不含空行注释）。

## 接口契约

```python
class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        gather_output: bool = True,
        process_group: dist.ProcessGroup | None = None,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features), 全卡相同
        # if gather_output: 返回 (..., out_features)，全卡相同
        # else:             返回 (..., out_features // world_size)，各卡分片
        ...
```

`RowParallelLinear` 镜像版本，新增参数 `input_is_parallel: bool`：

- `input_is_parallel=False`：输入是完整 `(..., in_features)`，需要本地切片
- `input_is_parallel=True`：输入是 `(..., in_features // world_size)`，直接用

## 不变量（写代码时心里要装着）

1. **数学正确性**：TP=N 的 forward 必须与 TP=1 在同一输入上 `torch.allclose(atol=1e-5)`。
2. **梯度正确性**：`torch.autograd.gradcheck` 在 `(in=8, out=16, world=2)` 的最小配置下必须通过（用 fp64）。
3. **通信位置**：
   - Column forward：无 collective（输入全卡相同，输出可选 all-gather）
   - Column backward：grad-input 必须 all-reduce
   - Row forward：output 必须 all-reduce
   - Row backward：grad-output 直接广播（无 collective）
4. **bias 处理**：Column 的 bias 和权重一起切；Row 的 bias 只在 rank 0 加（防止重复加 N 次）。

## 怎么验证（这就是评分）

```bash
# 起两个 rank 的 single-host 进程组，用 gloo（CPU 也行，方便调试）
make patch-test M=l05_distributed_primitives

# 等价于：
torchrun --standalone --nproc_per_node=2 -m pytest patch/tests/test_patch.py -v
```

测试分三级：

| 级别 | 测试名 | 说明 |
|---|---|---|
| 结果 | `test_column_parallel_matches_single_gpu` | TP=2 forward 输出与 nn.Linear `allclose(atol=1e-10)` |
| 结果 | `test_row_parallel_matches_single_gpu` | 同上 |
| 结果 | `test_column_grad_matches_single_gpu` | backward 后 grad_x / grad_W 都与单卡相等 |
| 结果 | `test_row_grad_matches_single_gpu` | 同上 |
| 结果 | `test_row_bias_added_once` | 输出与单卡 nn.Linear 完全相等（bias 加错位置时差 ≈ bias × world_size）|

**5 项全部是"结果对比"**——不规定你怎么写，只看输出 / 梯度对不对。爱用 `all_reduce_coalesced` 还是 `all_reduce` 自己决定，bias 写在哪一行也随意，只要数学等价就 PASS。

没有 GPU 也能跑（用 gloo backend），所以这是本课最严格、也最容易上手的 patch。

## 卡住怎么办

1. 先在 `notebooks/n04_tensor_parallel_linear.ipynb` 用矩阵切分图把 forward / backward 通信路径画一遍。
2. 30 分钟解不出，运行 `make patch-hint M=l05_distributed_primitives` 查看 `reference/tp_linear.py` 的关键骨架（不是完整答案，是关键的 5 个函数签名 + 注释）。
3. 仍然卡住，运行 `make patch-show-solution M=l05_distributed_primitives` 看完整答案——但你应该尝试**先关掉答案，自己照着写一遍**，否则等于没学。

## 写完之后你能做什么

- 解释 Megatron `ColumnParallelLinear` 源码每一行（你已经写过同样的东西）。
- 在面试里讲清"TP forward 哪里需要 collective、backward 哪里需要 collective"。
- 在 L05.5 的 MoE 里复用这两个 primitive 当 expert FFN。
- 在 Capstone Stage A 里直接用它们搭 image / audio projector 的 TP 版本。
