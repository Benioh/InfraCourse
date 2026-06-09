# L06 Tensor Parallel Debug Checklist

这份清单用于排查 `ColumnParallelLinear`、`RowParallelLinear` 和 2-rank gloo patch-test。先定位失败类型，再改代码。

## 1. 初始化和进程组

- `WORLD_SIZE` 是否等于实际 worker 数。
- 每个 worker 是否设置 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE`。
- 是否所有 rank 都初始化同一个 backend。
- 是否有某个 rank 提前异常退出，导致另一个 rank hang。

对应源码：

- `patch/tests/conftest.py` L54-L65
- `patch/tests/conftest.py` L93-L108

## 2. Column Forward Diff

优先检查：

- `weight` shape 是否为 `(out_features / world_size, in_features)`。
- 本地 bias 是否为 `(out_features / world_size,)`。
- canonical weight 是否按输出维切片。
- `gather_output=True` 时是否 all-gather 后沿最后一维 concat。
- 是否错误地 all-reduce 了输出片段。

对应源码：

- `patch/reference/tp_linear.py` L96-L113
- `patch/tests/worker_cases.py` L20-L49

## 3. Column Backward Diff

优先检查：

- `_CopyToParallelRegion.forward` 是否 identity。
- `_CopyToParallelRegion.backward` 是否 all-reduce `grad_output`。
- gather 的 backward 是否 split 回本 rank。
- `layer.weight.grad` 是否只比较 canonical 的本地输出切片。

对应源码：

- `patch/reference/tp_linear.py` L20-L31
- `patch/reference/tp_linear.py` L48-L67
- `patch/tests/worker_cases.py` L95-L125

## 4. Row Forward Diff

优先检查：

- `weight` shape 是否为 `(out_features, in_features / world_size)`。
- `input_is_parallel=False` 时是否沿最后一维 chunk 输入。
- 本地 `F.linear` 是否没有传入 bias。
- partial output 是否 all-reduce SUM。
- bias 是否在 reduce 后加一次。

对应源码：

- `patch/reference/tp_linear.py` L143-L171
- `patch/tests/worker_cases.py` L52-L79
- `patch/tests/worker_cases.py` L235-L270

## 5. Collective Hang

快速判断：

```text
所有 rank 的 collective 次数必须一致。
每一次 collective 的 tensor shape/dtype 必须一致。
所有 rank 必须按同一顺序进入 collective。
```

排查动作：

- 在 collective 前后打印 rank、函数名、shape、dtype。
- 先只跑第一个失败 pytest case。
- 检查某个 rank 是否因 shape assert 提前退出。
- 检查 `input_is_parallel` 分支是否所有 rank 一致。

## 6. Checkpoint 和性能边界

- TP=2 的本地 weight shape 不能直接用于 TP=4。
- CPU/gloo patch-test 不代表 GPU/NCCL throughput。
- 讨论性能时必须记录 world size、backend、shape、dtype、硬件、计时方法和误差。
- toy 和 smoke 不能替代 patch-test。
