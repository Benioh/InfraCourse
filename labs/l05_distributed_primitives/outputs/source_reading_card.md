# L06 Source Reading Card

## 一句话主线

Column Parallel 切输出维，Row Parallel 切输入维；copy/reduce/gather 三个 autograd primitive 固定 forward 和 backward 的通信规则，让 TP Linear 与单卡 `nn.Linear` 数值对齐。

## 必读源码

| 顺序 | 文件 | 行号 | 记住什么 |
|---|---|---|---|
| 1 | `patch/starter/tp_linear.py` | L30-L80 | copy、reduce、gather 三个 primitive 的 TODO |
| 2 | `patch/starter/tp_linear.py` | L88-L142 | Column 参数分片、本地 linear、可选 gather |
| 3 | `patch/starter/tp_linear.py` | L145-L213 | Row 输入分片、output reduce、bias 陷阱 |
| 4 | `patch/reference/tp_linear.py` | L20-L67 | 三个 primitive 的最小正确实现 |
| 5 | `patch/reference/tp_linear.py` | L96-L113 | Column reference |
| 6 | `patch/reference/tp_linear.py` | L143-L171 | Row reference |
| 7 | `patch/tests/worker_cases.py` | L20-L161 | forward/backward 与单卡对齐 |
| 8 | `patch/tests/worker_cases.py` | L235-L270 | Row bias 只加一次 |
| 9 | `github_repo/Megatron-LM/.../mappings.py` | L197-L273 | Megatron copy/reduce/gather autograd Function |

## 关键判断

| 问题 | 判断方式 |
|---|---|
| Column 切分是否正确 | 本地 weight 是 `(out/ws, in)`，输出片段沿最后一维 gather |
| Row 切分是否正确 | 本地 weight 是 `(out, in/ws)`，partial output all-reduce |
| Column backward 是否正确 | `_CopyToParallelRegion.backward` reduce `grad_x` |
| Gather backward 是否正确 | 上游 grad 沿最后一维 split，当前 rank 取自己的片段 |
| Row bias 是否正确 | reduce 后加一次，不能传给本地 `F.linear` |

## 复述模板

```text
TP 切的是一个 Linear 内部的矩阵；切 batch 并同步梯度属于 DP。
Column 切 out_features，每个 rank 产生不同输出片段。
Row 切 in_features，每个 rank 产生同一输出的一部分贡献。
Column backward 对输入梯度求和。
Row forward 对 partial output 求和。
Row bias 在 reduce 后加一次。
```

## 常见误读

- PyTorch `weight` 是 `(out, in)`，不是 `(in, out)`。
- all-gather 的 backward 是 split，不是再次 all-gather。
- CPU/gloo 语义测试不能写成 GPU/NCCL 性能结论。
- TP checkpoint 分片不能在改变 TP size 后直接加载。
