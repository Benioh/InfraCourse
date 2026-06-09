# L06 Tensor Parallel 复盘模板

## 1. Run 信息

- 日期：
- git commit：
- 命令：
- Python / PyTorch：
- backend：
- world size：
- 是否 GPU/NCCL：
- 是否 CPU/gloo：

## 2. 本次验证目标

| 项 | 内容 |
|---|---|
| 目标 |  |
| 比较对象 | 单卡 `nn.Linear` / toy / smoke |
| 成功标准 |  |
| 已知边界 |  |

## 3. 测试结果

| case | 结果 | 关键证据 |
|---|---|---|
| `test_column_parallel_matches_single_gpu` |  |  |
| `test_row_parallel_matches_single_gpu` |  |  |
| `test_column_grad_matches_single_gpu` |  |  |
| `test_row_grad_matches_single_gpu` |  |  |
| `test_row_bias_added_once` |  |  |

## 4. 失败定位

| 现象 | 可能位置 | 证据 |
|---|---|---|
| Column forward diff | 输出维切片 / gather |  |
| Column backward diff | copy backward reduce / gather backward split |  |
| Row forward diff | 输入维切片 / output reduce |  |
| Row bias diff | bias 加在 reduce 前 |  |
| hang | collective 次数或顺序不一致 |  |

## 5. 源码对应

| 机制 | 源码位置 | 判断 |
|---|---|---|
| copy primitive | `patch/reference/tp_linear.py` L20-L31 |  |
| reduce primitive | `patch/reference/tp_linear.py` L34-L45 |  |
| gather primitive | `patch/reference/tp_linear.py` L48-L67 |  |
| Column forward | `patch/reference/tp_linear.py` L108-L113 |  |
| Row forward | `patch/reference/tp_linear.py` L160-L171 |  |

## 6. 结论

- 本次能证明：
- 本次不能证明：
- 如果失败，最小复现命令：
- 下一步要改的代码或实验：
