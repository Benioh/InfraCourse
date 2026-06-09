# Patch · L06 Tensor Parallel Linear

**这个目录就是本关的全部任务。** 读 `task.md` → 改 `starter/tp_linear.py` → 跑测试。

## 一句话任务

不用 Megatron，用 `torch.distributed` 的 `all_reduce` / `all_gather` + 自己写 `autograd.Function`，把 `ColumnParallelLinear` 和 `RowParallelLinear` 写出来，并通过全部 7 个测试。

## 命令

```bash
# 运行全部测试（2-rank gloo，CPU 即可）
make patch-test M=l05_distributed_primitives

# 看接口骨架提示（不是答案）
make patch-hint M=l05_distributed_primitives

# 实在卡住，打开参考解
make patch-show-solution M=l05_distributed_primitives

# 把测试跑在 reference 实现上（验证测试本身正确）
TP_IMPL=reference make patch-test M=l05_distributed_primitives
```

## 文件结构

```
patch/
├── task.md                  # 任务详细说明（必读）
├── starter/                 # ★ 你要改的代码
│   ├── __init__.py
│   └── tp_linear.py
├── reference/               # 参考解，卡住再看
│   └── tp_linear.py
├── tests/                   # 自动验证（不要改）
│   ├── conftest.py
│   ├── worker_cases.py
│   └── test_patch.py
└── README.md                # 本文件
```

## 通过的标准

7 个 pytest 全部通过。无需写报告，PASS 即过。
