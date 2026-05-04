# L01 · PyTorch 系统：显存账本

> 本关只做一件事：**实现 3 个显存计数函数**，让你能精确拆分任意模型的 params / grads / optimizer states。

写完这关你能立刻在面试里答清"7B 模型用 Adam 训练需要多少显存"。

## 闭环

```bash
cat labs/l02_pytorch_systems/patch/task.md
$EDITOR labs/l02_pytorch_systems/patch/starter/memory_probe.py
make patch-test M=l02_pytorch_systems   # 7 个测试，CPU 运行约 5 秒
```

## 你要改的文件

```
labs/l02_pytorch_systems/patch/
├── task.md
├── starter/memory_probe.py    ★ 唯一要改的文件
├── reference/memory_probe.py
└── tests/test_patch.py
```

## 测试覆盖

7 个**结果对比**测试：
- 参数字节数与手算公式相等（普通 + 混合 dtype）
- backward 后 grad_bytes == param_bytes
- grad=None 时返回 0
- SGD（无动量）= 0、SGD+momentum ≈ 1×、Adam ≈ 2×

## 卡住怎么办

1. 看 `notebooks/n01_gpu_memory_anatomy.ipynb`。
2. `make patch-hint M=l02_pytorch_systems`。
3. `make patch-show-solution M=l02_pytorch_systems`。

## 进入下一关的前置

`make patch-test M=l02_pytorch_systems` 全绿后，继续做源码理解口试。下一关 [L01.5 NCCL/DDP](../l03_nccl_ddp_smoke/README.md) 让你不用 PyTorch DDP，自己写 bucketed grad all-reduce。
