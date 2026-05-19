# L00 · 环境探针：把环境变成机器可读的证据

> 本关只做一件事：**实现 3 个环境探针函数**，让你能在 10 分钟内判断任何陌生 4090/H200 机器能不能跑后续 lab，并留下可复现证据。

写完这关你能立刻在面试里答清"`torch.cuda.is_available()=False` 怎么定位"，以及"为什么 LOCAL_RANK=1 跑出来的进程在 nvidia-smi 看是 GPU 3"。

## 闭环

```bash
cat labs/l01_env_conda_cuda/patch/task.md
$EDITOR labs/l01_env_conda_cuda/patch/starter/env_probe.py
make patch-test M=l01_env_conda_cuda   # 7 个测试，CPU 即可，约 2 秒
```

## 你要改的文件

```
labs/l01_env_conda_cuda/patch/
├── task.md
├── starter/env_probe.py        ★ 唯一要改的文件
├── reference/env_probe.py
└── tests/test_patch.py
```

## 测试覆盖

7 个**结果对比**测试：
- `collect_python_env()` 必须返回 6 个字段、`pythonpath` 是 list、CVD 区分 unset/空串
- `parse_local_rank_mapping()` 处理 CVD 重映射、越界报错、错误格式提示
- `summarize_drift()` 漂移描述按字母序、跨 list 字段比较

## 卡住怎么办

1. 看 `scripts/env/check_cuda.py` —— 真实项目里探针怎么写。
2. 看 `labs/l01_env_conda_cuda/scripts/torchrun_hello.py` —— RANK/LOCAL_RANK/CVD 的实战互动。
3. `make patch-show-solution M=l01_env_conda_cuda` 打开参考解。

## 配套源码研读

- `mini_infra/observability/evidence.py` —— 把环境快照写成可审计 JSON 的最小骨架。
- `scripts/env/collect_env.py` —— 真实工程级 env collector。

## 进入下一关的前置

`make patch-test M=l01_env_conda_cuda` 全绿后，继续做源码理解口试。下一关 [L01 显存账本](../l02_pytorch_systems/README.md) 让你用同样"小函数 + 单元测试"的形式精算 7B Adam 训练的显存。
