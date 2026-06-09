# L04 Source Reading Card

## 一句话主线

`ManualDDP` 在 backward 后遍历有效 `param.grad`，对每个 grad 执行 `all_reduce(SUM)`，再除以 `world_size`，让所有 rank 在 optimizer step 前持有同一份平均梯度。

## 必读源码

| 顺序 | 文件 | 行号 | 记住什么 |
|---|---|---|---|
| 1 | `labs/l03_nccl_ddp_smoke/patch/starter/manual_ddp.py` | L22-L59 | starter 只要求实现构造函数和 `synchronize_grads()` |
| 2 | `labs/l03_nccl_ddp_smoke/patch/reference/manual_ddp.py` | L11-L24 | no-op、skip、all-reduce、average |
| 3 | `labs/l03_nccl_ddp_smoke/patch/tests/conftest.py` | L33-L74 | worker 如何设置环境变量并启动 gloo group |
| 4 | `labs/l03_nccl_ddp_smoke/patch/tests/worker_cases.py` | L14-L81 | 对齐 PyTorch DDP，并抓忘记除法 |
| 5 | `labs/l03_nccl_ddp_smoke/patch/tests/worker_cases.py` | L84-L131 | no-op、frozen 参数、partial grad 边界 |
| 6 | `labs/l03_nccl_ddp_smoke/scripts/ddp_hello.py` | L24-L65 | 最小 all-reduce 和 barrier artifact |
| 7 | `labs/l03_nccl_ddp_smoke/scripts/run_smoke.py` | L28-L124 | torchrun、fallback、metrics、report |
| 8 | `mini_infra/distributed/collectives.py` | L9-L13, L14-L23 | rank/world_size snapshot、期望求和和 backend 边界 |

## 关键判断

| 问题 | 判断方式 |
|---|---|
| 是否平均 | 和单进程 baseline 对比，忘记除法会大 `world_size` 倍 |
| 是否和 PyTorch DDP 对齐 | `test_grads_match_pytorch_ddp` 比较每个参数 grad |
| 是否处理单 rank | `world_size == 1` 时同步函数不改 grad |
| 是否处理 frozen 参数 | `requires_grad=False` 参数保持 `grad is None` |
| 是否处理 partial grad | `p.grad is None` 时跳过 |
| smoke 是否真实通信 | `fallback_used=False` 且 all_reduce_sum/barrier 正常 |

## 复述模板

```text
本讲的 DDP 合同是：
每个 rank 先独立 backward，得到 local grad。
ManualDDP 遍历有效参数，对每个 grad 做 all_reduce(SUM)。
SUM 后每个 rank 得到相同的梯度和。
再除以 world_size，得到平均梯度。
optimizer.step 可以在每个 rank 本地执行，因为每个 rank 看到的 grad 一致。
```

## 常见误读

- all-reduce sum 已完成平均：错，平均需要除以 `world_size`。
- barrier 能证明梯度正确：错，barrier 只证明所有 rank 到达同步点。
- fallback smoke 能证明 DDP 通过：错，它只能说明 validation artifact 生成链路可用。
- ManualDDP 可直接用于生产大模型训练：错，它没有 bucket、hook、overlap、unused parameter 处理和容错。
