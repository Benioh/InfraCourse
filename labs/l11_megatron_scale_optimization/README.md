# L12 · Megatron Scale Optimization：Bucketed Manual DDP

这一讲从 L11 的 `train_step` 出发，进入训练扩展时最先暴露的通信问题：反向传播结束后，每个 data parallel rank 都有一份本地梯度，必须在 optimizer step 前求平均。如果每个参数单独发起一次 all-reduce，小参数会把 collective 启动开销放大到 step time 里。

本关 patch 实现 `BucketedManualDDP`：按字节数把 trainable params 分桶，把同一桶里的 grad 拼成连续 buffer，一次 all-reduce 后再拆回原 shape。它保留 data parallel 的数学语义，只改变梯度通信的组织方式。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L12 在 Megatron scale optimization 主线中的位置。
2. 读 [lecture.md](lecture.md)：理解 bucket、flatten buffer、all-reduce 平均、Megatron buffer 和扩展评审边界。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、测试、Megatron DDP、ParamAndGradBuffer 和 TorchTitan mesh 读源码。
4. 做 quiz：确认 bucket size、overlap、distributed optimizer 和 checkpoint 切片这些易混点。
5. 做 patch：实现 `BucketedManualDDP` 并通过 5 个 CPU/gloo 测试。
6. 跑 drill：用 `run_scale_stub.py` 产出 `scale_table.json`、`metrics.jsonl` 和 report。
7. 填写 [outputs/training_step_template.md](outputs/training_step_template.md)，沉淀一次扩展方案复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / Megatron scale optimization |
| 它承接什么 | L11 的训练 step 闭环和 metrics 证据 |
| 它解决什么问题 | 用 bucketed grad sync 降低 data parallel 梯度同步的 collective 启动开销 |
| 它连接哪些指标或证据 | bucket count、world size、grad diff、peak memory、tokens/sec、MFU、checkpoint layout |
| 它连接哪些源码 | patch `BucketedManualDDP`、PyTorch DDP 测试参照、Megatron DDP、ParamAndGradBuffer、TorchTitan `ParallelDims` |
| lab 检验什么 | bucket 构造、flatten/unflatten、all-reduce 平均、world_size=1、超大参数、frozen 参数 |

## Patch 闭环

```bash
cat labs/l11_megatron_scale_optimization/patch/task.md
$EDITOR labs/l11_megatron_scale_optimization/patch/starter/bucketed_ddp.py
make patch-test M=l11_megatron_scale_optimization
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_grads_match_pytorch_ddp` | bucketed grad 与 PyTorch DDP 对齐 |
| `test_works_with_huge_param` | 单个参数超过 bucket size 时仍能同步 |
| `test_works_with_tiny_params` | 很多小参数合到大 bucket 后数值不变 |
| `test_world_size_1_is_noop` | 单 rank 路径不改 grad |
| `test_skips_no_grad_params` | frozen 参数不进入同步路径 |

## Drill 闭环

```bash
python labs/l11_megatron_scale_optimization/scripts/run_scale_stub.py \
  --config configs/4090_debug.yaml --run-id l12_validation
```

drill 会比较几个 TP/PP/recompute 场景，写出 `artifacts/scale_table.json`、`metrics.jsonl`、`config.resolved.yaml` 和 `report.md`。这些数字是估算，用来练习“显存、吞吐、MFU 和 checkpoint 语义一起看”的评审格式。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 bucket 同步、低 MFU、并行配置和 checkpoint 切片问题 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 BucketedManualDDP、Megatron DDP buffer 和 TorchTitan mesh 主路径 |
| [outputs/training_step_template.md](outputs/training_step_template.md) | 记录一次 scale optimization drill 的配置、指标、证据和判断 |

## 进入下一讲

`make patch-test M=l11_megatron_scale_optimization` 通过，并完成一次 scale drill 复盘后，进入 [L13 FSDP2](../l12_fsdp2_llama/README.md)。
