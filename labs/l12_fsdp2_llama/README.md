# L05.3 · FSDP2：把 Llama 1B 切成 sharded params

> 本关只做一件事：**手写一个 `wrap_transformer_blocks_fsdp2(model, block_cls, ...)`**——
> 用 `torch.distributed._composable.fsdp.fully_shard` 把每个 transformer block 单独
> wrap，再 wrap root，并支持 `MixedPrecisionPolicy` 与 `reshard_after_forward`。

L01.5 / L05 让你手写了 DDP 和 bucketed DDP——那是 70% 工业代码 5 年前的形态。
**今天的工业默认是 FSDP2**：参数 + 梯度 + optimizer state 全部分片，靠 NCCL 在前向/反向
按需 unshard / reshard。L05.3 让你第一次把 Llama-style 模型切碎并跑通。

## 闭环

```bash
cat labs/l12_fsdp2_llama/patch/task.md
$EDITOR labs/l12_fsdp2_llama/patch/starter/fsdp2_wrap.py
make patch-test M=l12_fsdp2_llama          # 5 个 CPU 测试 + 2 个 GPU 测试

# 真实 8×H200 跑 Llama-2 1B
RUN_GPU_TESTS=1 make patch-test M=l12_fsdp2_llama
bash labs/l12_fsdp2_llama/scripts/run_fsdp2_smoke.sh
```

## 测试覆盖

| 测试 | 标记 | 验证 |
|---|---|---|
| `test_wrap_marks_each_block` | cpu | 每个 transformer block 都被 `fully_shard` 包过 |
| `test_wrap_marks_root_last` | cpu | root wrap 必须最后调用，否则 reshard 顺序错 |
| `test_mp_policy_propagates` | cpu | `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` 正确传给每层 |
| `test_skip_blocks_filter` | cpu | `skip(name)` callback 可以排除指定 block |
| `test_reshard_after_forward_default_true` | cpu | 默认 reshard_after_forward=True |
| `test_forward_backward_smoke` | gpu | 8×H200 上 Llama-style 1B 模型前向 + 反向，sharded grad 形状校验 |
| `test_state_dict_round_trip` | gpu | save → load 后 `state_dict_type=SHARDED_STATE_DICT` 数值一致 |

## Drill

`scripts/run_fsdp2_smoke.sh` 在 8×H200 上跑 Llama-2 1B 单 rank-group 100 步训练，
acceptance：
- step100 loss 下降 ≥ 0.5
- peak memory < 35 GB / GPU
- forward + backward 时间稳定（无连续 OOM 重试）

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_dryrun.yaml` | CPU 上验证 wrap 逻辑（不真分片，仅占位） |
| `configs/h200_llama1b.yaml` | 8×H200 真实 Llama-2 1B |
| `configs/h200_llama7b.yaml` | 8×H200 Llama-2 7B（默认 mp policy bf16） |

## 调试工单

见 `tickets/INDEX.md`。建议至少做 `fsdp2_uneven_shard` 与 `fsdp2_mp_policy_dtype_mismatch`。

## 为什么这关重要

FSDP2 是 modern PyTorch（torch ≥ 2.4）训练 1B+ 模型的默认形态。
不会 FSDP2 ≈ 不能上 1B。

## 进入下一关

通过后进入 [L05.5 MoE / EP](../l13_moe_ep/README.md)。
