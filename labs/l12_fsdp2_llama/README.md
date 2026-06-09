# L13 · FSDP2：把 Llama Block 变成可分片训练单元

这一讲解决训练扩展中的参数分片问题：当模型、梯度和 optimizer state 放不进单卡，Data Parallel 只复制模型会把显存压力放大。FSDP2 用 composable `fully_shard` 把模块参数按 data parallel rank 分片，在 forward/backward 需要完整参数时再 all-gather，用计算前后的 reshard 换显存。

本关 patch 很小，只实现 `wrap_transformer_blocks_fsdp2(model, block_cls, ...)`：先 wrap 每个 transformer block，再 wrap root，并把 mixed precision 和 `reshard_after_forward` 传给每一次 `fully_shard`。讲授重点放在 FSDP2 的系统位置、wrap 粒度、reshard 代价、TorchTitan 源码主路径和 checkpoint 证据。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L13 在训练系统、并行与数据主线中的位置。
2. 读 [lecture.md](lecture.md)：理解 FSDP2 解决的显存问题、wrap 粒度、reshard、mixed precision 和 checkpoint 证据。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、测试、smoke、TorchTitan `parallelize.py` 和 checkpoint 主路径读源码。
4. 做 quiz：确认 FSDP2/FSDP1、wrap 顺序、reshard、mixed precision 和 checkpoint 的边界。
5. 做 patch：实现最小 wrap 合同并通过测试。
6. 跑 drill：CPU dryrun 验证 wrap 产物；有 GPU 时再跑 train smoke。
7. 填写 [outputs/fsdp2_wrap_template.md](outputs/fsdp2_wrap_template.md)，沉淀一次 FSDP2 wrap 和训练复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / sharded data parallel |
| 它承接什么 | L12 的 data parallel 通信与 scale optimization 视角 |
| 它解决什么问题 | 用 FSDP2 把参数、梯度和 optimizer state 分片，降低单卡训练显存压力 |
| 它连接哪些指标或证据 | wrapped blocks、root wrap 顺序、reshard policy、mixed precision、loss、peak memory、checkpoint state dict |
| 它连接哪些源码 | patch `fsdp2_wrap.py`、TorchTitan `parallelize.py`、`distributed/fsdp.py`、checkpoint manager、smoke 脚本 |
| lab 检验什么 | block-first/root-last wrap、policy 透传、skip 过滤、`reshard_after_forward` 默认值和基础 smoke |

## 你会学到什么

- 解释 FSDP2 为什么能降低 Data Parallel 的显存重复。
- 区分 FSDP2、FSDP1、DDP 和 ZeRO 类优化的状态边界。
- 说明 wrap 粒度如何影响 all-gather、reshard、显存峰值和通信次数。
- 读懂 TorchTitan 如何先处理 TP/CP/AC/compile，再对 Llama block 和 root 应用 FSDP2。
- 用 patch-test 和 smoke artifact 证明 wrap 逻辑、配置和训练证据是否自洽。

## Patch 闭环

```bash
cat labs/l12_fsdp2_llama/patch/task.md
$EDITOR labs/l12_fsdp2_llama/patch/starter/fsdp2_wrap.py
make patch-test M=l12_fsdp2_llama
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_wrap_marks_each_block` | 每个 transformer block 都被 wrap |
| `test_wrap_marks_root_last` | root wrap 必须最后发生 |
| `test_mp_policy_propagates` | `MixedPrecisionPolicy` 原样传给每次 `fully_shard` |
| `test_skip_blocks_filter` | `skip(name, module)` 可以排除指定 block |
| `test_reshard_after_forward_default_true` | 默认 `reshard_after_forward=True` |
| `test_forward_backward_smoke` | CUDA 可用时跑 FSDP2 前向/反向 smoke |
| `test_state_dict_round_trip` | CUDA 可用时验证 state dict 往返 |

## Drill 闭环

CPU dryrun：

```bash
IMPL=reference bash labs/l12_fsdp2_llama/scripts/run_fsdp2_smoke.sh l13_validation
```

有 GPU 时可用 profile 覆盖：

```bash
IMPL=reference PROFILE=h200_llama1b bash labs/l12_fsdp2_llama/scripts/run_fsdp2_smoke.sh l13_h200_smoke
```

drill 会写出 `artifacts/wrap_report.json`、`metrics.jsonl`、`config.resolved.yaml` 和 `report.md`。CPU dryrun 只证明 wrap 逻辑和 artifact 格式；真实显存、loss 和吞吐结论必须来自 GPU train smoke 或真实 TorchTitan 日志。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 FSDP2 wrap、reshard、mixed precision、OOM 和 checkpoint 问题 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、TorchTitan parallelize、FSDP helper 和 checkpoint 主路径 |
| [outputs/fsdp2_wrap_template.md](outputs/fsdp2_wrap_template.md) | 记录一次 FSDP2 wrap / smoke / benchmark 的配置、指标和判断 |

## 进入下一讲

`make patch-test M=l12_fsdp2_llama` 通过，并完成一次 FSDP2 dryrun 复盘后，进入 [L14 MoE / EP](../l13_moe_ep/README.md)。
