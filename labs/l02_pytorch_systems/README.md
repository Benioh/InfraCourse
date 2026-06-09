# L02 · PyTorch 系统：显存账本与训练 step 证据链

这一讲解决训练系统里最常见的基础问题：一个模型到底占多少显存，训练 step 的时间花在 dataloader、forward、backward、optimizer 的哪一段，哪些证据能支撑你的判断。

patch 很小，只实现 3 个显存计数函数：参数、梯度、optimizer state。讲授重点放在 PyTorch 训练 step 的显存组成、时间分解、activation 边界、profiler 证据，以及这些概念如何映射到后续 Megatron training lifecycle。

## 学习路线

建议按下面顺序走，先把系统讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L02 在训练系统主线里的位置。
2. 读 [lecture.md](lecture.md)：理解 params、grads、optimizer state、activation、step timing 和 profiler。
3. 读 [source_walkthrough.md](source_walkthrough.md)：跟着路径读 patch、tiny trainer、MiniInfra trainer 和 Megatron-shaped `train_step`。
4. 跑 notebook：[n01_gpu_memory_anatomy.ipynb](../../notebooks/n01_gpu_memory_anatomy.ipynb) 和 [n02_pytorch_profiler.ipynb](../../notebooks/n02_pytorch_profiler.ipynb)。
5. 做 quiz：确认显存账本、grad 生命周期、optimizer state 和 profiler 证据。
6. 做 patch：实现显存计数函数。
7. 跑 smoke：观察 tiny transformer 的 step timing、loss、tokens/s、peak memory 和 profiler trace。
8. 填写 [outputs/training_step_template.md](outputs/training_step_template.md)，沉淀本讲复盘结论。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems |
| 它解决什么问题 | 训练前先估算静态显存，训练中按阶段记录 step 时间和 peak memory |
| 它连接哪些指标 | param bytes、grad bytes、optimizer state bytes、activation peak、step_time_ms、tokens_per_sec、dataloader/forward/backward/optimizer time |
| 它连接哪些源码 | `patch/reference/memory_probe.py`、`scripts/train_tiny_transformer.py`、`mini_infra/training/trainer.py`、`mini_infra/model/tiny_transformer.py`、`mini_infra/megatron/training/training.py` |
| lab 检验什么 | 参数、梯度和 optimizer state 字节数的最小合同 |

## 你会学到什么

- 用 `numel() * element_size()` 精确计算模型参数字节数。
- 解释 backward 后 `.grad` 何时存在，`zero_grad(set_to_none=True)` 如何影响 grad 显存。
- 区分 plain SGD、SGD momentum、Adam/AdamW 的 optimizer state 显存。
- 说明 activation 显存为什么和 batch size、sequence length、层数相关，并且不能只靠参数账本估算。
- 用训练脚本把 dataloader、forward、backward、optimizer、step 总时间拆开。
- 读懂 profiler trace 的边界：它能定位阶段和 kernel，但结论必须带输入规模和硬件条件。
- 把 tiny trainer 的 forward/backward/optimizer/checkpoint 证据链映射到 Megatron-shaped `train_step`。

## Patch 闭环

```bash
cat labs/l02_pytorch_systems/patch/task.md
$EDITOR labs/l02_pytorch_systems/patch/starter/memory_probe.py
make patch-test M=l02_pytorch_systems
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_count_param_bytes_matches_manual` | 参数字节数等于手算公式 |
| `test_works_with_mixed_dtype` | fp16/fp32 混合 dtype 按各自 element size 计算 |
| `test_count_grad_bytes_after_backward` | backward 后 grad bytes 与 fp32 param bytes 相同 |
| `test_count_grad_bytes_zero_when_none` | grad 为 `None` 时返回 0 |
| `test_optimizer_state_sgd_zero` | plain SGD 没有 tensor state |
| `test_optimizer_state_sgd_momentum_one_x_params` | momentum buffer 约等于 1 倍参数 |
| `test_optimizer_state_adam_two_x_params` | Adam 的一阶/二阶矩约等于 2 倍参数 |

## Smoke 闭环

CPU 或本地 GPU：

```bash
python labs/l02_pytorch_systems/scripts/train_tiny_transformer.py \
  --config configs/4090_debug.yaml
```

显存压力配置：

```bash
python labs/l02_pytorch_systems/scripts/train_tiny_transformer.py \
  --config configs/memory_stress.yaml
```

smoke 会生成：

- `metrics.jsonl`：每 step 的 loss、step time、tokens/s、peak memory、阶段耗时和 grad norm。
- `train.log`：每 step 的 JSON 行。
- `artifacts/profiler/trace.json`：一次 warmup forward 的 profiler trace。
- `report.md`：训练目标、配置、结果和排查提示。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | OOM、step time、dataloader、backward、optimizer state 的排查顺序 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速复习显存账本和训练 step 源码主路径 |
| [outputs/training_step_template.md](outputs/training_step_template.md) | 跑 smoke 或真实训练后填写的 step 复盘模板 |

## 进入下一讲

`make patch-test M=l02_pytorch_systems` 通过，并完成一次 tiny transformer smoke 复盘后，进入 [L03 Memory Snapshot](../l02.5_memory_snapshot/README.md)。下一讲会把显存账本扩展到长期 OOM 的泄露归因。
