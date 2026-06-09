# L09 讲义：Megatron 文本预训练训练环与 LR Scheduler

这一讲把数据预处理产物接入训练控制面。

L08 结束时，学生已经能得到一个 Megatron-style IndexedDataset prefix。到了 L09，问题变成：这个 prefix 怎样进入训练 step？一次 step 里哪些状态会改变？学习率曲线怎样跟 optimizer、日志和 checkpoint 对齐？如果本机没有真实 Megatron runtime，drill 应该留下哪些证据，才能说明“本地只验证了启动边界”。

本讲 patch 只做一个小组件：`CosineWithRestartsLR`。它不替代整套 Megatron scheduler，课程用途是训练学生把 scheduler 写成训练框架组件：有输入、有内部计数、有 param group 输出、有边界条件，也要考虑恢复训练时的状态连续性。

## 1. 本讲目标

学完这一讲，你应该能回答：

1. Megatron 预训练从 `--data-path <prefix>` 到 training log 经过哪些边界。
2. `train_step` 中 forward/backward、optimizer、scheduler 的顺序是什么。
3. iteration-based schedule 和 sample-based schedule 的计数单位有什么差异。
4. LR scheduler 的输入、中间状态、输出和恢复边界分别是什么。
5. Cosine with restarts 的 boundaries、段内相对位置和 total clamp 怎样计算。
6. 多 param group 为什么要同步写 lr，真实 Megatron 为什么还要 canonical lr logging。
7. 本地 fallback drill 能证明什么，不能证明什么。

## 2. 真实问题：训练启动成功还不够

预训练命令通常看起来像这样：

```bash
torchrun --nproc_per_node=1 pretrain_gpt.py \
  --data-path data/wikitext/indexed_dataset/wikitext_text_document \
  --seq-length 1024 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1
```

这个命令只说明训练入口拿到了若干配置。真正要证明训练链路成立，还要回答：

- `--data-path` 指向的 prefix 是否能找到 `.bin/.idx`。
- model、optimizer、scheduler 是否被构造出来。
- forward/backward 是否产生 loss。
- optimizer step 是否成功，是否出现 skipped iteration。
- scheduler 是否按正确计数推进。
- training log 是否记录 learning rate、loss、tokens/sec 等字段。
- checkpoint 是否保存 optimizer 和 scheduler 的状态。

很多训练事故发生在这些边界上。命令能启动，但 `--data-path` 错了；loss 有值，但 scheduler 每次恢复都回到初始 lr；日志里看不到 lr，无法判断 loss 抖动是否来自 restart；checkpoint 有 model 权重，却缺少 scheduler state，第二天继续训练时 lr 曲线断裂。

L09 的目标就是让这些状态可见。

## 3. 从数据 prefix 到训练 step

L08 的输出是一个 prefix：

```text
data/wikitext/indexed_dataset/wikitext_text_document
```

Megatron 根据 prefix 找到：

```text
wikitext_text_document.bin
wikitext_text_document.idx
```

进入训练后，可以把路径抽象成：

```text
IndexedDataset prefix
  -> dataset / data iterator
  -> forward_step_func
  -> forward_backward_func
  -> optimizer.step()
  -> opt_param_scheduler.step()
  -> training_log()
  -> checkpoint
```

每一层处理的对象不同：

| 层 | 输入 | 状态变化 | 输出 |
|---|---|---|---|
| Dataset | `.bin/.idx` prefix | 读取样本位置 | token sequence / batch |
| Forward/backward | batch、model | activation、gradient | loss、grad buffer |
| Optimizer | gradients、optimizer state | 参数、动量、方差 | update success、grad norm |
| Scheduler | step 或 consumed samples | `num_steps` / `step_count`、param group lr | 当前 lr |
| Training log | metrics、lr、loss | `metrics.jsonl`、TensorBoard、stdout | 可排查证据 |
| Checkpoint | model/optimizer/scheduler state | checkpoint files | resume 边界 |

把这张表记住，读源码时就不会把 scheduler 当成孤立公式。它位于 optimizer 成功更新之后，日志和 checkpoint 之前。

## 4. 有效 batch 和 scheduler 计数

训练配置里常见三个 batch 量：

```text
micro_batch_size
data_parallel_size
num_microbatches
```

一个 optimizer update 消费的样本数通常是：

```text
micro_batch_size * data_parallel_size * num_microbatches
```

Megatron 的 `train_step` 在 optimizer update 成功后，会用这个增量推进 `opt_param_scheduler.step(increment=...)`。这意味着 scheduler 计数不一定每个 iteration 只加 1。sample-based schedule 关注已经消费多少样本；iteration-based schedule 会先把 iteration 配置换算到样本数或全局 batch 相关的计数。

这个差异会影响三类排查：

1. **LR 曲线错位。** 如果把 Megatron 的 sample-based 计数误当成每 step 加 1，学习率衰减速度会不对。
2. **batch 调整后的 schedule。** 改 `global_batch_size` 会改变每次更新消耗的样本数，LR warmup 和 decay 的实际时间轴也要复查。
3. **resume 连续性。** 恢复训练时要对齐 consumed samples、iteration 和 scheduler state，否则日志曲线会断开。

本关 patch 为了教学简化，用 `step_count += 1`。这不是 Megatron 的完整计数模型，但能把 scheduler 的边界练清楚：什么时候 step、如何算 lr、写到哪里、超过 total 后怎样处理。

## 5. LR scheduler 是训练框架组件

一个 scheduler 至少包含四类信息。

**输入：**
optimizer、`max_lr`、`min_lr`、restart steps、total steps，以及训练循环对 `step()` 的调用。

**中间状态：**
`step_count` 或 `num_steps`，还有由 restart steps 推导出的 boundaries。

**输出：**
写入 optimizer 每个 param group 的 `lr`。日志和 `get_lr()` 再从 optimizer 或 scheduler 里读取当前值。

**边界：**
total 之后钳到 `min_lr`；restart step 属于新 segment 起点；多个 param groups 要同步；恢复训练时要恢复计数。

这个组件的代价也要说清。更复杂的 schedule 会增加配置和恢复状态，restart 还会制造 lr 跳变。它可以给优化器重新提高步长，但也可能让 loss 短期抖动。工程上不能只看最终 loss，要同时看 restart 点、grad norm、skipped iteration 和 loss scale。

## 6. Cosine with restarts 的机制

本关使用显式 restart 列表。给定：

```python
restart_steps = [1000, 3000]
total_steps = 5000
```

先构造 boundaries：

```python
boundaries = [0, 1000, 3000, 5000]
```

相邻两个边界组成一个 segment：

```text
[0, 1000)
[1000, 3000)
[3000, 5000)
```

在某一段内，设：

```text
segment_start = boundaries[i]
segment_end = boundaries[i + 1]
t = step - segment_start
segment_len = segment_end - segment_start
ratio = t / segment_len
```

LR 公式是：

```text
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * ratio))
```

这几个点要能手算：

| step 位置 | ratio | lr |
|---|---:|---|
| segment 起点 | 0 | `max_lr` |
| segment 中点 | 0.5 | `(max_lr + min_lr) / 2` |
| 接近 segment 末端 | 接近 1 | 接近 `min_lr` |
| restart step | 0，新 segment 起点 | `max_lr` |
| `step >= total_steps` | 不再进 segment | `min_lr` |

边界最容易出错。restart step 本身属于新段起点；`total_steps` 之后不要继续套 cosine；空 restart 列表要退化成 `[0, total_steps]` 这一段。

## 7. 多 param group 和 canonical lr

PyTorch optimizer 可以有多个 param groups。例如 embedding、attention、MLP、norm 或特殊参数可能使用不同配置。教学 patch 要求所有 groups 同步写同一个 lr：

```python
for group in optimizer.param_groups:
    group["lr"] = lr
```

真实 Megatron 更复杂。`OptimizerParamScheduler.get_lr(param_group)` 会允许 param group override `max_lr` 和 `min_lr`，`step()` 会遍历所有 param groups 写 lr 和 weight decay。日志层还要找 canonical lr，因为某些并行 rank 上可能存在空的默认 param group。这个细节说明：scheduler 输出不是一个孤立数字，它要服务训练更新和跨 rank 观测。

课堂里先把所有 param groups 同步，是为了让测试聚焦在边界和公式。读真实源码时，再看 override、空组、tensor lr 和 weight decay 的生产复杂度。

## 8. MiniInfra 怎样保留主路径

MiniInfra 的 `train_step` 把 Megatron 主路径压缩成几个动作：

```text
optimizer.zero_grad()
forward_backward_func(...)
optimizer.step()
if success:
    lr_scheduler.step()
metrics["lr"] = current lr
```

这个简化足够说明三件事。

第一，scheduler 只在 optimizer 更新成功后推进。若 step 被跳过，强行推进 lr 会让学习率时间轴和实际参数更新不一致。

第二，metrics 要记录 lr。只看 loss 无法判断一次抖动来自数据、梯度、restart、loss scale 还是 batch 配置。

第三，checkpoint 要保存 scheduler state。MiniInfra 保存 `scheduler_state`，真实 Megatron 也有 scheduler `state_dict()` 和 `load_state_dict()`。

## 9. Drill 和 fallback artifact 怎么看

运行：

```bash
(cd labs/l08_megatron_text_pretrain && bash scripts/train_4090.sh l09_local)
```

脚本会检查三件事：

1. `data/wikitext/train.jsonl` 是否存在。
2. L08 的 indexed `.bin/.idx` 是否存在。
3. Python 环境能否导入 Megatron。

缺任何一项时，drill 会走 fallback，写出：

| artifact | 含义 |
|---|---|
| `artifacts/fallback_reason.txt` | 哪些前置条件缺失 |
| `train.log` | 记录 expected command |
| `metrics.jsonl` | 标记 `fallback_validated` 或 `ready_for_manual_megatron_launch` |
| `report.md` | 汇总配置、预测、结果和风险 |

这不是坏事。教学环境经常没有完整 Megatron runtime。关键是不要把 fallback 写成真实训练结果。只要 artifact 清楚，后续在集群上接真实 Megatron 时就知道还缺什么。

日志解析器 `parse_megatron_log.py` 提供另一个小工具。它从文本里提取 iteration、loss、grad norm、consumed samples、tokens/sec 和 MFU。真实训练日志里这些字段越完整，越能支撑训练状态判断；只有 expected command 时，只能说明启动边界被记录了。

## 10. Patch 验收什么

Patch 不要求改 Megatron 源码，只实现：

```python
CosineWithRestartsLR(
    optimizer,
    max_lr,
    min_lr,
    restart_steps,
    total_steps,
)
```

测试覆盖的最小合同是：

1. 初始化后 lr 是 `max_lr`。
2. 段内 lr 单调下降。
3. restart step 回到 `max_lr`。
4. `step >= total_steps` 后返回 `min_lr`。
5. 多 param groups 同步更新。
6. `get_lr()` 与 optimizer 中的值一致。
7. 无 restart 时等价于单段 cosine。

测试没有覆盖 warmup、checkpoint restore、weight decay、param group override、分布式 rank 差异和真实 loss 曲线。它只保证 scheduler 的核心合同正确。真实训练中还要把它接入配置、日志、checkpoint 和 resume。

## 11. 生产排查顺序

遇到“训练不稳定”或“LR 曲线不对”时，按下面顺序排查：

1. 看 resolved config：`lr`、`min_lr`、warmup、decay、restart、global batch、TP/PP/DP。
2. 看训练日志：learning rate 是否按预期变化，restart 点是否和 loss 抖动对齐。
3. 看 optimizer update：是否有 skipped iteration、grad norm 异常、loss scale 问题。
4. 看 scheduler state：`num_steps` 或 `step_count` 是否和 consumed samples / iteration 对齐。
5. 看 checkpoint：resume 前后的 scheduler、optimizer 和 data state 是否连续。
6. 看 fallback artifact：本地结果是否只是启动边界验证。

这个顺序的好处是先确认证据，再定位组件。很多训练问题源于计数单位、恢复状态或日志缺字段，公式错误只是其中一种可能。

## 12. 小结

L09 讲的是训练控制面，不是单独的 LR 数学。你应该能从 L08 的 data prefix 画到 L09 的 training log，解释 optimizer 成功更新后 scheduler 为什么推进，说明 cosine with restarts 怎样计算 boundaries 和段内相对位置，并用 artifact 判断一次训练验证的证据强度。

完成 patch 后，你得到的是一个最小 scheduler 组件。把它放回真实训练系统时，还要关心配置、计数单位、param group、日志、checkpoint 和 resume。
