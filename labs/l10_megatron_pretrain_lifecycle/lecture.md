# L11 讲义：Megatron 预训练生命周期与 Train Step

这一讲把训练循环拆成可测试合同。

前面几讲已经分别处理了数据入口、LR scheduler、长上下文 attention。L11 关注这些组件如何在一次训练 step 中协作：清梯度、执行 forward/backward、更新 optimizer、推进 scheduler、写 metrics，并在生命周期脚本里产出 loss 下降、checkpoint 和 report 证据。

Patch 的目标很小：实现 `train_step`。它不训练真实 8 卡模型，也不处理 pipeline schedule。它训练的是工程判断：什么时候认为本步成功，什么时候跳过 scheduler，哪些指标必须写入日志。

## 1. 本讲目标

学完后你应该能回答：

1. Megatron task entry 和通用 training loop 怎样分工。
2. `forward_backward_func` 的输入、输出和失败边界是什么。
3. 多 microbatch loss 应该怎样汇总。
4. optimizer step 的返回值如何解析。
5. optimizer skip 时 scheduler 和 metrics 应怎样处理。
6. `run_lifecycle.py` 产出的 artifact 能证明什么。
7. 真实 Megatron `training.py` 的主路径如何映射到本关 patch。

## 2. Task entry 与通用训练循环

Megatron 预训练通常分两层。

第一层是任务入口。GPT 预训练会提供：

- `model_provider`
- batch 读取函数
- loss 函数
- `forward_step_func`

第二层是通用训练循环。它负责：

- 初始化并行环境。
- 构造模型、optimizer 和 scheduler。
- 准备 dataset 和 data iterator。
- 选择 forward/backward schedule。
- 循环调用 `train_step`。
- 写日志、保存 checkpoint、处理退出条件。

MiniInfra 保留同样边界。`pretrain_gpt.py` 很小，只提供 provider 和 forward step。`training.py` 的 `pretrain` 负责构造模型、optimizer、scheduler、schedule，并在循环中写 metrics 和 checkpoint。

## 3. Train step 的输入输出

本关 patch 的函数签名是：

```python
def train_step(
    forward_backward_func,
    data_iterator,
    model,
    optimizer,
    lr_scheduler,
    iteration,
) -> dict:
    ...
```

输入职责如下：

| 输入 | 作用 |
|---|---|
| `forward_backward_func` | 执行当前 step 的模型计算，返回 loss 信息 |
| `data_iterator` | 提供 batch 或 microbatch |
| `model` | 被训练的模型对象 |
| `optimizer` | 清梯度、更新参数、返回 update 结果 |
| `lr_scheduler` | 在 update 成功后推进 lr |
| `iteration` | 当前训练步编号，进入 metrics |

输出是 metrics dict，至少包含：

```python
{
    "iteration": 7,
    "loss": 2.0,
    "num_microbatches": 2,
    "skipped_iter": 0,
    "lr": 0.05,
}
```

如果 optimizer 返回 grad norm，metrics 也要包含 `grad_norm`。如果 forward/backward 返回 `tokens`，metrics 要把它转成整数写出。

## 4. 调用顺序

主路径是：

```text
optimizer.zero_grad()
forward_backward_func(data_iterator, model)
optimizer.step()
if success:
    lr_scheduler.step()
return metrics
```

顺序很重要。

`zero_grad` 必须在 forward/backward 前执行，否则上一轮梯度会累积进本轮。真实训练中可能故意做 gradient accumulation，但那需要明确的 accumulation boundary；本关的单步合同默认每步清梯度。

`scheduler.step()` 必须在 optimizer update 成功后执行。若发生 overflow 或 skip，本步参数没有更新，lr 时间轴也不应前进。

metrics 最后生成，确保它读取的是本步更新后的 lr，或者 skip 后保持不变的 lr。

## 5. Loss 输出合同

`forward_backward_func` 可以返回两种 loss 形式：

```python
{"loss": 1.2}
{"losses": [1.0, 1.4], "tokens": 4096}
```

第二种对应多个 microbatch。`train_step` 应该对 `losses` 求平均：

```python
loss = sum(losses) / len(losses)
num_microbatches = len(losses)
```

缺少 `loss` 和 `losses`，或者 `losses` 是空列表，应该抛 `TrainStepError`。这类错误通常意味着 forward step、loss function 或 pipeline 汇总出了问题，不能用 0 或 None 悄悄填过去。

## 6. Optimizer step 返回值

本关接受三种 optimizer 返回：

| 返回值 | 含义 |
|---|---|
| `True` | update 成功 |
| `False` | update 被跳过，通常表示 overflow 或其他失败 |
| `None` | 视为成功，兼容常见 PyTorch optimizer |
| `{"success": bool, "grad_norm": float}` | 同时带成功标志和 grad norm |

其他类型应抛 `TrainStepError`。明确边界的好处是日志可解释：`skipped_iter=1` 表示本步没有参数更新，scheduler 不推进；`grad_norm` 存在时可以排查梯度爆炸或 clipping。

## 7. Scheduler 与 LR 读取

当 optimizer update 成功，并且 `lr_scheduler` 不为 None 时：

```python
lr_scheduler.step()
```

当前 lr 的读取优先级：

1. 如果 scheduler 有 `get_lr()`，使用它。
2. 否则从 `optimizer.param_groups[0]["lr"]` 读取。
3. 都没有时返回 None。

这个顺序和真实训练系统的观测需求一致：日志中的 lr 要能解释 loss 曲线和 restart 行为。若 scheduler 在 skip 时误推进，loss 和 lr 的关系会错位。

## 8. Lifecycle drill 怎么看

运行：

```bash
python labs/l10_megatron_pretrain_lifecycle/scripts/run_lifecycle.py \
  --config configs/cpu_smoke.yaml --run-id l11_validation
```

脚本会：

1. 加载 student patch，必要时 fallback 到 reference。
2. 尝试加载 L09 的 `CosineWithRestartsLR`，必要时使用本地 fallback。
3. 构造 tiny decoder、AdamW、scheduler 和 synthetic data iterator。
4. 循环调用 `train_step`。
5. 写 `metrics.jsonl`、checkpoint marker、`acceptance.json` 和 `report.md`。

`acceptance.json` 会记录头部 loss、50 步附近 loss、loss drop、restart 处 lr、最终 loss 和是否全部通过。它是本地生命周期证据，不替代真实 Megatron 集群训练。

## 9. 真实 Megatron 对照

真实 Megatron 的 `pretrain` 文档说明了同一条主线：初始化、构造模型和 lr schedule、获取数据集、用 `forward_step_func` 训练模型。真实 `train_step` 内部会传入 `forward_backward_func`、`num_microbatches`、序列长度、micro batch size 和 pipeline shape；optimizer 成功后再用 batch 相关 increment 推进 scheduler。

真实代码多了很多分支：pipeline parallel、activation logging、溢出检查、分布式 optimizer、tensorboard、checkpoint、auto resume、energy monitor。学习时先抓主线，再把分支归类到“计算调度、优化器状态、日志观测、保存恢复、退出条件”。

## 10. Patch 验收什么

5 个测试覆盖：

1. 调用顺序和 metrics 字段。
2. optimizer skip 时 scheduler 不推进。
3. optimizer 返回 `None` 视为成功。
4. 缺少 loss 时抛 `TrainStepError`。
5. 空 microbatch loss 列表时抛 `TrainStepError`。

测试没有覆盖真实 pipeline、分布式通信、loss scale、activation recompute 或 checkpoint restore。它只保证单步合同正确。

## 11. 生产排查顺序

遇到训练 lifecycle 异常时，按下面顺序查：

1. 调用顺序：`zero_grad` 是否在 forward/backward 前。
2. Loss：是否有缺失、空列表、microbatch 平均错误。
3. Optimizer：step 返回值、skip、grad_norm。
4. Scheduler：成功时推进，skip 时不推进，lr 是否进入 metrics。
5. Metrics：`iteration/loss/num_microbatches/skipped_iter/lr/tokens` 是否齐全。
6. Checkpoint：marker、iteration、optimizer state、scheduler state 是否写出。
7. Lifecycle artifact：`metrics.jsonl`、`acceptance.json`、`report.md` 是否支持结论。

## 12. 小结

L11 的核心是把训练循环从“能跑”改成“可测试、可解释、可恢复”。`train_step` 很小，但它定义了 loss、optimizer、scheduler 和 metrics 的边界。后续进入更大规模训练时，这个边界会继续存在，只是周围多了并行、通信和更复杂的状态管理。
