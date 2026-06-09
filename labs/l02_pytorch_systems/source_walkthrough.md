# 源码带读：L02 PyTorch 显存账本与训练 step

这份带读按“显存计数函数 -> tiny transformer smoke -> MiniInfra trainer -> Megatron-shaped train_step”的顺序走。每一步只看主路径，先建立输入、状态、输出和 artifact 关系。

## 0. 源码地图

```text
labs/l02_pytorch_systems/patch/reference/memory_probe.py
  -> count_param_bytes
  -> count_grad_bytes
  -> count_optimizer_state_bytes

labs/l02_pytorch_systems/scripts/train_tiny_transformer.py
  -> SyntheticTokens
  -> TinyDecoder
  -> main training loop
  -> metrics/report/profiler artifact

mini_infra/training/trainer.py
  -> MiniTrainer.run
  -> _run_simulated / _run_torch

mini_infra/model/tiny_transformer.py
  -> TinyModelConfig
  -> build_torch_model

mini_infra/megatron/training/training.py
  -> train_step
  -> training_log
  -> pretrain
```

patch 负责静态账本，smoke 负责观察训练 step，MiniInfra/Megatron-shaped 文件负责把这个模型扩展到后续课程。

## 1. Patch reference：参数 bytes

文件：`labs/l02_pytorch_systems/patch/reference/memory_probe.py`

重点看第 9-10 行：

```python
def count_param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())
```

读完要得到三个结论。

第一，参数字节数按每个 parameter 自己的 `element_size()` 计算，支持混合 dtype。

第二，这里只统计 `model.parameters()`，不统计 buffer、activation 或 CUDA allocator cache。

第三，返回值是整数 bytes，不是 MB 或 GB。单位转换应该在报告层做。

## 2. Patch reference：grad bytes

重点看第 13-18 行：

```python
total = 0
for p in model.parameters():
    if p.grad is not None:
        total += p.grad.numel() * p.grad.element_size()
return total
```

这里的关键分支是 `p.grad is not None`。没有 backward 或 `zero_grad(set_to_none=True)` 后，grad 可以是 `None`，这时不占用 grad tensor bytes。

读完要形成一个判断：grad bytes 是训练运行状态，不是模型结构的静态属性。它取决于是否已经 backward，以及 zero_grad 的策略。

## 3. Patch reference：optimizer state bytes

重点看第 21-27 行：

```python
for state in optimizer.state.values():
    for v in state.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
```

这段代码只统计 optimizer state 中的 tensor entry。plain SGD 没有 tensor state，Adam 一步后通常有 `exp_avg` 和 `exp_avg_sq`。

可以先跳过：optimizer state 里非 tensor 的 step 计数或 Python 标量。本讲只把 tensor bytes 计入显存账本。

## 4. train_tiny_transformer：合成数据和模型

文件：`labs/l02_pytorch_systems/scripts/train_tiny_transformer.py`

先看第 36-54 行：

```python
class SyntheticTokens(Dataset):
    ...
    def __getitem__(self, index):
        if self.sleep_ms:
            time.sleep(...)
        generator = torch.Generator().manual_seed(index)
        x = torch.randint(...)
        y = torch.roll(x, shifts=-1)
        return x, y
```

这里生成可复现 token 序列，`sleep_ms` 用来模拟 dataloader 慢。读完要知道：dataloader 时间异常时，不一定是 GPU 慢。

再看第 57-99 行。`TinyDecoder` 包含 embedding、position embedding、若干 `TransformerEncoderLayer`、LayerNorm 和 head。第 95-98 行根据 `checkpoint_layers` 决定是否用 activation checkpointing。

可以先跳过：具体 TransformerEncoderLayer 内部实现。这里只关心 forward 会产生 activation，并且 checkpointing 会改变保存/重算边界。

## 5. train_tiny_transformer：配置、设备和 profiler

重点看第 102-116 行：

```python
def load_config(path: str) -> dict: ...
def make_device() -> torch.device: ...
def set_precision(config: dict) -> tuple[torch.dtype | None, bool]: ...
```

这三个函数把配置转成运行条件：设备、precision、autocast 是否开启。性能和显存结论必须带这些条件。

再看第 119-128 行：

```python
profiler_dir = run_dir / "artifacts" / "profiler"
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)
with profile(activities=activities) as prof:
    with record_function("warmup_forward"):
        model(batch)
prof.export_chrome_trace(...)
```

这里导出 warmup forward 的 trace。读完要记住：这份 trace 只覆盖一次 forward，不代表完整训练长期吞吐。

## 6. train_tiny_transformer：训练 step 主路径

重点看第 167-227 行。

第 171-177 行记录 dataloader 时间。第 179-181 行把数据移动到 device，并在 step 开始时 `zero_grad(set_to_none=True)`。

第 183-193 行记录 forward 和 loss 时间。autocast 分支只在配置允许且 CUDA 可用时生效。

第 195-199 行执行 backward 和 grad clip。这里是 grad tensor 产生的地方，也是 `count_grad_bytes()` 变得有意义的位置。

第 201-204 行执行 optimizer step 并记录 optimizer 时间。AdamW 的 state 通常在第一次 step 后出现。

第 206-226 行写 metrics row。重点字段是 `step_time_ms`、`tokens_per_sec`、`peak_memory_gb`、`dataloader_time_ms`、`forward_time_ms`、`backward_time_ms`、`optimizer_time_ms` 和 `grad_norm`。

读完这段要能画出 step 时序图，并说明每个指标来自哪个计时代码块。

## 7. train_tiny_transformer：报告和 artifact

重点看第 230-240 行：

```python
write_text(run_dir / "train.log", ...)
last_loss = json.loads(log_lines[-1])["loss"]
write_text(run_dir / "report.md", ...)
```

这段把 step 结果落成 `train.log` 和 `report.md`。后续几行会写模型、数据、并行和结果说明。

可以先跳过报告文案细节。关键结论是：训练脚本不仅跑模型，还要把证据写成后续能复查的 artifact。

## 8. MiniInfra trainer：从 smoke 到可复用 trainer

文件：`mini_infra/training/trainer.py`

先看第 14-24 行。`TrainConfig` 固定 backend、steps、batch_size、seq_len、learning_rate 和 seed。这是训练 run 的最小配置。

看第 27-39 行。`MiniTrainer.run()` 根据 backend 选择 simulated 或 torch。这个分支让课程在没有 GPU 或没有完整 torch 环境时仍能跑证据链。

看第 41-79 行。`_run_simulated()` 不训练真实模型，但写 `metrics.jsonl` 和模拟 checkpoint。它用来证明 artifact 合同。

看第 81-139 行。`_run_torch()` 执行真实 PyTorch 训练：建模型、建 AdamW、生成 input/label、zero_grad、forward、loss、backward、optimizer.step、metrics、checkpoint。

读完要得到一个结论：trainer 的核心是固定训练生命周期和 artifact 边界，代码长度不是目标。

## 9. Tiny model：参数规模来自配置

文件：`mini_infra/model/tiny_transformer.py`

看第 7-17 行。`TinyModelConfig` 定义 vocab、hidden、layer、head、seq_len 和 dropout。

看第 20-50 行。`build_torch_model()` 根据配置构建 embedding、position、TransformerEncoder 和 head。这个模型是 `MiniTrainer._run_torch()` 的训练对象。

看第 53-58 行。`count_parameters_from_config()` 用近似公式估参数量。它用于模拟模式和规划，不能替代 patch 的精确 `count_param_bytes()`。

读完要能解释：配置改变会同时影响参数规模、activation 规模和训练时间。

## 10. Megatron-shaped train_step：后续课程的主路径

文件：`mini_infra/megatron/training/training.py`

先看第 83-116 行：

```python
def train_step(...):
    if hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()
    output = forward_backward_func(data_iterator, model)
    losses = _as_loss_list(output)
    loss = sum(losses) / len(losses)
    success, grad_norm = _parse_step_result(optimizer.step())
    if success and lr_scheduler is not None:
        lr_scheduler.step()
    metrics = {...}
    return metrics
```

这就是 Megatron-shaped 训练 step 的核心合同：清梯度、forward/backward、聚合 loss、optimizer step、lr scheduler、metrics。

再看第 119-122 行。`training_log()` 把 metrics 写入 `metrics.jsonl`。

最后看第 125-187 行。`pretrain()` 初始化状态、读数据、建模型、建 optimizer、选择 pipeline schedule、循环 train_step、写 checkpoint 和 artifacts。

读完要形成一个判断：L02 的单进程显存账本会在后续并行训练中继续存在，只是参数、梯度、optimizer state 可能被分片，activation 和通信 buffer 会变得更复杂。

## 11. Patch tests：行为合同从哪里来

文件：`labs/l02_pytorch_systems/patch/tests/test_patch.py`

重点看这些测试：

- `test_count_param_bytes_matches_manual`
- `test_works_with_mixed_dtype`
- `test_count_grad_bytes_after_backward`
- `test_count_grad_bytes_zero_when_none`
- `test_optimizer_state_sgd_zero`
- `test_optimizer_state_sgd_momentum_one_x_params`
- `test_optimizer_state_adam_two_x_params`

读测试时不要只看断言结果，要问每个测试制造了什么状态：有没有 backward，grad 是否存在，optimizer 是否 step 过，optimizer 是否带 momentum，dtype 是否混合。

## 12. 读完后的自检问题

1. `count_param_bytes()` 为什么不能写死 fp32？
2. 为什么 `count_grad_bytes()` 在 backward 前应该返回 0？
3. Adam optimizer 刚创建时 state 可能为空，为什么测试要先 backward 和 step？
4. `peak_memory_gb` 比静态账本大很多时，优先怀疑哪些运行时内存？
5. `dataloader_time_ms` 很高时，为什么不能先怪 GPU？
6. MiniInfra `train_step` 和真实 Megatron 相比，保留了哪些生命周期不变量？
