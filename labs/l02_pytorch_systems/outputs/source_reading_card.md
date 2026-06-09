# Source Reading Card：L02 PyTorch 显存账本

## 主路径

1. `labs/l02_pytorch_systems/patch/reference/memory_probe.py`
   - `count_param_bytes()`：参数字节数 = `numel * element_size`。
   - `count_grad_bytes()`：只统计非 `None` 的 `.grad`。
   - `count_optimizer_state_bytes()`：只统计 optimizer state 里的 tensor entry。

2. `labs/l02_pytorch_systems/scripts/train_tiny_transformer.py`
   - `SyntheticTokens`：生成可复现 token 和 shifted target。
   - `TinyDecoder`：构建 embedding、Transformer layers、head。
   - main loop：记录 dataloader、forward、backward、optimizer、tokens/s、peak memory。
   - `maybe_profile()`：导出 warmup forward trace。

3. `mini_infra/training/trainer.py`
   - `MiniTrainer.run()`：选择 simulated 或 torch backend。
   - `_run_torch()`：zero_grad、forward、loss、backward、optimizer.step、metrics、checkpoint。

4. `mini_infra/model/tiny_transformer.py`
   - `TinyModelConfig`：模型规模入口。
   - `build_torch_model()`：构建可训练 tiny LM。
   - `count_parameters_from_config()`：配置级参数量估算。

5. `mini_infra/megatron/training/training.py`
   - `train_step()`：Megatron-shaped 训练 step 合同。
   - `training_log()`：写 metrics。
   - `pretrain()`：初始化、数据、schedule、循环、checkpoint、artifact。

## 关键结论

- 参数字节数必须按每个 tensor 的 dtype 计算。
- `.grad` 是运行时状态，backward 前可能为 `None`。
- optimizer state 通常在第一次 `optimizer.step()` 后出现。
- Adam/AdamW 的静态 state 通常约等于 2 倍参数。
- activation 显存要通过运行时 peak memory 和 profiler 观察。
- step timing 需要拆开 dataloader、forward、backward 和 optimizer。
- MiniInfra 保留训练生命周期主合同，真实 Megatron 会加入并行和容错复杂度。

## 自检

- 我能否解释 patch 三个函数各自统计什么、排除什么？
- 我能否用 `metrics.jsonl` 判断 step 慢在哪一段？
- 我能否说明 profiler trace 的测试条件和覆盖范围？
- 我能否把 tiny trainer 的 step 映射到 Megatron-shaped `train_step`？
