# L36 · SLiME Weight Sync Coordinator

RL 训练里的 rollout engine 不能长期使用旧 policy。actor 每训练一轮，权重都会变化；如果新权重没有及时交给 rollout 侧，下一批样本仍来自旧模型，PPO/GRPO 看到的 old logprob、KL、reward 和 advantage 都会变得难解释。L36 讲的就是这个交接点：weight sync。

本讲先写一个 CPU 本地版 `WeightSyncCoordinator`，把 key、shape、dtype、clone 和 stats 这些同步合同练清楚。然后对照 SLiME 的真实路径：Ray actor 触发更新，Megatron actor 选择 updater，updater 把权重交给 SGLang rollout engines。

## 1. 本讲目标

- 解释 weight sync 在 generate -> train -> update_weights loop 中的位置。
- 实现本地 state_dict 同步合同：只复制 shape/dtype 匹配的 tensor。
- 说明 `bytes_synced`、`num_tensors` 和 `mismatched_keys` 的排障价值。
- 对照 SLiME 源码读懂 Ray、Megatron updater 和 SGLang engine 的分工。
- 用 smoke artifact 判断 actor/rollout 资源切分和同步成本的边界。

## 2. Weight Sync 的系统位置

RLHF 训练通常有两组角色。训练侧 actor 负责反向传播、optimizer step 和 checkpoint；rollout 侧 inference engine 负责接收 prompt、生成 response、计算或收集 reward/logprob。两侧可能在不同进程、不同 GPU 组，甚至不同节点上。

一次同步 RL loop 可以简化成：

```text
rollout_manager.generate()
  -> actor_model.train()
  -> actor_model.update_weights()
  -> rollout_manager uses newer weights
```

如果 `update_weights()` 没有真正生效，训练侧会继续推进新 actor，rollout 侧却仍用旧 actor 采样。短期 stale policy 可以被 PPO 的 ratio 和 clip 吸收一部分；stale 太大时，ratio 偏离、clip 频繁触发，KL 和 reward 曲线都会失去清晰解释。

## 3. 本地 Patch 的输入、状态和输出

`WeightSyncCoordinator` 有三个外部接口：

- `train_state_provider()`：返回训练侧 `state_dict`。
- `inference_state_provider()`：返回推理侧当前 `state_dict`，可选。
- `inference_state_setter(new_state)`：把 accepted tensors 写到推理侧。

`sync()` 的输出是 stats：

```python
{
    "bytes_synced": 589824,
    "num_tensors": 12,
    "mismatched_keys": ["lm_head.weight"],
}
```

这三个数字对应三类问题。`bytes_synced` 用来估算传输量和有效带宽；`num_tensors` 用来确认是否真的同步了参数；`mismatched_keys` 用来定位 checkpoint 转换、TP/PP 切片、dtype 策略或模型结构差异。

本地 patch 不处理真实分布式通信。它先把“哪些 tensor 允许同步”定义清楚。

## 4. 同步合同

同步过程按四步走：

1. 读取 train state。
2. 读取 inference state。
3. 遍历 train key，并逐项检查 key、shape、dtype。
4. 将 accepted tensors clone 后交给 setter，并返回 stats。

只有 key 存在、shape 一致、dtype 一致时才复制。train 多出的 key 不创建到 inference；shape 或 dtype 不一致时写入 `mismatched_keys`；inference 多出的 key 保持原样。

`detach().clone()` 很重要。直接把 train tensor 放进 inference dict，会让两边共享对象，后续训练侧修改可能影响推理侧状态。clone 之后，inference 拿到的是独立快照。

dtype 不自动转换。真实训练里常见 fp32 master weights、bf16/fp16 inference weights。转换应在发送前明确发生，并记录目标 dtype；同步函数静默 cast 会掩盖带宽变化和数值策略。

## 5. Patch Tests 验收什么

五个测试覆盖本地合同：

- `test_basic_sync`：两个匹配 tensor 都被复制，数值完全相同。
- `test_shape_mismatch_skipped`：shape 不同的 key 进入 mismatch，inference 旧值不变。
- `test_dtype_mismatch_skipped`：dtype 不同不自动 cast。
- `test_extra_train_keys_skipped`：train 多出的 key 不创建到 inference。
- `test_stats_correct`：`bytes_synced` 和 `num_tensors` 只统计 accepted tensors。

这些测试没有覆盖 Ray object store、NCCL group、Megatron/HF 名称转换、SGLang HTTP endpoint 或 engine lock。它们只证明本地同步合同正确。通过 patch-test 后，还要用 smoke 和源码对照解释生产路径多出的复杂度。

## 6. MiniInfra SLiME Loop

MiniInfra 用很短的代码保留 SLiME 的主形状。`train.py` 创建 `RolloutManager` 和 `TrainRayActor`，每步先 `generate`，再 `train`，最后 `update_weights`。`TrainRayActor.train` 递增 `weight_version`，`update_weights` 把版本写回 rollout manager。`RolloutManager.generate` 只选择 fresh enough 的 server。

这条链路说明两个事实。第一，weight sync 是 loop 的显式步骤，不会自动发生。第二，`weight_version` 是最小证据；没有版本记录，就无法判断 rollout 样本来自哪个 policy。

## 7. SLiME 真实源码对照

真实 SLiME 的同步路径分几层：

```text
train.py
  -> actor_model.update_weights()
  -> RayTrainGroup.update_weights()
  -> MegatronTrainRayActor.update_weights()
  -> UpdateWeightFromTensor / UpdateWeightFromDistributed
  -> SGLang engine update_weights_from_*
```

`RayTrainGroup.update_weights()` 对每个 train actor 发 remote 调用。Megatron actor 初始化时会根据 `args.colocate` 选择 updater：co-locate 走 tensor/IPC 路径，disaggregate 走 distributed/NCCL 路径。真正更新时，actor 从 rollout manager 获取可更新 engines 和 engine lock，然后调用 updater。

SGLang engine 侧暴露多种在线更新接口。tensor 路径通过 HTTP 发送元数据，真实 tensor 走 GPU/IPC；distributed 路径先把 names、dtypes、shapes、group_name 和 weight_version 发给 engines，再用 NCCL broadcast 发送 tensor。disk update 也存在，但不适合作为高频 RL sync 主路径。

## 8. 分布式同步为什么复杂

训练侧参数不一定长得像推理侧 state_dict。Megatron 训练可能有 Tensor Parallel、Pipeline Parallel、Expert Parallel、checkpoint padding 和 Megatron/HF 名称差异。SLiME 的 updater 要做 TP gather、EP gather、HF 转换、bucket、engine lock 和 group broadcast。

同步慢时，不要只看网络带宽。还要拆：

- gather 时间：train shards 变成目标 tensor。
- conversion 时间：Megatron 名称和布局转成 HF/SGLang 需要的形式。
- bucket 时间：参数按 buffer size 分批。
- lock 等待：rollout engine 是否正在 generation。
- broadcast 时间：NCCL/RDMA 传输。
- engine 应用时间：SGLang 接收 metadata、更新权重、flush cache。

本地 patch 的 `bytes_synced` 是这个链路的缩影。生产排查时，把参数量除以耗时可以估算有效带宽，再和理论带宽比较，定位瓶颈是在传输还是在前后处理。

## 9. Dtype、版本和同步频率

dtype 是最常见的同步边界之一。训练侧可能保存 fp32 master weights，推理侧使用 bf16/fp16。直接同步 fp32 会增加传输量，也可能和推理侧 dtype 不匹配。正确做法是在发送方显式决定目标 dtype，并让接收方校验。

版本是另一条安全线。每次 actor 权重更新后，rollout engine 应记录新的 `weight_version`。如果 metrics 里 reward 变化了，但 rollout version 没推进，优先怀疑 sync 未生效。同步频率也要平衡：太勤会增加 step time；太疏会让 rollout policy 变旧，importance ratio 和 KL 变差。

## 10. 本地 SLiME Smoke

`scripts/run_slime_lab.py` 不跑真实 SLiME 训练。它验证配置和数据闭环：

- 读取 `configs/actor_rollout_split.yaml`。
- 准备 toy GSM8K prompts。
- 检查 `slime` 和 `sglang` 包是否可导入。
- 写 `artifacts/slime_config_validation.json`。
- 写 `metrics.jsonl`，包含 `rollout_time_sec`、`actor_update_time_sec`、`weight_sync_time_sec`、tokens/sec 和 GPU allocation。
- 写报告，声明本地模式不能替代 H200/NCCL 真实验证。

这个 smoke 的价值在于训练证据格式。它不能证明真实 SGLang engine 成功更新，也不能证明 NCCL sync 带宽达标。

## 11. Debug 路线

sync 慢时，先固定 run：命令、配置、actor/rollout GPU 切分、参数量、sync interval、`weight_sync_time_sec` 和 `total_step_time`。然后拆传输路径：Ray 元数据、NCCL broadcast、checkpoint 文件、SGLang reload、engine lock、cache flush。用 bytes / seconds 算有效带宽，不要只看绝对耗时。

sync 错时，先看 `mismatched_keys`。shape mismatch 常来自模型结构、TP/PP/EP 切片或名称转换；dtype mismatch 常来自 fp32 master、bf16/fp16 inference 或量化配置。版本不推进时，检查 `actor_model.update_weights()` 是否被调用、rollout manager 是否返回 updatable engines、SGLang engine 是否更新 `weight_version`。

rollout reward 异常时，把它和 L35 的 rollout artifacts、L31 的 KL/reward evidence 串起来看：policy version、KL、reward parser、entropy 和 response length 必须一起解释。

## 12. Lab 验收

patch 命令：

```bash
IMPL=reference make patch-test M=l32_slime_rl_core
```

smoke 命令：

```bash
python labs/l32_slime_rl_core/scripts/run_slime_lab.py --run-id l36_smoke
```

完成 L36 后，学生应该能从本地 tensor copy 讲到 SLiME 生产路径：Ray 触发、Megatron 准备权重、updater 传输、SGLang engine 接收，最后用 stats 和 version 判断同步是否可信。
