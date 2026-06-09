# L16：Megatron Parallel Checkpoint

## 0. 本讲目标

- 理解完整训练恢复需要哪些状态，权重文件为什么只是其中一部分。
- 能解释 latest marker、iteration、format 和 checkpoint payload 的职责。
- 能判断 TP/PP/DP/EP 变化为什么会影响 checkpoint 兼容性。
- 能实现 save/load 的 strict 与 non-strict parallel_state 校验。
- 能把 patch 的 JSON 合同对照到 Megatron 的 checkpoint path、state_dict 和 distributed optimizer metadata。

## 1. 问题入口：checkpoint 文件存在，不等于可以 resume

训练中断后，工程上最危险的情况是表面加载成功，实际训练状态已经错位。模型权重可能能恢复，但 optimizer 的动量、scheduler 的 step、RNG、数据进度或并行拓扑不一致，都会让后续训练偏离中断前状态。

Megatron 训练里还多了并行切片。Tensor Parallelism 改变线性层权重的切分方式；Pipeline Parallelism 改变 layer 属于哪个 stage；Expert Parallelism 改变 expert shard 的 rank 坐标；distributed optimizer 会把 optimizer state 按数据并行或其他格式分片。checkpoint 里的 shard 带有保存时的坐标系。坐标系变了，旧 shard 的含义也变了。

L16 用一个小 patch 讲清这件事：保存时把训练状态和并行状态一起写入 payload；加载时先读 latest marker，再检查 format 和 parallel_state。这个合同不能完成真实 reshard，但它能把不兼容拓扑挡在 load 阶段。

## 2. Checkpoint payload：恢复合同的主体

**定义：** checkpoint payload 是落盘的结构化训练状态。本关要求包含：

- `format`：格式标识，用来拒绝未知或旧格式。
- `iteration`：保存时的训练步。
- `model_state`：模型权重或教学版权重 dict。
- `optimizer_state`：优化器继续更新所需的状态。
- `scheduler_state`：学习率调度器的位置。
- `parallel_state`：保存时 TP/PP/DP/EP 等并行拓扑。

直觉上，payload 是“这次训练在某一刻的合同”。如果只保存 model_state，最多能说明模型当前位置；optimizer_state 决定下一步更新方向和尺度；scheduler_state 决定下一步学习率；parallel_state 决定这些状态的切片坐标。

本关用 JSON 保存 payload。真实 Megatron 会用 torch、torch_dist、torch_dcp、fsdp_dtensor 等不同格式保存 sharded state dict，但恢复合同相同：loader 必须知道自己读到的是什么、属于哪一步、适用于哪种并行拓扑。

## 3. Latest marker：恢复入口不能靠猜

**定义：** latest marker 是 `latest_checkpointed_iteration.txt`，内容是最近一次成功保存的 iteration。本关的 loader 先读 marker，再拼出 `iter_0000010.json` 这样的 checkpoint 文件名。

这个 marker 解决的是恢复入口问题。训练目录里可能有多个 iteration 文件，也可能有一次保存失败后留下半成品。如果 loader 用文件名排序猜最新文件，容易读到未完成或不该使用的 checkpoint。正确做法是由 save 流程在 checkpoint 文件写完后更新 marker，load 流程把 marker 当成入口。

Megatron 的 `get_checkpoint_tracker_filename` 也使用 `latest_checkpointed_iteration.txt`。`read_metadata` 会解析 marker，如果分布式已经初始化，还会跨 rank all-reduce iteration，防止某些 rank 读到不同 metadata。

## 4. Parallel state：checkpoint 的坐标系

**定义：** parallel_state 是描述 checkpoint shard 坐标系的元数据。L16 patch 使用 `tp`、`pp`、`dp`、`ep` 这类字段。保存时写入 actual parallel state；加载时由当前作业传入 `expected_parallel_state`。

校验逻辑很直接：逐个比较 expected key。如果 actual 值不同，`strict=True` 抛 `CheckpointError`，阻止错误 resume；`strict=False` 返回 warnings，允许诊断、转换或 finetune 流程继续读取 payload。

strict 和 non-strict 对应两种工作流。训练恢复默认应 strict，因为目标是继续同一个训练状态。转换工具或调试脚本可以 non-strict，因为它们需要先读 payload，分析哪些维度不匹配，再决定是否转换或丢弃 optimizer state。

比较时不能只看 world size。`tp=2, pp=4` 和 `tp=4, pp=2` 的 world size 可能相同，但权重和 layer 的 shard 坐标不同。`ep` 或 `cp` 这类新维度也应该进入检查范围，否则旧 loader 可能静默接受新拓扑。

## 5. Optimizer 与 scheduler：resume 的连续性

model_state 决定 forward 输出，optimizer_state 决定下一步参数更新，scheduler_state 决定下一步学习率。三者缺一，训练都可能不再是同一条轨迹。

以 Adam 为例，optimizer state 通常包含一阶动量、二阶矩、step、master weight 或分布式优化器内部 buffer。缺少这些状态，下一步更新会像从新优化器开始，loss 可能跳变。scheduler_state 也类似；如果恢复后学习率回到 warmup 初始值，训练曲线会突然改变。

真实 Megatron 的 `generate_state_dict` 会保存 args、checkpoint_version、iteration、model、optimizer、opt_param_scheduler、rerun state 和 RNG state。L16 patch 没有展开 RNG 和数据进度，但它把 optimizer_state 和 scheduler_state 放进最小合同，防止学生把 checkpoint 简化成权重文件。

## 6. Patch 机制：save 与 load 的输入、状态、输出

`save_checkpoint` 的输入是输出目录、iteration、model/optimizer/scheduler/parallel state。执行过程：

1. 拒绝负 iteration。
2. 创建输出目录。
3. 组装 payload。
4. 写 `iter_{iteration:07d}.json`。
5. 写 `latest_checkpointed_iteration.txt`。
6. 返回 checkpoint 文件路径和 marker 路径。

`load_checkpoint` 的输入是 checkpoint 目录、可选 expected parallel state 和 strict 标志。执行过程：

1. 读取 latest marker。
2. 解析 iteration。
3. 拼出 checkpoint 文件路径。
4. 读取 JSON payload。
5. 校验 `format`。
6. 比较 parallel_state。
7. strict mismatch 抛错，non-strict mismatch 写入 warnings。
8. 返回 payload、checkpoint_path 和 warnings。

下面的最小示例演示 reference 行为：

```bash
python - <<'PY'
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, "labs/l15_megatron_parallel_checkpoint/patch")
from reference.checkpointing import load_checkpoint, save_checkpoint

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    save_checkpoint(
        root,
        iteration=10,
        model_state={"layer": [1, 2]},
        optimizer_state={"step": 10},
        scheduler_state={"last_lr": 3e-4},
        parallel_state={"tp": 2, "pp": 1, "dp": 4},
    )
    ok = load_checkpoint(root, expected_parallel_state={"tp": 2}, strict=True)
    warn = load_checkpoint(root, expected_parallel_state={"tp": 4}, strict=False)
    print(ok["iteration"], warn["warnings"])
PY
```

关键输出应包含 iteration `10`，并在 non-strict 读取时返回一条 `tp` mismatch warning。

## 7. Megatron 源码对照

Megatron 的 `get_checkpoint_name` 会把 iteration、TP rank、PP rank 和 EP rank 编进路径。`get_checkpoint_tracker_filename` 返回 latest marker 路径，`read_metadata` 负责解析 marker，并在分布式环境中对 iteration 做一致性处理。

保存时，Megatron 的 `save_checkpoint` 会收集 RNG、rerun state、dataloader state、optimizer state 和 model state。`generate_state_dict` 把 model、optimizer、opt_param_scheduler、RNG 和 rerun state 组装成 state dict。使用 distributed checkpoint 时，`_build_sharded_state_dict_metadata` 会写入 distributed optimizer sharding type 和 DP/CP group，后续 optimizer 的 `sharded_state_dict` 根据 metadata 选择具体 sharding 实现。

加载时，Megatron 会读 checkpoint args，比较 checkpoint 中的 TP/PP 与当前运行 TP/PP。TP/PP mismatch 会影响 RNG 是否加载、distributed optimizer 是否支持当前 sharding type、rerun state 是否可用。L16 patch 的 `parallel_state` 校验就是这条主线的教学切片。

## 8. Drill：怎样判断一次 checkpoint smoke

`scripts/run_checkpoint_drill.py` 做四件事：保存一个 checkpoint，同拓扑 strict load，改变 TP strict load，改变 EP non-strict load。默认配置把保存拓扑设为 `{tp: 2, pp: 1, dp: 4, ep: 1}`，再检查 `tp_doubled` 和 `ep_introduced` 两个 mismatch case。

看结果时要记录：

- `artifacts/save_result.json`：checkpoint 文件和 marker 是否写出。
- `artifacts/drill.json`：每个 case 是否符合 strict / non-strict 预期。
- `metrics.jsonl`：每个 case 的 `case_ok`、`strict_raised` 和 warning 数。
- `config.resolved.yaml`：保存拓扑和 expected 拓扑是否就是本次要验证的条件。

如果 strict mismatch 没抛错，先检查 loader 是否真的遍历 expected keys。若 non-strict 没有 warning，检查是否把 mismatch 收集到 `warnings`。若 missing marker 没失败，检查 loader 是否绕过了 latest marker。

## 9. 小结

L16 的知识链路是：checkpoint 是训练恢复合同，payload 保存训练状态，latest marker 决定恢复入口，parallel_state 描述 shard 坐标系，strict load 防止错误 resume，non-strict load 服务诊断和转换。patch 验证最小 JSON 合同；Megatron 源码把同一套语义扩展到 rank 路径、sharded state dict、distributed optimizer metadata 和真实恢复流程。
