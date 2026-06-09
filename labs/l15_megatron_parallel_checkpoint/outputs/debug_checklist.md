# L16 Debug Checklist：Megatron Parallel Checkpoint

## 1. 先固定现场

- 记录命令、配置文件、run id、git commit、Python 环境、节点/GPU 数和并行配置。
- 记录保存时和加载时的 `tp`、`pp`、`dp`、`ep`、`cp`、world size、checkpoint format。
- 保存 `latest_checkpointed_iteration.txt`、checkpoint payload、`artifacts/drill.json`、`metrics.jsonl` 和 stdout/stderr。
- 明确当前是 patch-test、CPU drill、Megatron dryrun、finetune 权重加载，还是真实 resume。

## 2. 先看恢复入口

| 检查项 | 期望 | 异常时先看 |
|---|---|---|
| latest marker | 文件存在，内容是 iteration 或 release | save 是否在文件写完后更新 marker |
| checkpoint 文件 | marker 指向的文件或目录存在 | iteration 格式和路径拼接 |
| format | payload format 与 loader 支持的格式一致 | 是否混用了旧课程或旧版本 checkpoint |
| iteration | payload iteration 与 marker 一致 | 保存中断或手动改文件 |
| warnings | non-strict mismatch 有明确 warning | loader 是否吞掉 mismatch |

## 3. 再看状态完整性

| 状态 | 缺失时的风险 | 证据 |
|---|---|---|
| model_state | 无法恢复模型权重 | payload key 或 sharded state dict |
| optimizer_state | 下一步更新方向和尺度错位 | Adam state、distributed optimizer metadata |
| scheduler_state | LR 从错误 step 继续 | step_count、last_lr、opt_param_scheduler |
| parallel_state | shard 坐标系无法判断 | TP/PP/DP/EP/CP 字段 |
| RNG / data progress | dropout 和数据顺序不连续 | Megatron state_dict 中的 rng_state、rerun state、dataloader state |

## 4. 常见故障

| 现象 | 可能原因 | 下一步 |
|---|---|---|
| missing marker | save 没写 marker 或目录不对 | 检查 save_result 和 checkpoint_dir |
| unsupported format | loader 与 checkpoint format 不匹配 | 看 payload `format` 和 Megatron `ckpt_format` |
| TP mismatch | 当前 TP 与保存 TP 不同 | strict resume 停止，考虑 checkpoint 转换或 finetune |
| EP introduced warning | 新拓扑多了 expert parallel 维度 | 检查 expert shard 是否可转换 |
| resume 后 LR 跳变 | scheduler_state 或 opt_param_scheduler 没恢复 | 查 step_count、last_lr 和训练日志 |
| optimizer shape mismatch | distributed optimizer sharding metadata 不兼容 | 查 `distrib_optim_sharding_type` 和 TP/PP/DP |

## 5. 沿源码主路径复查

1. `labs/l15_megatron_parallel_checkpoint/patch/reference/checkpointing.py`：最小 payload、marker 和 topology 校验。
2. `labs/l15_megatron_parallel_checkpoint/scripts/run_checkpoint_drill.py`：same topology、TP mismatch 和 EP mismatch 的证据。
3. `mini_infra/megatron/training/checkpointing.py`：教学版完整 save/load。
4. `github_repo/Megatron-LM/megatron/training/checkpointing.py`：真实 path、marker、metadata 和 load 逻辑。
5. `github_repo/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py`：distributed optimizer sharded state dict。

## 6. 结束条件

- 最小复现命令可以稳定重跑。
- 能说明失败发生在 marker、format、parallel_state、state_dict 生成、optimizer shard 或真实 tensor load 哪一层。
- strict / non-strict 行为和当前工作流一致。
- 结论写入 `checkpoint_debug_template.md`，并明确下一步是转换 checkpoint、丢弃 optimizer、修复 marker，还是停止 resume。
