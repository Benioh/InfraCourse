# L16 Source Reading Card：Megatron Parallel Checkpoint

## 主路径

1. `labs/l15_megatron_parallel_checkpoint/patch/starter/checkpointing.py`：最小 save/load 合同。
2. `labs/l15_megatron_parallel_checkpoint/patch/reference/checkpointing.py`：JSON payload、latest marker、parallel_state 校验。
3. `labs/l15_megatron_parallel_checkpoint/patch/tests/test_patch.py`：5 条 checkpoint 行为不变量。
4. `labs/l15_megatron_parallel_checkpoint/scripts/run_checkpoint_drill.py`：strict / non-strict topology mismatch drill。
5. `mini_infra/megatron/training/checkpointing.py`：教学版完整 checkpoint 实现。
6. `github_repo/Megatron-LM/megatron/training/checkpointing.py`：Megatron path、metadata、state_dict 和 load 主路径。
7. `github_repo/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py`：distributed optimizer sharded state dict。

## 关键行

| 文件 | 行 | 读完要得到的结论 |
|---|---|---|
| patch starter | L16-L40 | 学生要实现 save payload、marker、load 校验和 warning |
| patch reference | L28-L45 | payload 与 latest marker 是保存侧最小合同 |
| patch reference | L48-L84 | load 必须从 marker 进入，并比较 parallel_state |
| patch tests | L30-L72 | 测试覆盖 payload、roundtrip、strict、non-strict 和 missing marker |
| run_checkpoint_drill | L59-L123 | drill 把拓扑 mismatch 变成可落盘证据 |
| MiniInfra checkpointing | L17-L85 | 教学版完整实现和 patch 语义一致 |
| Megatron checkpointing | L197-L236 | checkpoint path 带 TP/PP/EP rank 坐标 |
| Megatron checkpointing | L980-L1055 | state_dict 包含 model、optimizer、scheduler 和 RNG |
| DistributedOptimizer | L1336-L1456 | optimizer sharding type 由 metadata 驱动 |

## 先跳过

- async checkpoint save：先掌握同步 save/load 语义。
- non-persistent local checkpoint：本讲不展开本地缓存和删除策略。
- FSDP DTensor planner：等本讲 metadata 与 L13 FSDP2 都读通后再看。
- ModelOpt 分支：属于额外 payload 扩展，不影响 marker 与 topology 主线。

## 自检

- 我能否解释 latest marker 为什么不能用文件名排序替代？
- 我能否说明 strict 和 non-strict load 的使用场景？
- 我能否指出 Megatron 在哪里把 TP/PP/EP rank 编进 checkpoint path？
- 我能否解释 optimizer sharding metadata 为什么会影响 load 行为？
