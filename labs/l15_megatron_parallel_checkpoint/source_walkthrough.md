# L16 源码带读：Megatron Parallel Checkpoint

这份带读只走 checkpoint resume 主路径。第一次读时先跳过异步保存、非持久化本地 checkpoint、ModelOpt、FSDP DTensor 的细节。目标是看清：checkpoint 路径怎样带上 rank 坐标，latest marker 怎样定位 iteration，state_dict 包含哪些训练状态，distributed optimizer metadata 怎样影响 optimizer shard load。

## 1. 源码地图

```text
labs/l15_megatron_parallel_checkpoint/patch/starter/checkpointing.py
  -> 学生要补齐的 save/load 合同

labs/l15_megatron_parallel_checkpoint/patch/reference/checkpointing.py
  -> JSON payload、latest marker、parallel_state 校验

labs/l15_megatron_parallel_checkpoint/patch/tests/test_patch.py
  -> L16 patch 的行为合同

labs/l15_megatron_parallel_checkpoint/scripts/run_checkpoint_drill.py
  -> same topology / TP mismatch / EP mismatch 的 drill

mini_infra/megatron/training/checkpointing.py
  -> 教学版完整 save/load 实现

github_repo/Megatron-LM/megatron/training/checkpointing.py
  -> 真实 Megatron checkpoint path、metadata、state_dict、load 主路径

github_repo/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py
  -> distributed optimizer sharded state dict
```

## 2. 阅读步骤一：先看 patch starter 的合同

文件：`labs/l15_megatron_parallel_checkpoint/patch/starter/checkpointing.py`

重点：

- L9-L13：format 常量和 `CheckpointError` 是 loader 拒绝坏 checkpoint 的基础。
- L16-L24：`save_checkpoint` 的输入包含四类 state 和 iteration。
- L25-L28：save TODO 要求写 payload、checkpoint 文件和 latest marker。
- L31-L40：load TODO 要求读 marker、校验 format、比较 parallel_state，并处理 strict / non-strict。

读完要能回答：为什么 loader 不能直接从目录里随便找一个 JSON 文件？

## 3. 阅读步骤二：对照 reference 的 save 主路径

文件：`labs/l15_megatron_parallel_checkpoint/patch/reference/checkpointing.py`

重点：

- L17-L24：save 函数的参数就是恢复合同的最小输入。
- L25-L27：负 iteration 直接拒绝，并创建输出目录。
- L28-L35：payload 同时保存 format、iteration、model、optimizer、scheduler 和 parallel state。
- L36-L45：写 `iter_XXXXXXX.json`，再写 latest marker，并返回两个路径。

读完要能说明：`scheduler_state` 和 `parallel_state` 分别防止哪类 resume 错误？

## 4. 阅读步骤三：对照 reference 的 load 主路径

文件：`labs/l15_megatron_parallel_checkpoint/patch/reference/checkpointing.py`

重点：

- L48-L59：`_latest_checkpoint_path` 从 marker 解析 iteration，并确认文件存在。
- L62-L70：load 读取 payload 并拒绝未知 format。
- L72-L80：逐个比较 expected parallel state，strict mismatch 立即抛错。
- L82-L84：返回 payload 时补上 checkpoint_path 和 warnings。

读完要能解释：strict load 和 non-strict load 的使用场景为什么不同。

## 5. 阅读步骤四：用测试反推不变量

文件：`labs/l15_megatron_parallel_checkpoint/patch/tests/test_patch.py`

重点：

- L30-L42：save 后必须写正确文件名、latest marker、format、iteration 和 scheduler state。
- L44-L51：同拓扑 load 应该 roundtrip，warnings 为空。
- L54-L58：strict TP mismatch 必须抛 `CheckpointError`。
- L61-L66：non-strict TP mismatch 必须返回 warning。
- L69-L72：缺少 latest marker 必须失败。

读完要能判断：如果 loader 绕过 latest marker，哪条测试最容易漏掉，真实系统会留下什么风险？

## 6. 阅读步骤五：看 drill 怎样覆盖拓扑 mismatch

文件：`labs/l15_megatron_parallel_checkpoint/scripts/run_checkpoint_drill.py`

重点：

- L49-L57：选择实现、读取配置、创建 run 目录并写 resolved config。
- L59-L69：保存 checkpoint，并把 save_result 写入 artifacts。
- L71-L81：为每个 expected topology case 建立 outcome。
- L82-L92：同一个 case 分别跑 strict 和 non-strict load。
- L94-L106：根据 strict 是否抛错和 warning 数判断 case 是否通过。
- L108-L118：把每个 case 写入 `metrics.jsonl`。
- L120-L123：把整体 drill 结果写入 `artifacts/drill.json`。

读完要能说明：CPU drill 的 `all_passed=true` 能证明什么，不能证明什么。

## 7. 阅读步骤六：MiniInfra 完整实现

文件：`mini_infra/megatron/training/checkpointing.py`

重点：

- L17-L24：教学版 save 接收同样的四类 state。
- L28-L36：payload 增加 `created_at`，其余字段和 patch 语义一致。
- L37-L43：写 checkpoint 文件和 latest marker。
- L46-L57：读 marker、解析 iteration、确认文件存在。
- L60-L68：读取 payload，并校验 format。
- L70-L85：比较 parallel_state，strict 抛错，non-strict 返回 warnings。

读完要能说明：patch reference 和 MiniInfra 实现的差异在哪里，语义是否相同。

## 8. 阅读步骤七：Megatron checkpoint path 与 marker

文件：`github_repo/Megatron-LM/megatron/training/checkpointing.py`

重点：

- L197-L206：`get_checkpoint_name` 先根据 iteration 或 release 选择目录名。
- L211-L221：未传 rank 时从 parallel state 查询 TP、PP、EP rank。
- L223-L236：checkpoint 路径把 TP、PP、EP rank 编进去。
- L312-L317：latest tracker 文件名固定为 `latest_checkpointed_iteration.txt`。
- L326-L345：`read_metadata` 解析 marker，支持 iteration 或 release。
- L348-L368：分布式已初始化时跨 rank 对齐 iteration；离线编辑时直接使用本地 iteration。

读完要能回答：为什么 Megatron 的 checkpoint 路径里需要 TP/PP/EP rank？

## 9. 阅读步骤八：Megatron state_dict 与 optimizer metadata

文件：`github_repo/Megatron-LM/megatron/training/checkpointing.py`

重点：

- L417-L450：`_build_sharded_state_dict_metadata` 写入 distributed optimizer sharding type 和 DP/CP group。
- L598-L619：save 阶段调用 `generate_state_dict` 收集模型、optimizer、scheduler、RNG 和 rerun state。
- L770-L792：保存完成后更新 latest marker。
- L980-L999：`generate_state_dict` 写入 args、checkpoint_version 和 iteration。
- L1000-L1016：模型 state dict 根据 checkpoint format 选择 sharded 或普通格式。
- L1018-L1045：optimizer 和 opt_param_scheduler 都进入 state dict。
- L1051-L1054：RNG state 也会保存。

读完要能说明：真实 resume 比 L16 patch 多保存了哪些状态。

## 10. 阅读步骤九：DistributedOptimizer 的 sharded_state_dict

文件：`github_repo/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py`

重点：

- L1336-L1342：`sharded_state_dict` 接收 model sharded state、loading 标记、sharding type 和 metadata。
- L1343-L1361：docstring 说明 dp_reshardable、fully_reshardable 和 fsdp_dtensor 三类主要格式。
- L1374-L1387：metadata 决定 optimizer sharding type 的默认选择。
- L1408-L1424：部分格式会把普通 optimizer state 包成 `ShardedObject`。
- L1431-L1456：根据 sharding type 选择 param_state 构造方式，并写回 state_dict。

读完要能说明：为什么 optimizer state load 比 model weight load 更依赖 metadata。

## 11. 可以先跳过的分支

- async checkpoint save：本讲只讲同步语义，异步只改变落盘时机和 finalize。
- non-persistent local checkpoint：涉及本地 SSD/ramdisk 策略，不改变 resume 合同。
- ModelOpt 和 distillation 分支：它们扩展 payload，不改变 marker 与 topology 校验主线。
- FSDP DTensor 具体 planner：等理解 L13 FSDP2 和本讲 metadata 后再读。

## 12. 自检问题

1. latest marker 缺失时，loader 为什么应该失败？
2. `strict=False` 返回 warnings 后，调用方下一步应该做什么？
3. Megatron `generate_state_dict` 比 patch payload 多了哪些状态？
4. distributed optimizer 的 sharding type 为什么会影响 TP/PP mismatch 是否可加载？
5. 如果 resume 后 LR 跳变，应该先检查 payload 的哪个字段？
