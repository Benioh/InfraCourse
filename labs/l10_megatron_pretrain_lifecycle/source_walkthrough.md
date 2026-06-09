# L11 源码带读：Train Step 与预训练生命周期

这份带读先读 patch，再读 MiniInfra lifecycle，最后对照真实 Megatron `training.py`。

## 0. 源码地图

```text
labs/l10_megatron_pretrain_lifecycle/patch/starter/train_step.py
labs/l10_megatron_pretrain_lifecycle/patch/reference/train_step.py
labs/l10_megatron_pretrain_lifecycle/patch/tests/test_patch.py
labs/l10_megatron_pretrain_lifecycle/scripts/run_lifecycle.py
mini_infra/megatron/pretrain_gpt.py
mini_infra/megatron/training/training.py
mini_infra/megatron/training/checkpointing.py
github_repo/Megatron-LM/megatron/training/training.py
```

## 1. 先读 patch starter

文件：`labs/l10_megatron_pretrain_lifecycle/patch/starter/train_step.py`

重点看：

1. L17-L24：`train_step` 输入。
2. L25-L31：forward/backward 和 optimizer 返回值合同。
3. L32-L38：TODO 列出的调用顺序、loss 校验、optimizer 结果解析、scheduler 和 metrics。

读完要能回答：这个函数返回什么？什么时候抛 `TrainStepError`？

## 2. 再读 patch reference

文件：`labs/l10_megatron_pretrain_lifecycle/patch/reference/train_step.py`

阅读顺序：

1. L13-L22：`_as_loss_list` 如何处理 `loss`、`losses` 和异常输入。
2. L25-L32：`_parse_step_result` 如何解释 optimizer 返回值。
3. L35-L41：`_current_lr` 的读取优先级。
4. L44-L63：`zero_grad`、forward/backward、loss 平均、optimizer、scheduler。
5. L65-L76：metrics 字段。

读完要能回答：optimizer 返回 `False`、`None` 和 dict 时有什么差异？

## 3. 读 patch tests

文件：`labs/l10_megatron_pretrain_lifecycle/patch/tests/test_patch.py`

重点看：

1. L16-L17：`IMPL` 选择 starter 或 reference。
2. L20-L31：DummyOptimizer 记录调用和返回 step result。
3. L34-L44：DummyScheduler 记录调用次数和 lr。
4. L47-L69：调用顺序、loss 平均、tokens、grad_norm、skip 和 lr。
5. L72-L87：optimizer skip 时 scheduler 不推进。
6. L90-L98：optimizer 返回 None。
7. L100-L109：异常输入。

读完要能回答：测试如何证明 scheduler 只在 update 成功后推进？

## 4. 读 MiniInfra task entry

文件：`mini_infra/megatron/pretrain_gpt.py`

重点看：

1. L6-L12：model provider 和 get_batch。
2. L15-L29：loss_func 和 forward_step。

读完要能回答：任务入口给通用训练循环提供了哪些回调？

## 5. 读 MiniInfra training lifecycle

文件：`mini_infra/megatron/training/training.py`

重点看：

1. L83-L116：`train_step` 的同构实现。
2. L125-L140：pretrain 创建 run_dir、初始化、数据、模型、optimizer 和 scheduler。
3. L146-L168：循环调用 train_step，补充并行和 optimizer 指标，并写 training log。
4. L169-L187：保存 checkpoint、resolved config、schedule 和 checkpoint artifact。

读完要能回答：本关 patch 与 MiniInfra training.py 的关系是什么？

## 6. 读 lifecycle drill

文件：`labs/l10_megatron_pretrain_lifecycle/scripts/run_lifecycle.py`

阅读顺序：

1. L44-L55：加载 student patch，必要时 fallback 到 reference。
2. L58-L84：加载 L09 scheduler，必要时使用本地 fallback。
3. L235-L243：构造 scheduler。
4. L252-L258：定义真实会 backward 的 forward_backward。
5. L268-L288：循环调用 train_step 并写 metrics。
6. L289-L293：写 checkpoint marker。
7. L294-L327：计算 acceptance 并写 `acceptance.json`。

读完要能回答：drill 能证明哪些生命周期证据？哪些结论仍需真实 Megatron 集群验证？

## 7. 读 checkpointing

文件：`mini_infra/megatron/training/checkpointing.py`

重点看：

1. L17-L43：保存 model、optimizer、scheduler、parallel state 和 latest marker。
2. L60-L85：读取 checkpoint 并检查 parallel state。

读完要能回答：resume 排查为什么必须同时看 scheduler 和 parallel state？

## 8. 读真实 Megatron training.py

文件：`github_repo/Megatron-LM/megatron/training/training.py`

重点看：

1. L829-L849：真实 `pretrain` 文档列出主流程。
2. L1888-L1900：真实 `train_step` 调用 `forward_backward_func`。
3. L1945-L1983：optimizer update 和 scheduler update 边界。
4. L3211-L3221：读取 canonical lr 并传入 `training_log`。

读完要能回答：真实代码在哪些位置对应本关的 forward/backward、optimizer、scheduler 和 metrics？

## 9. 读完后的自检问题

1. `train_step` 的输入和输出是什么？
2. 缺少 loss 与 optimizer 返回未知类型为什么要抛错？
3. `skipped_iter=1` 时 lr 应该怎样变化？
4. lifecycle drill 的 `acceptance.json` 能证明什么？
5. 哪些真实 Megatron 分支本关没有覆盖？
