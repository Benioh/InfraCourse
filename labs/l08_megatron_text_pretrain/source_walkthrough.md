# L09 源码带读：训练 step、Scheduler 与日志闭环

这份带读按调用链组织。不要从每个文件顶部顺序扫到底，先用下面的路径建立主线，再回头看生产分支。

## 0. 源码地图

```text
mini_infra/megatron/training/training.py
github_repo/Megatron-LM/megatron/core/optimizer_param_scheduler.py
github_repo/Megatron-LM/megatron/training/training.py
labs/l08_megatron_text_pretrain/patch/starter/lr_scheduler.py
labs/l08_megatron_text_pretrain/patch/reference/lr_scheduler.py
labs/l08_megatron_text_pretrain/patch/tests/test_patch.py
labs/l08_megatron_text_pretrain/scripts/run_train.py
labs/l08_megatron_text_pretrain/scripts/parse_megatron_log.py
```

## 1. 先读 MiniInfra 训练 step

文件：`mini_infra/megatron/training/training.py`

重点看：

- `MiniLRScheduler`
- `_current_lr`
- `train_step`
- `pretrain` 保存 scheduler state 的位置

阅读顺序：

1. L29-L46：看教学 scheduler 保存哪些状态，怎样写 param group。
2. L74-L80：看 metrics 读取当前 lr 的 fallback 顺序。
3. L83-L116：看一次 step 的顺序，尤其是 optimizer 成功后才调用 scheduler。
4. L169-L175：看 checkpoint 保存 scheduler state。

读完要能回答：如果 optimizer step 被跳过，scheduler 是否应该推进？如果 checkpoint 缺 scheduler state，resume 后会发生什么？

可以先跳过：

- pipeline schedule 的具体事件结构。
- toy data 的文本采样细节。
- DistributedOptimizer 的内部状态估算。

## 2. 再读 Megatron scheduler 组件

文件：`github_repo/Megatron-LM/megatron/core/optimizer_param_scheduler.py`

重点看：

- `OptimizerParamScheduler.__init__`
- `get_lr`
- `step`
- `state_dict`
- `load_state_dict`
- `get_canonical_lr_for_logging`

阅读顺序：

1. L125-L160：看构造函数保存哪些 schedule 参数和计数状态。
2. L218-L282：看 warmup、constant、inverse-square-root、linear、cosine 和 WSD 的分支。
3. L284-L300：看 `step(increment)` 怎样写所有 param groups。
4. L302-L316：看 scheduler state 里保存哪些字段。
5. L341-L352：看加载 checkpoint 时怎样对齐配置值。
6. L40-L57：看日志为什么需要 canonical lr。

读完要能回答：Megatron 为什么用 increment 推进 scheduler？param group override 会改变哪些字段？日志层为什么不能随便读一个局部变量？

可以先跳过：

- weight decay 的完整分支。
- WSD 的所有 decay style 细节。
- 与旧 checkpoint key 兼容的长分支，先记住 `load_state_dict` 负责恢复边界。

## 3. 对照 Megatron 训练 loop

文件：`github_repo/Megatron-LM/megatron/training/training.py`

重点看：

- `get_optimizer_param_scheduler`
- `train_step`
- `training_log`
- 主训练 loop 中 learning rate 的读取

阅读顺序：

1. L1546-L1599：看 args 怎样构造成 `OptimizerParamScheduler`。
2. L1945-L1983：看 optimizer update 成功后怎样计算 increment 并推进 scheduler。
3. L2118-L2128：看 TensorBoard / wandb 写 learning rate 的入口。
4. L2308-L2311：看 stdout log string 中的 learning rate。
5. L3211-L3221：看主 loop 从 optimizer param groups 取 canonical lr，再传给 `training_log`。

读完要能回答：训练日志里的 learning rate 来自哪里？`skipped_iter` 时 scheduler 是否推进？sample-based schedule 和 global batch 有什么关系？

可以先跳过：

- activation logging、TPE logging、vision pretraining 分支。
- pipeline shape 调整和 modelopt distillation 分支。
- straggler、energy monitor 等观测扩展。

## 4. 读 patch starter

文件：`labs/l08_megatron_text_pretrain/patch/starter/lr_scheduler.py`

重点看：

- `__init__` 中 boundaries 的 TODO。
- `_set_lr` 写 param groups 的 TODO。
- `_compute_lr` 中 total clamp、segment 查找和 cosine 公式的 TODO。
- `step` 和 `get_lr` 的调用边界。

阅读顺序：

1. L33-L46：看输入参数和 `step_count`。
2. L48-L51：看 boundaries 的预期形式。
3. L56-L58：看 lr 必须写入所有 param groups。
4. L60-L75：看 `_compute_lr` 的计算步骤。
5. L77-L85：看每次 `step()` 后如何写回 optimizer，并由 `get_lr()` 读出。

读完要能回答：restart step 属于旧段还是新段？空 restart 列表怎样处理？total 之后为什么直接返回 `min_lr`？

## 5. 读 patch reference

文件：`labs/l08_megatron_text_pretrain/patch/reference/lr_scheduler.py`

重点看：

- L26-L27：boundaries 和初始 lr。
- L29-L31：所有 param groups 的同步写入。
- L33-L49：total clamp、segment 查找、segment_len 防御和 cosine 公式。
- L51-L56：`step()` 和 `get_lr()`。

读完要能回答：reference 的 `segment_len <= 0` 分支在防什么？为什么 `step_count` 先加 1 再计算新 lr？

## 6. 读 patch tests

文件：`labs/l08_megatron_text_pretrain/patch/tests/test_patch.py`

重点看：

- L18-L20：`IMPL` 环境变量如何选择 starter 或 reference。
- L28-L33：初始 lr。
- L36-L45：段内单调下降。
- L48-L55：restart step。
- L58-L64：total clamp。
- L67-L76：多 param group 同步。
- L88-L97：无 restart 的单段 cosine。

读完要能回答：这 7 个测试覆盖了哪些合同？哪些生产问题仍然没有覆盖？

## 7. 读 drill 脚本和日志解析

文件：

- `labs/l08_megatron_text_pretrain/scripts/run_train.py`
- `labs/l08_megatron_text_pretrain/scripts/parse_megatron_log.py`

`run_train.py` 阅读顺序：

1. L45-L52：读取配置并写出 resolved config。
2. L54-L72：检查 JSONL、indexed dataset 和 Megatron runtime。
3. L80-L90：fallback artifact 和 expected command。
4. L93-L106：写 metrics。
5. L107-L147：写 report。

`parse_megatron_log.py` 阅读顺序：

1. L12-L28：要抽取哪些日志字段。
2. L46-L68：怎样解析文本并标记缺失字段。
3. L71-L86：self-test 的期望输出。

读完要能回答：fallback 能证明什么？真实训练日志缺少 `iteration`、`lm_loss`、`tokens/sec` 时，为什么只能给出弱结论？

## 8. 读完后的自检问题

1. 你能否从 `--data-path` 画到 `training_log` 和 checkpoint？
2. `optimizer.step()` 返回失败时，scheduler 为什么不应该推进？
3. Megatron 的 scheduler increment 和本关 patch 的 `step_count += 1` 有什么差异？
4. 哪些测试证明了 restart 边界，哪些测试证明了 param group 输出？
5. 如果恢复训练后 loss 突然抖动，你会先查哪三个 artifact？
