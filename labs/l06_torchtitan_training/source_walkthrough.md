# L07 源码带读：Selective Activation Checkpoint

这份带读按“starter 最小接口 -> reference -> tests -> TorchTitan apply_ac -> smoke 边界”的顺序读。目标是看清 activation checkpoint 的训练 step 语义，不把它和磁盘训练 checkpoint 混在一起。

## 1. 源码地图

| 文件 | 读什么 | 结论 |
|---|---|---|
| `patch/starter/selective_ckpt.py` | wrapper、policy 和 TODO | 学生只需补一层 child wrapping 和 attention policy |
| `patch/reference/selective_ckpt.py` | 最小正确实现 | 保存原 child，policy True 时 in-place setattr |
| `patch/tests/test_patch.py` | 行为合同 | 输出、梯度、policy count、no-op、GPU peak memory |
| `torchtitan/distributed/activation_checkpoint.py` | 生产 apply_ac | 支持 full/selective/memory-budget，多出 RNG、compile 和 op policy 边界 |
| `torchtitan/models/llama3/parallelize.py` | 系统调用位置 | AC 在 TP 后、compile 和 FSDP 前应用 |
| `scripts/run_torchtitan_stub.py` | 教学 smoke | 训练循环、metrics、磁盘 checkpoint 和 resume_delta |

## 2. 阅读顺序

### Step 1：读 starter 的 wrapper

文件：`labs/l06_torchtitan_training/patch/starter/selective_ckpt.py`

重点行：

- L21：`PolicyFn` 的类型。
- L24-L33：`_CheckpointWrapper` 保存原 module，并在 forward 调用 checkpoint。
- L36-L48：`selective_checkpoint_wrap` 的 TODO。
- L51-L55：`attention_only_policy` 的 TODO。

读完要得到的结论：

本关 wrapper 改的是 forward 保存策略，不复制参数。policy 的选择逻辑独立于 wrapper。

### Step 2：读 reference，确认最小实现

文件：`labs/l06_torchtitan_training/patch/reference/selective_ckpt.py`

重点行：

- L13-L19：wrapper 持有原 module，forward 使用 `use_reentrant=False`。
- L22-L26：遍历 `list(model.named_children())`，policy True 时 `setattr`。
- L29-L31：attention policy 匹配 `attn` 或 `attention`。

读完要得到的结论：

reference 很短。本关真正的不变量是数值等价、梯度等价、policy 计数和 no-op，而不是复杂框架配置。

### Step 3：读 tests，理解验收语义

文件：`labs/l06_torchtitan_training/patch/tests/test_patch.py`

重点行：

- L23-L57：测试模型包含 `attn0`、`mlp0`、`attn1`、`mlp1`。
- L65-L75：输出与裸模型 allclose。
- L78-L99：输入梯度和参数梯度与裸模型 allclose。
- L102-L114：attention child 被包，MLP child 未被包。
- L117-L123：全 False policy 不改变直接 child 类型。
- L126-L148：GPU 可用时比较 peak memory。

读完要得到的结论：

测试用 clone 保证初始参数一致。参数名允许因为 wrapper 多一层而变化，但梯度值必须对齐。

### Step 4：读 TorchTitan activation checkpoint

文件：`github_repo/torchtitan/torchtitan/distributed/activation_checkpoint.py`

重点行：

- L28-L87：`_get_save_ops` 选择哪些 op 必须保存，哪些可以重算。
- L90-L164：`_apply_op_sac` 构造 selective checkpoint context。
- L186-L202：full 和 selective 模式分发。
- L204-L255：`apply_ac` 遍历 transformer block 并注册 AC。

读完要得到的结论：

TorchTitan 的 selective AC 比本关更细：它能按 op policy 保存或重算，处理 compile、RNG、determinism、memory budget 等边界。本关只实现 child-level policy。

### Step 5：读 TorchTitan parallelize 顺序

文件：`github_repo/torchtitan/torchtitan/models/llama3/parallelize.py`

重点行：

- L35-L47：`parallelize_llama` 的输入包括 parallel dims、training、compile 和 AC config。
- L59-L74：CP 和 TP 先应用。
- L76-L90：AC 在 compile 前应用。
- L92-L102：随后进入 FSDP。

读完要得到的结论：

activation checkpoint 的位置会影响 compile 和 FSDP wrapping。生产框架把它放在固定顺序中，不是随便在训练 loop 里包一层。

### Step 6：读 smoke，分清训练 checkpoint

文件：`labs/l06_torchtitan_training/scripts/run_torchtitan_stub.py`

重点行：

- L37-L45：读取 toml 配置并写 resolved config。
- L50-L62：小模型训练循环。
- L63-L72：写 `metrics.jsonl`。
- L74-L94：保存并加载磁盘 checkpoint，计算 `resume_delta`。
- L96-L105：写 `framework_validation.json`。

读完要得到的结论：

stub 验证训练框架 artifact 边界。这里的 `checkpoint.pt` 是磁盘训练 checkpoint，与 activation checkpoint 是两个概念。

## 3. 读完后的自检问题

1. `_CheckpointWrapper` 保存的 `self.module` 为什么必须是原 child？
2. `selective_checkpoint_wrap` 为什么只遍历 `named_children()`？
3. policy 全 False 时，模型应保持什么不变？
4. GPU peak memory test skip 时，报告应该怎样写？
5. TorchTitan 的 `apply_ac` 在 `parallelize_llama` 中位于 compile 和 FSDP 的哪一侧？
