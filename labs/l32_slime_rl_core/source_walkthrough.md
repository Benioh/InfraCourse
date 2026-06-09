# 源码带读：L36 SLiME Weight Sync Coordinator

这份带读按“patch contract -> MiniInfra loop -> SLiME production path -> smoke”的顺序走。目标是先理解本地 state_dict 同步合同，再把同样的检查放进 Ray、Megatron updater 和 SGLang engine 的真实链路。

## 0. 源码地图

```text
labs/l32_slime_rl_core/patch/starter/weight_sync.py
labs/l32_slime_rl_core/patch/reference/weight_sync.py
labs/l32_slime_rl_core/patch/tests/test_patch.py

mini_infra/slime/train.py
mini_infra/slime/ray/train_actor.py
mini_infra/slime/ray/rollout.py

github_repo/slime/train.py
github_repo/slime/slime/ray/actor_group.py
github_repo/slime/slime/backends/megatron_utils/actor.py
github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py
github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
github_repo/slime/slime/backends/sglang_utils/sglang_engine.py

labs/l32_slime_rl_core/scripts/run_slime_lab.py
```

## 1. Patch contract

文件：[patch/starter/weight_sync.py](patch/starter/weight_sync.py)

第 20-33 行定义 `WeightSyncCoordinator` 的初始化参数。读完要能说清 train provider、inference setter 和 inference provider 的职责。

第 35-45 行写出 sync 的返回格式和规则。第 46-74 行是伪代码：取 state、筛选 accepted tensors、记录 mismatch、统计 bytes。

文件：[patch/reference/weight_sync.py](patch/reference/weight_sync.py)

第 21-33 行是无 inference provider 的简化模式：全量 clone 后调用 setter。第 35-49 行是 shape/dtype gate。第 51-56 行返回 stats。

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

按顺序读：

- 第 32-49 行：基本同步和 `num_tensors`。
- 第 51-66 行：shape mismatch 不覆盖 inference 旧值。
- 第 69-80 行：dtype mismatch 不自动 cast。
- 第 83-95 行：train 多余 key 不创建到 inference。
- 第 98-110 行：bytes 和 tensor 数统计。

## 2. MiniInfra SLiME loop

文件：[mini_infra/slime/train.py](../../mini_infra/slime/train.py)

第 12-19 行创建 rollout manager 和 actor，并按 generate、train、update_weights 推进。第 20-29 行把 rollout id、reward 和 weight version 写进 history。

文件：[mini_infra/slime/ray/train_actor.py](../../mini_infra/slime/ray/train_actor.py)

第 44-53 行模拟 actor train：有 response 时 reward_mean 为 1，随后递增 `weight_version`。第 55-58 行把新版本写回 rollout manager。

文件：[mini_infra/slime/ray/rollout.py](../../mini_infra/slime/ray/rollout.py)

第 27-49 行生成 response 和 meta_info，其中包含 rollout server 的 `weight_version` 和传入的 `actor_version`。第 67-82 行选择 fresh enough 的 server。第 84-93 行执行 rollout servers 的 weight version 更新。

## 3. SLiME train path

文件：[github_repo/slime/train.py](../../github_repo/slime/train.py)

第 15-30 行创建 rollout manager、actor/critic，并在权重加载后初次 `actor_model.update_weights()`。第 66-85 行展示训练 loop 中 generate 和 actor/critic train 的顺序。第 90-99 行展示 offload rollout 场景下的 onload、update_weights 和 eval。

文件：[github_repo/slime/slime/ray/actor_group.py](../../github_repo/slime/slime/ray/actor_group.py)

第 111-129 行把训练请求发给每个 Ray actor。第 135-138 行对每个 train actor 调用 `update_weights.remote()`。

文件：[github_repo/slime/slime/ray/train_actor.py](../../github_repo/slime/slime/ray/train_actor.py)

第 109-119 行抽象出 `train`、`save_model` 和 `update_weights`。第 125-128 行把 rollout manager 引用交给 train actor。

## 4. Megatron updater path

文件：[github_repo/slime/slime/backends/megatron_utils/actor.py](../../github_repo/slime/slime/backends/megatron_utils/actor.py)

第 127-134 行根据 `args.colocate` 选择 `UpdateWeightFromTensor` 或 `UpdateWeightFromDistributed`。第 538-550 行进入 `update_weights`，并从 rollout manager 获取可更新 engines 和 lock。第 555-569 行连接 engines 并调用 `weight_updater.update_weights()`。第 571-577 行在 CI 模式校验 engine weight version。

文件：[github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py](../../github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py)

第 20-34 行定义 distributed updater 的输入。第 45-53 行连接 rollout engines 并说明 group 名称。第 81-99 行 pause generation、flush cache，并进入同步前 barrier。第 142-164 行展示非 expert 参数的 gather、buffer 和 HF 转换。第 190-226 行展示 expert 参数的 EP gather 和 bucket update。第 310-337 行用 Ray 发 metadata，用 NCCL broadcast tensor。

文件：[github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py](../../github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py)

第 24-30 行说明 tensor updater 覆盖 co-locate 和 distributed 的传输差异。第 61-70 行开始连接 rollout engines，并计算 colocated/distributed engines。

## 5. SGLang engine side

文件：[github_repo/slime/slime/backends/sglang_utils/sglang_engine.py](../../github_repo/slime/slime/backends/sglang_utils/sglang_engine.py)

第 266-289 行是 `update_weights_from_tensor`，HTTP 只发 metadata，真实 tensor 通过 GPU 路径复制。第 349-355 行读取 engine 的 `weight_version`。第 373-380 行是 disk update 入口，适合冷路径或开发调试。

## 6. 本地 smoke

文件：[scripts/prepare_gsm8k_slime.py](scripts/prepare_gsm8k_slime.py)

第 7-18 行在 prompts 缺失时调用 L29 的 toy GSM8K 数据脚本。第 32-40 行写出带 `source=slime` 的 `slime_prompts.jsonl`。

文件：[scripts/run_slime_lab.py](scripts/run_slime_lab.py)

第 28-41 行读取配置、创建 run 目录和 resolved config。第 42-58 行准备 prompts，检查 `slime` / `sglang` 可导入性，并写 validation artifact。第 59-72 行写 actor/rollout 时间和 sync 时间指标。第 77-103 行写报告并声明本地边界。

## 可以先跳过

- Megatron 的完整 checkpoint 保存和 optimizer state。
- SGLang engine 的 shutdown、profiling 和 frozen model reload 分支。
- UpdateWeightFromTensor 的 IPC gather 细节，先理解它和 distributed updater 的路径差异。

## 自检问题

1. 本地 patch 的 mismatch 检查对应生产路径里的哪些风险？
2. MiniInfra 的 `weight_version` 在 generate、train、update 之间如何变化？
3. SLiME 在哪里根据 co-locate / disaggregate 选择不同 updater？
4. distributed updater 为什么先发 names、dtypes、shapes，再 broadcast tensors？
5. 本地 smoke 不能证明哪些 H200/NCCL 能力？
