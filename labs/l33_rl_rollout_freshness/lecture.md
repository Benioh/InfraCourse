# L38：SLiME Rollout Freshness and Versioned RolloutManager

L36/L37 已经把 actor 权重如何交给 rollout engine 讲清楚。L38 继续问下一步：权重同步完成以后，训练侧怎样证明下一批 rollout 样本来自足够新的 policy。RL 训练里，actor 每训练一步都会推进参数版本；rollout engine 可能因为同步频率、局部更新、offload/onload 或故障恢复而落后。这个差值如果没有记录，reward、KL、advantage 和吞吐曲线都会少一层关键上下文。

本讲要实现一个 versioned `RolloutManager`。每个 `RolloutServer` 有自己的 `weight_version`；每次生成时，manager 接收当前 `actor_version`，计算 `staleness = actor_version - server.weight_version`，只选择足够新的 server。如果一圈 server 都过旧，系统要抛 `RuntimeError`，让 stale rollout 在训练前暴露。

## 1. 本讲目标

- 解释 rollout freshness 指什么，为什么它和 on-policy 程度相关。
- 说清 `actor_version`、`weight_version`、`staleness` 和 `server_id` 四个字段的作用。
- 读懂 MiniInfra、patch reference、SLiME rollout manager、Sample meta_info 和 SGLang weight version 的源码落点。
- 完成 L38 patch，并说明每个测试覆盖的行为合同。
- 使用 drill 输出复盘 update 频率、局部更新范围和 `max_staleness` 的取舍。

## 2. 从一个故障场景进入

假设一个 RL 任务每轮先生成 rollout，再训练 actor。第 20 轮训练完成后，actor 已经更新到版本 20，但四台 rollout engine 里只有两台完成了权重更新，另外两台还停在版本 17。如果下一轮 manager 随机挑中旧 engine，样本格式仍然正常，reward parser 也可能正常返回分数；问题是这些 token 来自旧 policy。

这个差距会影响训练信号解释。PPO/GRPO 会用 rollout 样本、old logprob、reward、KL 或 group-level reward 统计更新当前 actor。rollout policy 越旧，样本越偏离当前 actor，ratio 更容易被 clip，KL 变化也更难和当前参数更新对应。训练日志里看到 reward 下降时，不能只问 reward parser 是否坏了，也要问样本来自哪一版 policy。

L38 的核心不是追求所有 server 永远同步到最新版本。真实系统里，同步频率和吞吐之间有成本。它的目标是把版本差写成显式合同：允许落后多少步，超过阈值怎么拒绝，样本里留下哪些证据。

## 3. 四个版本字段

`actor_version` 是训练侧当前 policy 的版本。MiniInfra 里 `TrainRayActor.train` 每训练一次就递增这个值，真实 SLiME 里则由 actor updater 和 rollout engine 的 weight version 共同体现。

`weight_version` 是 rollout server 实际持有的权重版本。一个 server 刚被 `update_weights(version)` 更新后，应该把自己的 `weight_version` 改成这个版本。

`staleness` 是两者的差：

```text
staleness = actor_version - server.weight_version
```

`server_id` 用来定位是哪一台 server 产出了样本。没有 `server_id`，只能看到全局 staleness 分布，无法判断是否有固定子集长期落后。

这四个字段进入 `RolloutData.meta_info` 后，训练侧才有证据判断样本来源。它们不替代 reward、KL 或吞吐指标，而是给这些指标加上版本上下文。

## 4. Versioned Manager 的最小合同

L38 patch 里的 `RolloutManager` 有三个公开方法。

`generate(prompts, actor_version)` 从 `next_server_index` 开始按 round-robin 顺序尝试每个 server。对候选 server 计算 `actor_version - server.weight_version`，只要差值小于等于 `max_staleness`，就调用该 server 的 `generate`。成功后把 `next_server_index` 移到下一个位置，避免所有流量长期打到同一台 server。遍历一圈都没有 fresh server 时抛 `RuntimeError("no rollout server is fresh enough")`。

`update_weights(version, server_ids=None)` 负责推进 server 的版本。`server_ids is None` 表示更新全部 server；传入列表时，只更新命中的 server，并返回实际更新的 id。这个子集语义很重要，因为真实集群里可能只恢复了一部分 engine，或者某些 server 属于 reference/reward 这类 frozen model。

`freshness(actor_version)` 返回 `{server_id: actor_version - weight_version}`。它属于诊断视图。drill 和 debug 报告应该把它和 failure、p50/p99 staleness、reward、KL 放在一起看。

## 5. MiniInfra 如何简化真实系统

MiniInfra 的 `RolloutServer.generate` 会把 `weight_version`、`actor_version` 和 `staleness` 写进 `meta_info`。它还接了一个极简 scheduler，用来保留 serving 路径的形状，但 L38 不讨论 KV cache 或 batch admission。这里要抓住的是版本证据从 server 进入 rollout data 的位置。

MiniInfra 的 `RolloutManager.generate` 和 patch reference 同构：保存 server 列表、阈值和 round-robin 指针；每次 generate 都按 freshness 筛选；没有可用 server 就失败。`TrainRayActor.update_weights` 会把 actor 当前版本写回 rollout manager，这让一段最小训练循环也能产生版本变化。

这种简化有边界。MiniInfra 不启动 Ray，不连接 SGLang HTTP endpoint，不做 tensor broadcast，不处理 engine crash。它提供的是控制面模型：版本如何存、如何比、如何拒绝旧副本。

## 6. SLiME 生产路径

真实 SLiME 的训练脚本先创建 rollout manager，再创建 actor/critic。初始化阶段会把 actor 权重推到 rollout，训练循环中每轮先 `rollout_manager.generate.remote(rollout_id)`，再训练 actor，必要时执行 offload/onload，然后调用 `actor_model.update_weights()` 把新权重交给 rollout engine。

SLiME 的 rollout manager 包含多个 `RolloutServer`，每个 server 又可以包含多个 SGLang engine group。`get_updatable_engines_and_lock` 会选择 `update_weights=True` 的模型；reference、reward 或其他 frozen model 不应该被 actor 权重覆盖。engine lock 则避免同步时和 generation、fault recovery 产生冲突。

Sample 侧也有版本证据。`Sample` 类型包含 `weight_versions`，`update_from_meta_info` 会从 rollout engine 返回的 `meta_info` 中抽取 `weight_version`。SGLang engine 的 update 接口可以携带 `weight_version`，并提供 `get_weight_version()` 查询当前版本。L38 patch 把这些生产事实压缩成一个本地合同，方便先测试清楚。

## 7. Drill 怎么读

`run_rl_drill.py` 创建若干 server，每一步把 actor version 当成 step 号。脚本按 `update_rate` 概率选择一部分 server 更新到当前版本，再调用 `manager.generate(prompts, actor_version=step)`。成功时记录 staleness，失败时记录 failure。

脚本会跑两个场景：

- `nominal`：按配置中的 `update_rate` 和 `update_subset_size` 更新 server。
- `no_update`：完全不更新，用来验证 stale-safe 合同会在 server 全部过旧时失败。

CPU smoke 默认 4 个 server、50 step、`max_staleness=1`、`update_rate=0.5`、每次更新 2 个 server。它不是 GPU benchmark；它只检查版本准入逻辑和证据格式。H200 模板会写出 SLiME actor/rollout 的启动命令，但仍需要真实集群运行才能验证 NCCL、SGLang 和长训行为。

## 8. 参数取舍

`max_staleness` 越小，样本越接近当前 actor，但系统更容易因为没有 fresh server 而失败。`max_staleness` 越大，成功率可能提高，但样本更旧，reward/KL 解释要更谨慎。

`update_rate` 控制同步发生的频率。频率提高会让 fresh server 变多，也会增加同步和等待成本。`update_subset_size` 控制每次覆盖多少 server。覆盖范围太小会让部分 server 长期落后；覆盖范围太大则可能拉高单次同步压力。

看这些参数时，要同时记录 failure、successful_rollouts、p50/p99 staleness、weight sync 时间和 step time。单看 reward 或单看吞吐，都会漏掉版本层面的解释。

## 9. 测试闭环

五个 patch tests 固定 L38 的最小合同。

`test_generate_uses_fresh_server_and_records_meta` 检查正常 generate 的 response 与 `meta_info`，确保版本字段完整。`test_generate_skips_stale_server_round_robin` 构造一个 stale server 和一个 fresh server，要求 manager 跳过 stale，并推进指针。`test_generate_raises_when_all_servers_stale` 要求没有 fresh server 时抛错。

`test_update_weights_all_or_subset` 验证局部更新不会误改其他 server，再验证全量更新会覆盖所有 server。`test_freshness_reports_actor_minus_engine_version` 直接校验 freshness 公式。

这些测试不覆盖真实 Ray、SGLang endpoint、NCCL、engine lock 或多轮长训稳定性。它们的价值是先把版本准入合同固定住，再带着这个合同去读真实系统。

## 10. 排障顺序

遇到 RL 指标异常时，先固定版本现场。

1. 记录命令、配置、run id、server 列表、`max_staleness`、`update_rate` 和 `update_subset_size`。
2. 检查每条样本或每个 run 的 `actor_version`、`weight_version`、`staleness` 和 `server_id`。
3. 对比 failure、p50/p99 staleness、successful_rollouts 和 no-update 场景是否符合预期。
4. 再看 reward_mean、KL、entropy、parse_success_rate、rollout latency 和 weight sync 时间。
5. 沿源码确认版本证据在哪一层丢失：RolloutServer、RolloutManager、SGLang engine、Sample meta_info，还是日志/metrics 写出。

这条顺序能避免把 stale rollout 误判成 reward parser 或模型质量问题。先说明“样本来自哪一版 policy”，再讨论“样本质量如何”。

## Lab 验收边界

本讲 patch 命令：

```bash
make patch-test M=l33_rl_rollout_freshness
```

参考实现验证：

```bash
IMPL=reference make patch-test M=l33_rl_rollout_freshness
```

patch 验收的是 versioned rollout manager 的本地行为。测试通过后，还要能把这些行为对应到 MiniInfra、SLiME rollout manager、SGLang engine weight version 和 Sample meta_info。
