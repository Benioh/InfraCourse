# L38 源码带读：Rollout Freshness

这份带读按“本地合同 -> MiniInfra -> SLiME 生产路径 -> drill 验收”组织。读的时候不要从文件顶部扫到底，先抓住版本字段在哪里产生、比较、写入样本和落盘。

## 0. 源码地图

```text
labs/l33_rl_rollout_freshness/patch/starter/rollout_manager.py
labs/l33_rl_rollout_freshness/patch/reference/rollout_manager.py
labs/l33_rl_rollout_freshness/patch/tests/test_patch.py
mini_infra/slime/ray/rollout.py
mini_infra/slime/ray/train_actor.py
mini_infra/slime/train.py
github_repo/slime/train.py
github_repo/slime/slime/ray/rollout.py
github_repo/slime/slime/utils/types.py
github_repo/slime/slime/backends/sglang_utils/sglang_engine.py
labs/l33_rl_rollout_freshness/scripts/run_rl_drill.py
```

## 1. Patch Starter：先看要补的合同

文件：`labs/l33_rl_rollout_freshness/patch/starter/rollout_manager.py`

阅读顺序：

- L8-L13：`RolloutData` 的四个字段，注意 `meta_info` 是版本证据出口。
- L16-L28：`RolloutServer` 保存 `server_id` 和 `weight_version`，并按版本生成 response。
- L32-L38：`meta_info` 写入 `server_id`、`weight_version`、`actor_version` 和 `staleness`。
- L47-L56：`RolloutManager` 保存 server 列表、阈值和 round-robin 指针，`generate` 留给学生实现。
- L58-L65：`update_weights` 和 `freshness` 的 TODO，分别对应版本推进和诊断视图。

读完要得到的结论：patch 的核心在于保证每次生成都带着可解释的版本字段。

可以先跳过：response 字符串的具体格式，它只是测试用的可见输出。

## 2. Patch Reference：对照正确控制流

文件：`labs/l33_rl_rollout_freshness/patch/reference/rollout_manager.py`

阅读顺序：

- L47-L55：构造函数拒绝空 server，并定义 `_fresh_enough`。
- L57-L65：`generate` 从 `next_server_index` 开始遍历，找到 fresh server 后推进指针。
- L67-L74：`update_weights` 把 `server_ids` 转为允许集合，只更新命中的 server。
- L76-L80：`freshness` 返回 actor 和每个 server 的版本差。

读完要得到的结论：freshness 和 round-robin 需要同时成立。只做其中一项都会留下训练风险或负载偏斜。

可以先跳过：dataclass 定义，它和 starter 相同。

## 3. Patch Tests：看验收边界

文件：`labs/l33_rl_rollout_freshness/patch/tests/test_patch.py`

阅读顺序：

- L20-L30：正常 generate 要返回正确 response，并写完整 `meta_info`。
- L33-L41：stale server 被跳过，fresh server 被选中，round-robin 指针回到 0。
- L44-L48：全部 server stale 时必须抛 `RuntimeError`。
- L51-L58：先局部更新 `s1`，再全量更新所有 server。
- L61-L64：`freshness` 的公式是 `actor_version - engine_version`。

读完要得到的结论：测试只覆盖版本准入、局部更新、异常边界和证据字段，不覆盖真实 Ray 或 SGLang。

可以先跳过：`_impl` 的 import 细节，只要知道测试会在 starter/reference 之间切换。

## 4. MiniInfra Rollout：看最小系统如何承接 patch

文件：`mini_infra/slime/ray/rollout.py`

阅读顺序：

- L27-L39：server 接收 prompts 和 actor_version，生成带版本号的 response。
- L40-L49：`meta_info` 同时包含 cache、server、actor、weight 和 staleness 字段。
- L55-L68：manager 保存 server 列表、`max_staleness` 和 `next_server_index`。
- L70-L82：generate 自动补 actor_version，按 freshness 遍历 server，失败时抛错。
- L84-L93：`update_weights` 支持全量和子集更新。
- L95-L99：`freshness` 导出当前版本差。

读完要得到的结论：MiniInfra 把 patch 合同放进一个可以运行的 toy rollout 路径，方便后续 train loop 和 drill 复用。

可以先跳过：Mini SGLang scheduler 的内部调度细节，本讲只需要知道 server 能生成 response。

## 5. MiniInfra Train Actor 和 Train Loop：看版本如何推进

文件：`mini_infra/slime/ray/train_actor.py`

阅读顺序：

- L44-L53：`train` 根据 rollout response 得到 reward，并递增 `weight_version`。
- L55-L58：`update_weights` 把 actor 当前版本写回 rollout manager。

文件：`mini_infra/slime/train.py`

阅读顺序：

- L16-L19：最小训练循环按 generate、train、update_weights 推进。
- L20-L29：history 记录 reward、weight version 和 rollout meta_info。

读完要得到的结论：actor version 的推进发生在训练侧，rollout freshness 的证据进入 history 后才能复盘。

可以先跳过：CLI 和 JSON 输出，它们只是方便命令行观察。

## 6. SLiME Train Loop：定位真实调用顺序

文件：`github_repo/slime/train.py`

阅读顺序：

- L15-L24：先创建 rollout manager，再创建 actor/critic。
- L26-L36：初始化阶段按需 onload rollout，并推一次 actor 权重。
- L66-L72：训练循环先向 rollout manager 请求数据。
- L78-L85：actor/critic 消费 rollout data 进行训练。
- L90-L94：offload train 后 onload rollout，并触发 `actor_model.update_weights()`。

读完要得到的结论：真实 SLiME 也遵循 rollout -> train -> update_weights 的主路径；freshness 是这条路径上的控制面证据。

可以先跳过：save/eval 触发条件和 tracking 细节。

## 7. SLiME Rollout Manager：看生产复杂度

文件：`github_repo/slime/slime/ray/rollout.py`

阅读顺序：

- L38-L45：`ServerGroup` 描述一组同构 SGLang engines。
- L47-L57：server group 保存 engine 列表、worker type、offset 和 model path。
- L210-L223：`RolloutServer` 聚合多个 server group，并标记 `update_weights`。
- L349-L363：Ray remote `RolloutManager` 初始化数据源和 rollout 函数。
- L444-L453：查找允许更新 actor 权重的 server。
- L460-L472：返回可更新 engines、engine lock、新 engine 数和 GPU 映射。
- L478-L486：`generate` 调用 `_get_rollout_data` 并记录 rollout 日志。
- L487-L491：非 debug 模式会把 samples 转成训练数据。
- L567-L585：从 debug 数据或 rollout 函数取样本和 metrics。
- L682-L689：样本先经过 reward 后处理。
- L694-L703：训练数据保留 tokens、response_lengths、rewards、truncated 和 sample indices。

读完要得到的结论：生产路径多了 Ray、engine group、fault tolerance、data source、debug dump 和训练数据切分，但版本问题仍然落在 rollout manager 和 sample meta_info 之间。

可以先跳过：engine 启动端口分配、健康监控和多模态字段扩展。

## 8. Sample 与 SGLang Engine：看版本证据出口

文件：`github_repo/slime/slime/utils/types.py`

阅读顺序：

- L8-L18：`Sample` 保存 prompt、tokens 和多模态输入。
- L19-L27：response、reward、`weight_versions` 和 rollout logprob 等字段。
- L153-L163：`update_from_meta_info` 更新 speculative 和 prefix cache 信息。
- L165-L166：meta_info 中存在 `weight_version` 时追加到 `sample.weight_versions`。

文件：`github_repo/slime/slime/backends/sglang_utils/sglang_engine.py`

阅读顺序：

- L266-L278：tensor update 接口接收 serialized tensors、load_format、flush_cache 和可选 weight_version。
- L279-L289：payload 写入 `weight_version` 后发给 SGLang server。
- L349-L355：engine 通过 HTTP endpoint 查询当前 `weight_version`。

读完要得到的结论：版本证据一端来自 update 请求，一端进入 sample；debug 时要能同时看发送和接收两边。

可以先跳过：磁盘 reload 和分布式 update 的完整参数，本讲只关心 weight_version。

## 9. Drill 脚本：看证据如何落盘

文件：`labs/l33_rl_rollout_freshness/scripts/run_rl_drill.py`

阅读顺序：

- L54-L68：每一步按 update_rate 更新部分 server，再尝试 generate。
- L69-L78：返回 failures、p50/p99 staleness 和 successful rollouts。
- L87-L95：读取配置、创建 run 目录、写 resolved config。
- main 后半段分别运行 nominal 和 no_update 场景。
- main 后半段把两类 summary 写入 `metrics.jsonl`，并写出 `artifacts/rl_drill.json`。
- report 构造部分记录目标、配置、结果、诊断和 PR review 检查项。

读完要得到的结论：drill 的目的在于验证版本合同的证据格式和失败边界，GPU 性能要用真实集群另测。

可以先跳过：H200 模板命令拼接，真实运行时再检查。

## 读完后的自检问题

1. `actor_version` 在 MiniInfra 中哪里递增？
2. patch reference 里哪一段同时保证 freshness 和 round-robin？
3. SLiME 如何找出可以被 actor 权重更新的 rollout engines？
4. `Sample.weight_versions` 从哪里填入？
5. drill 的 `no_update` 场景为什么必须出现 failure？
