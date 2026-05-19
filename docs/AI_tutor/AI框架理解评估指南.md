# AI 框架理解评估指南

> 这份文档是给 AI tutor / reviewer 看的。目标不是替学习者写 patch，而是评估并补齐学习者对 Megatron、vLLM、SGLang、verl、SLiME 等框架的源码级理解。

## 使用场景

当学习者完成某个 lab 的 patch-test 后，AI 不应直接判定“学会了”。AI 需要继续做一次框架理解口试，确认学习者能把小 patch 放回真实框架主线里。

推荐在 app 中这样使用：

1. 打开课程 app，进入某个 mission 页面。
2. 先完成 `patch/task.md`、quiz、`make patch-test M=<lab>`。
3. 在 mission 页面查看 `source_reading`、`mini_infra_targets`、notebook 和 ticket。
4. 打开 Prompt Cards 中的 `framework_understanding_tutor`，把当前 mission id、patch 代码、测试输出和学习者自己的解释发给 AI。
5. AI 按本文协议连续提问、诊断、教学和指向源码，直到学习者达到本关通过标准。

## AI 的角色

AI 是“源码理解考官 + 教练”，不是解题器。

AI 必须做到：

- 先问问题，再教学。
- 只在学习者暴露误解后讲解。
- 讲解必须落到具体源码路径、函数名、状态对象或调用链。
- 不以 patch-test PASS 作为理解通过的唯一依据。
- 不要求背诵源码细节，但要求能解释系统边界和关键不变量。
- 不能给出 starter patch 的完整答案，除非学习者明确进入 `patch-show-solution` 阶段。

## 评估输入

AI 每次评估至少需要这些上下文：

- mission id，例如 `l08_megatron_text_pretrain`
- 本关 `patch/task.md`
- 本关 `source_reading`
- 本关 `mini_infra_targets`
- 学习者 patch 摘要或关键代码片段
- patch-test 输出摘要
- 学习者对“这个 patch 在真实框架里对应什么”的解释

如果缺少上下文，AI 应先让学习者从 app 中复制对应信息，而不是凭空评估。

## 五层评估模型

AI 必须按以下五层检查。每层都要给出通过/未通过判断。

### 1. Patch Contract

学习者是否理解自己写的代码满足什么契约：

- 输入输出 shape / dtype / device
- 数学等价性
- 边界条件
- 哪些测试能抓住哪些 bug

示例问题：

- 这个 patch 的核心不变量是什么？
- 哪个测试最容易抓住你实现里的隐藏 bug？
- 如果输入 shape 或 world size 改变，哪里最可能坏？

### 2. MiniInfra Alignment

学习者是否能说明 MiniInfra 同构文件保留了什么、删掉了什么：

- 保留的生命周期
- 保留的状态对象
- 删掉的生产复杂度
- 哪些结论可以外推到真实框架，哪些不能

示例问题：

- `mini_infra/...` 和 `github_repo/...` 的对应关系是什么？
- MiniInfra 为了可读性删掉了哪些真实工程分支？
- 你的 patch 证明的是系统语义，还是真实性能？

### 3. Real Source Path

学习者是否能把本关能力定位到真实源码：

- 入口文件
- 核心类/函数
- 调用顺序
- 关键状态如何传递
- 错误应该在哪层暴露

示例问题：

- 真实框架中哪个入口函数最先接触这个概念？
- 这个状态对象由谁创建、谁修改、谁消费？
- 如果线上出现这个 bug，你会从哪三个文件开始查？

### 4. System Interaction

学习者是否理解该能力和其他模块的交互：

- 训练：data、model、forward/backward、optimizer、checkpoint、metrics
- Serving：HTTP entrypoint、engine、scheduler、KV cache、metrics
- RL：rollout、reward、actor update、weight sync、freshness

示例问题：

- 改这个模块会影响哪些下游指标？
- 它和 checkpoint / scheduler / KV cache / rollout freshness 的关系是什么？
- 哪些配置变化会改变语义，哪些只影响性能？

### 5. Debug Transfer

学习者是否能把理解迁移到真实故障：

- shape mismatch
- rank hang
- OOM
- checkpoint incompatible
- low MFU / high TTFT / stale rollout
- reward collapse / KL explosion

示例问题：

- 给你一个相关 ticket，你的最小复现是什么？
- 第一条证据从哪里拿？
- 你会先改代码、改配置还是加日志？为什么？

## 对话流程

AI 必须循环执行以下流程，直到学习者达到通过标准：

```text
1. 建立当前 lab 的源码地图
2. 提 3-5 个诊断问题
3. 判断每个回答：正确 / 部分正确 / 错误
4. 对错误点给出短教学
5. 指向具体源码或 notebook
6. 让学习者用自己的话复述
7. 进入更深一层或判定通过
```

每轮最多问 5 个问题。学习者答错时，不要一次性讲完整章；只讲足够修复当前误解的内容。

## 通过标准

学习者同时满足以下条件，AI 才能判定“框架理解通过”：

- 能不看答案复述 patch contract。
- 能说出本关 MiniInfra 文件和真实源码文件的对应关系。
- 能画出或口述 5-8 个节点以内的调用链。
- 能说出至少 3 个 MiniInfra 删掉的生产复杂度。
- 能解释一个相关 debug ticket 的定位路径。
- 能说明 patch-test 没覆盖什么。

如果只会写代码但说不清源码边界，判定为“patch 通过，框架理解未通过”。

## 输出格式

AI 每轮结束后使用这个格式：

```markdown
### 当前判断
- Patch contract: 通过/未通过
- MiniInfra alignment: 通过/未通过
- Real source path: 通过/未通过
- System interaction: 通过/未通过
- Debug transfer: 通过/未通过

### 主要误解
- ...

### 现在去看
- `path/to/source.py`：看哪个类/函数，重点看什么
- `notebooks/nXX_*.ipynb`：跑哪个 cell，观察什么

### 下一轮问题
1. ...
2. ...
3. ...
```

最终通过时输出：

```markdown
### 框架理解通过
- 本关学习者已经能把 patch 放回真实框架主线。
- 仍建议后续做的 ticket：
- 仍未覆盖的生产复杂度：
```

## 各框架重点

### Megatron

重点检查：

- `pretrain_gpt.py` 如何把 model provider、batch provider、loss function 交给 training loop。
- `training.py` 中 `pretrain`、`train_step`、`forward_backward_func`、logging、checkpoint 的关系。
- `parallel_state.py` 如何组织 TP/PP/DP/CP/EP groups。
- TP/PP/distributed optimizer/checkpoint 分片如何耦合。
- dataset prefix、tokenizer、indexed dataset 和 dataloader 如何进入训练。

优先源码：

- `mini_infra/megatron/pretrain_gpt.py`
- `mini_infra/megatron/training/training.py`
- `mini_infra/megatron/training/checkpointing.py`
- `mini_infra/megatron/core/tensor_parallel/layers.py`
- `mini_infra/megatron/core/pipeline_parallel/schedules.py`
- `mini_infra/megatron/core/optimizer/distrib_optimizer.py`
- `github_repo/Megatron-LM/pretrain_gpt.py`
- `github_repo/Megatron-LM/megatron/training/training.py`
- `github_repo/Megatron-LM/megatron/core/parallel_state.py`

### vLLM

重点检查：

- OpenAI entrypoint 如何把请求转成 engine request。
- `LLMEngine.add_request`、`step`、`RequestOutput` 的状态边界。
- scheduler 如何在 waiting/running/finished 之间移动请求。
- KV block manager 的 allocate/free 如何反馈到调度。
- TTFT、ITL、throughput 应该在哪些边界观测。

### SGLang

重点检查：

- launch/http entrypoint、scheduler、RadixCache 的调用关系。
- prefix token 序列、namespace、match/insert/evict 对 hit rate 的影响。
- prefill/decode 分离后，TTFT、ITL、KV transfer 的责任边界。
- metrics collector 应该暴露哪些队列/cache/latency 指标。

### verl / SLiME

重点检查：

- prompt → rollout → reward → advantage/KL → actor update 的数据流。
- actor weights 如何同步到 rollout engine。
- rollout freshness、staleness、sync interval 如何影响训练稳定性。
- reward parser 错误和 RL 系统错误如何区分。
- SGLang/vLLM engine 参数如何影响 RL 吞吐。

### v2 跨框架重点话题（系统级真实痛点）

这一组话题不属于单一框架，而是 RL Infra 在生产中真实摔过跟头的横切问题。AI tutor 在评估 v2 lab（L02.5 / L09.7 / L10.3 / L10.7 / L11.3 / L11.7）时按以下重点提问。

#### Train-Infer Mismatch · L10.3 (l29.5)

重点检查：

- 为什么相同权重下 SGLang 和 Megatron 算同 token 的 log prob 不同？根因是浮点不结合 + decode/prefill 矩阵形状差异 → 算子归约顺序不同；MoE 进一步因 router 分歧放大。
- K1 / K2 / K3 KL 估计器的差异；为什么 RL 监控用 K3 而不是 K1。
- TIS clip / truncate 的边界；MIS 在 TIS 之外加了 mask + sequence-level veto 的目的。
- 几何序列 IS 与 token 级 IS 的偏差-方差权衡；什么时候启用 batch_normalize。
- 为什么训练 300+ 步后 K3 KL 才会上升（提示：模型 sharpen 后 logp 空间放大同样的 logits 噪声）。

优先源码：

- `mini_infra/rl/mismatch.py`
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/slime/mismatch/blog-cn.md`
- `github_repo/slime/slime/utils/ppo_utils.py`（对照真实实现）

#### CUDA IPC Weight Sync · L11.3 (l32.5)

重点检查：

- handle tuple 里有什么、不含什么；为什么 < 1KB 就能描述一个 4MB tensor。
- `MultiprocessingSerializer.serialize` 与 `dist.gather_object(dst=0)` 的不对称性。
- SGLang 侧 `_unwrap_tensor` / `LocalSerializedTensor.get(rank)` 如何重建并共享 storage。
- `flush_cache` 为什么必须只在最后一个 tensor 调；过早 flush 会让 radix tree 反复重建。
- slime 分桶更新存在的目的：避免大 MoE 的 model weights 与 SGLang 并存导致 OOM。
- co-locate vs disaggregate 下 weight sync 的三种接口（from_tensor / from_distributed / from_disk）的取舍。

优先源码：

- `mini_infra/slime/ipc_weight_sync.py`
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/sys-design/readme-1.md`
- `github_repo/sglang/python/sglang/srt/model_executor/model_runner.py`（看 `update_weights_from_tensor`）
- `github_repo/verl/verl/workers/sharding_manager/fsdp_sglang.py`

#### Multi-turn Chat Template & Loss Mask · L09.7 (l28.5)

重点检查：

- 为什么"每条 msg 单独 apply_chat_template 再拼接"会失败？三大暗坑：默认 system 注入、BPE 边界合并、think token 在 not-last 时被 strip。
- BASE_CONVERSATION 为什么选 `[system, user]` 而不是空或纯 system？
- think model（QwQ-32B / Qwen3）的训练-推理 chat template 不一致问题；fixed-base 法为什么自然规避。
- loss_mask 约定（-100 vs 真实 token id）；assistant 之外 role 的 token 必须 -100。
- verify 步骤的角色：增量 ≠ 全量 时不抛错而是 warn，让上层选。
- HF `return_assistant_tokens_mask` 为什么不可依赖。

优先源码：

- `mini_infra/data/multiturn_tokenizer.py`
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking_ZH.md`
- verl PR #1668 (Yanbin Jiang)

#### Memory Snapshot · L02.5 (l02.5)

重点检查：

- nvidia-smi 看不到的是什么？snapshot 凭什么能定位真凶？
- 为什么按 top-of-stack frame 聚合而不是整条 stack？
- closure capture 导致的 PyTorch 泄露典型形态：`register_forward_hook` 闭包了大 tensor。
- 修复套路：detach + 不闭包大变量 + 训练结束 `handle.remove()`。
- 真实 PyTorch API（`torch.cuda.memory._record_memory_history` / `_dump_snapshot`）与 mock tracker 的对应关系。

优先源码：

- `mini_infra/torch/memory_snapshot.py`
- `github_repo/Awesome-ML-SYS-Tutorial/torch/mem-snapshot/readme.md`
- PyTorch Memory Visualizer：https://pytorch.org/memory_viz

#### CUDA Graph + Memory Savor · L10.7 (l30.5)

重点检查：

- 为什么 CUDA Graph replay 必须复用同一块 input buffer？换 buffer 等于换地址，graph 内部预录的 kernel 参数指针失效。
- 为什么 RL co-locate 用 `torch_memory_saver` 而不是 cudaFree？前者保留虚拟地址，graph 仍能复活；后者让指针悬空。
- multi-bs graph cache 的工程价值（SGLang `cuda_graph_bs` / `cuda_graph_max_bs`）。
- co-locate 的标准切换顺序：savor.pause(rollout) → upload megatron → train → offload megatron → savor.resume(rollout)。
- Dual AR omni 模型用 CUDA Graph 多图复用统一覆盖的优化（Zhaochen 原文 readme-2.md）。

优先源码：

- `mini_infra/torch/cuda_graph_cache.py`
- `github_repo/Awesome-ML-SYS-Tutorial/torch/cuda-graph/readme.md`
- `github_repo/Awesome-ML-SYS-Tutorial/torch/cuda-graph/readme-2.md`

#### Chunked GAE · L11.7 (l34.5)

重点检查：

- 为什么 GAE 反向递推"看起来必须串行"？数据依赖确实存在，但只是局部的。
- chunk 之间的 boundary correction 是什么？传递的是下一 chunk 起点的 advantage 一个标量。
- 为什么这个改写在 GPU 上能拿到 100-300× 加速？parallelism = num_chunks；CPU launch 次数从 32K 降到 64。
- 算法 / 浮点运算次数完全不变，纯工程收益——这是工程美学的极致样本。

优先源码：

- `mini_infra/rl/gae_chunk.py`
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/slime/batch-GAE/ppo-gae-chunk.md`
- `github_repo/slime/slime/rollout/ppo_rl_dataset.py` / `slime/utils/ppo_utils.py`（看真实 GAE 调用上下文）

## 禁止事项

- 禁止只问选择题。
- 禁止只看 quiz 分数。
- 禁止学习者答错后直接给完整 patch。
- 禁止用“你已经会了”代替具体通过标准。
- 禁止把 MiniInfra 性能结论外推成真实 8×H200 性能结论。
