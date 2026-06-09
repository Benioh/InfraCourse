# L32：Train-Infer Mismatch

本讲处理 RL 训练中的一个具体系统问题：rollout engine 用 decode 路径采样，training engine 用 forward/prefill 路径重算 log probability；二者即便加载同一份权重，也可能对同一段 token 序列给出不同的 logprob。差异进入 PPO/GRPO 后，会直接改变 importance ratio，让本应接近 on-policy 的更新带上 off-policy 噪声。

本讲不实现 PPO loss，也不启动多机 RL 训练。你要掌握的是六个可独立测试的算子：K3 KL 用来量化 mismatch，TIS 和 MIS 用 token-level ratio 修正 advantage，Geometric Sequence IS 把 token ratio 聚合到序列级，Veto 丢弃极端低概率样本，Batch Normalize 把一批权重的均值拉回 1。学完后，你应该能解释每个算子的输入、输出、数值边界和适用场景。

## 1. 本讲目标

- 解释 rollout decode 和 training prefill 为什么会产生 logprob 差异。
- 用 K3 KL 判断两套 engine 的概率是否开始分叉。
- 区分 TIS、MIS、Geometric IS、Veto 和 Batch Normalize 的作用边界。
- 读懂 patch reference、pytest 合同和 SLiME `mis.py` 的对应关系。
- 用 debug checklist 组织一次 mismatch 排查报告。

## 2. mismatch 从哪里来

在 RLHF 或 RLAIF 训练里，rollout engine 负责生成 response，training engine 负责把 response 放回模型里算 logprob、advantage 和 loss。常见部署会把这两部分放在不同框架或不同执行路径上，例如 SGLang/vLLM 负责高吞吐 decode，Megatron/verl 负责训练 forward 和 backward。

输入看似相同：同一份权重、同一批 token、相同 dtype。中间状态却不同：decode 通常一次处理一个新 token 并读取 KV cache，training forward 往往一次处理完整序列；batch shape、kernel 选择、Tensor Core 路径和归约顺序都会变。浮点加法不满足结合律，归约顺序改变后，logits 和 logprob 可能出现微小差异。

输出差异在 dense 模型里常常很小，但 RL loss 对 ratio 敏感。`ratio = exp(logp_new - logp_old)`，logprob 差 0.7 就会产生约 2 倍权重。MoE 模型还有一个额外放大器：router 是离散 top-k 选择，logits 的微小差异可能把 token 分到不同 expert，后续分支随之改变。

这类问题的主要风险在训练证据链变脏。reward 下降时，你很难只靠曲线判断是 reward parser、optimizer、rollout backend、权重同步还是 mismatch 修正配置的问题。L32 的策略是先把完整系统缩小成一张 token 表，再用可测试的小算子解释每个风险点。

## 3. K3 KL：先量化分歧

K3 KL 的输入是两组已采样 token 的 logprob：`logp_p` 和 `logp_q`。实现步骤是：

```python
log_ratio = logp_p - logp_q
ratio = torch.exp(log_ratio)
k3 = ratio - 1.0 - log_ratio
return k3.mean()
```

它的中间状态是 `log_ratio` 和 `ratio`，输出是一个标量均值。两个性质很关键：当两组 logprob 完全一致时，K3 为 0；对任意正 ratio，`ratio - 1 - log(ratio)` 非负。因此它适合作为 mismatch 的 sanity check。

K3 的边界也要明确。它只使用采样 token 的 logprob，不需要完整词表分布，所以不能用 `torch.distributions.kl_divergence` 替代。它也不直接修复 loss；看到 K3 KL 上升后，还要继续看 ratio 最大值、分位数、mask 命中率、veto 命中率和 grad norm。

## 4. Token-level TIS 与 MIS

TIS 和 MIS 都从同一个输入开始：`logp_old`、`logp_new` 和 `advantages`。通常 old 代表 rollout 或行为策略侧 logprob，new 代表 training 或当前策略侧 logprob。二者先算：

```python
ratio = torch.exp(logp_new - logp_old)
```

TIS 使用 `clamp(ratio, lo, hi) * advantages`。中间状态是被截断后的 ratio，输出 shape 与 advantage 一致。它的优点是所有 token 仍然参与训练；代价是越界 token 被压到边界后仍会贡献梯度，因此会引入偏差。

MIS 使用区间 mask：ratio 落在 `[lo, hi]` 内就保留 `ratio * advantages`，越界就置 0。它比 TIS 更硬，能阻止极端 ratio 污染梯度；代价是有效 token 数减少，mask 过严时学习信号会变少。

这两个算子不能互相替代。TIS 更适合做连续的 token-level 缓冲，MIS 更适合在信任区间外切断贡献。生产配置常把 token-level TIS、sequence-level mask 或 rejection、veto 和 batch normalization 组合起来，避免只依赖一个阈值。

## 5. Geometric Sequence IS：把 token 权重汇成序列权重

token-level ratio 只描述单个 token。整条 response 的分布偏移需要把多个 token 的 log ratio 聚合起来。朴素 sequence IS 会把所有 ratio 连乘，在 log 空间等价于对 `logp_new - logp_old` 求和后取 exp。它接近整序列权重，但长序列的方差很高，少数 token 的差异会被长度放大。

Geometric Sequence IS 采用长度归一化：

```python
log_ratios = logp_new - logp_old
mask = torch.arange(T, device=log_ratios.device)[None, :] < seq_lens[:, None]
sum_log = (log_ratios * mask).sum(dim=-1)
weight = torch.exp(sum_log / seq_lens.clamp(min=1))
```

输入 `logp_old` 和 `logp_new` 是 `(B, T)`，`seq_lens` 是 `(B,)`，输出是 `(B,)`。mask 是核心中间状态：padding 位置可能包含任意值，不能进入序列权重。几何平均的代价是引入偏差，收益是让长短 response 的权重幅度更可比。

## 6. Veto 与 Batch Normalize

Veto 处理极端低概率 token。若 rollout 侧某个 token 概率低于阈值，分母接近 0，ratio 可能达到很大数量级。本讲的 `apply_veto(logp_rollout, threshold=1e-6)` 用 log 空间比较：

```python
log_thresh = math.log(threshold)
mask = (logp_rollout >= log_thresh).to(logp_rollout.dtype)
```

输入是 rollout logprob，输出是 0/1 mask。用 log 比较可以避免先 `exp(logp)` 带来的下溢。真实系统里，veto 常按序列做拒绝：任一有效 token 触发灾难性阈值，就把整条 response 从更新中移除。

Batch Normalize 处理另一类问题：一批 IS weight 的平均值如果长期远离 1，会改变有效学习率。`batch_normalize_weights` 的合同很小：`weights / weights.mean().clamp(min=1e-12)`。输出保持相对顺序，只把均值拉回 1。它不能识别坏样本，因此通常放在 TIS/MIS/Veto 之后。

## 7. 实现路径和测试边界

写 patch 时按测试顺序实现：先写 K3，再写共享 ratio 的 TIS/MIS，接着写 Geometric IS 的 mask 和长度归一化，最后写 Veto 与 Batch Normalize。所有函数都是基础 PyTorch 张量操作，错误通常来自公式方向、shape、device、padding 或极值。

几个细节要特别检查。`torch.arange(T)` 必须放在 `log_ratios.device` 上。mask 要转成参与乘法的 dtype。`seq_lens` 做除法前要转成浮点 dtype 并 clamp 到至少 1。Batch Normalize 的 mean 要 clamp 一个很小的下界，避免全零权重时除零。

验收命令是：

```bash
make patch-test M=l29.5_train_infer_mismatch
```

pytest 文件包含 10 个测试：K3 的零值、公式和非负性，TIS 的 clip 与 pass-through，MIS 的 outlier mask，Geometric IS 的定义，Veto 的极低概率丢弃，以及 Batch Normalize 的均值和相对顺序。测试通过只证明算子合同正确，不证明完整 RL pipeline 已稳定。

## 8. 从 patch 走向生产排查

生产排查时，先把问题分成三类。第一类是观测：K3 KL、train/rollout PPL、logprob diff、ratio 分位数是否完整落盘。第二类是修正：TIS/MIS 的上下界、veto 阈值、batch norm factor 是否和当前任务匹配。第三类是链路：token 序列、loss mask、rollout logprob 和 training logprob 是否一一对应。

不要只问“要不要打开 TIS”。更有用的问题是：高 ratio 来自少量 token 还是整批偏移？veto 命中率是否突然升高？batch norm factor 是否长期大于 1 或小于 1？K3 KL 上升时，reward、entropy、response length、grad norm 是否同步异常？这些证据能帮助你判断该调修正阈值、查采样后处理、查 backend 对齐，还是回到权重同步。

算法修正也有边界。TIS/MIS/Veto 能降低 mismatch 进入 loss 的风险，但不能保证 rollout engine 和 training engine 位级一致。若业务要求严格同策略，需要考虑真正统一 backend 或更强的 on-policy 路径；若目标是在保留吞吐的同时降低训练崩溃风险，这些算子就是第一层可控安全网。

## Lab 验收边界

本讲 patch 命令：`make patch-test M=l29.5_train_infer_mismatch`。

patch 验收的是六个局部算子：K3 KL、TIS、MIS、Geometric Sequence IS、Veto 和 Batch Normalize。它不覆盖 PPO loss、rollout 调度、权重同步、reward parser、context parallel 切片或真实训练日志聚合。

课后使用 `outputs/rl_rollout_template.md` 记录一次完整复盘，把命令、输入、指标、源码判断和下一步动作写清楚。
