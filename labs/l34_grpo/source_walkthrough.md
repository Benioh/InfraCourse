# L39 源码带读：GRPO / RLOO

这份带读按“patch 合同 -> 测试 -> reward -> 真实训练工具 -> smoke”组织。先把数值合同读清楚，再看真实框架里的扩展分支。

## 0. 源码地图

```text
labs/l34_grpo/patch/starter/grpo.py
labs/l34_grpo/patch/reference/grpo.py
labs/l34_grpo/patch/tests/test_patch.py
mini_infra/rl/reward.py
github_repo/slime/slime/utils/ppo_utils.py
labs/l34_grpo/scripts/run_grpo_smoke.py
```

## 1. Patch Starter

文件：`labs/l34_grpo/patch/starter/grpo.py`

阅读顺序：

- L8-L9：`grpo_advantage` 的输入是一组 rewards。
- L14：starter 在 advantage 未实现时抛错。
- L17-L23：`rloo_advantage` 的 TODO 写出 leave-one-out baseline。
- L26-L34：`grpo_loss` 的输入包含 current、old、reference log-probs、advantages 和 mask。
- L35-L44：TODO 按 ratio、clip、KL、mask 和指标返回组织。

读完要得到的结论：L39 patch 的输入张量形状很小，但每个字段都对应真实 RL 训练日志中的一类证据。

可以先跳过：具体实现细节，先把输入输出和边界记住。

## 2. Patch Reference

文件：`labs/l34_grpo/patch/reference/grpo.py`

阅读顺序：

- L8-L9：GRPO advantage 函数入口。
- L13-L15：std 太小时返回去均值结果，避免 NaN。
- L18-L24：RLOO 使用排除自己的组内均值。
- L27-L37：loss 输入和 mask dtype 处理。
- L38-L47：ratio、clip、per-token loss、denom 和 KL。
- L48-L57：clipped fraction、ratio_mean 和返回字典。

读完要得到的结论：reference 把 GRPO 的最小数值合同压在 50 行以内，重点是 mask-normalized loss 和指标。

可以先跳过：PyTorch dtype 细节，先确认公式和归一化位置。

## 3. Patch Tests

文件：`labs/l34_grpo/patch/tests/test_patch.py`

阅读顺序：

- L21-L25：GRPO advantage 均值为 0。
- L28-L32：GRPO advantage 标准差为 1。
- L35-L46：G=1 和常数 reward 返回 0。
- L49-L56：RLOO 的第一个和最后一个样本用其他 responses 做 baseline。
- L59-L63：全相等 reward 下 RLOO 返回 0。
- L66-L78：KL beta 改变总 loss。
- L81-L91：ratio 超出 clip 区间时 clipped fraction 大于 0。
- L94-L99：mask 测试准备不同 token mask。

读完要得到的结论：测试覆盖的是数值边界，不覆盖真实 rollout、reward model 或长训。

可以先跳过：pytest import 细节。

## 4. MiniInfra Reward

文件：`mini_infra/rl/reward.py`

阅读顺序：

- L21-L33：`score` 从 prediction 和 target 中抽取最终数字，并返回 reward。

读完要得到的结论：rule-based reward 可以很简单，但它的输出分布会直接决定 GRPO group advantage 是否有信号。

可以先跳过：命令行入口。

## 5. SLiME PPO Utils

文件：`github_repo/slime/slime/utils/ppo_utils.py`

阅读顺序：

- L28-L41：`compute_approx_kl` 根据 kl_loss_type 选择 k1/k2/k3 等 estimator。
- L43-L51：importance ratio 和 low_var_kl clamp 是生产稳定性分支。
- L54-L60：OPSM 函数接收 log_probs、old_log_probs、advantages 和 masks。
- L79-L92：OPSM 用 sequence-level KL 和 negative advantage 生成 mask。
- L132-L138：policy loss 根据 KL 形式恢复 ratio，并构造 clipped 分支。
- L139-L148：dual-clip 分支和返回 clipfrac。

读完要得到的结论：真实系统围绕 KL、ratio、advantage 和 mask 增加了更多保护，patch 只保留最小合同。

可以先跳过：vocab parallel entropy 和 Megatron fused loss。

## 6. GRPO Smoke

文件：`labs/l34_grpo/scripts/run_grpo_smoke.py`

阅读顺序：

- L32-L49：根据 `IMPL` 选择 starter 或 reference。
- L52-L66：读取配置、创建 run 目录、写 resolved config。
- L85-L93：创建合成 policy、optimizer 和固定 rewards。
- L95-L99：进入 50 step smoke 循环，并准备 old/reference log-probs。
- loop 中段：构造 old/current/reference log-probs、reward、advantage 和 mask。
- report 部分：写出 `metrics.jsonl`、`artifacts/grpo_smoke.json` 和 `report.md`。

读完要得到的结论：smoke 只验证合成数值闭环和产物格式，不代表真实模型训练稳定。

可以先跳过：H200 verl 命令模板。

## 读完后的自检问题

1. GRPO advantage 的 group 边界是什么？
2. RLOO baseline 为什么要排除当前 response？
3. mask 在 policy loss、KL 和指标里分别怎么用？
4. SLiME 的 k3 KL 和 patch 的 KL 有什么对应关系？
5. smoke 通过以后还不能证明什么？
