# 源码带读：L32 Train-Infer Mismatch

这份带读按“patch 合同 -> pytest 边界 -> SLiME 生产分支”的顺序走。读源码时先建立数据流：`train_log_prob - rollout_log_prob -> log_ratio -> weight / mask / metrics`，再看每个分支怎样处理边界。

## 0. 源码地图

```text
labs/l29.5_train_infer_mismatch/patch/starter/mismatch.py
labs/l29.5_train_infer_mismatch/patch/reference/mismatch.py
labs/l29.5_train_infer_mismatch/patch/tests/test_patch.py

github_repo/slime/examples/train_infer_mismatch_helper/mis.py
```

## 1. Patch starter：确认学生要补的合同

文件：[patch/starter/mismatch.py](patch/starter/mismatch.py)

先看第 20-32 行。`compute_k3_kl` 的 TODO 已经把公式拆成 `log_ratio`、`ratio`、`k3` 和 `mean`。读完要能说清为什么输入是 logprob，为什么输出是标量。

再看第 36-66 行。`tis_correct` 和 `mis_with_mask` 共用 `ratio = exp(logp_new - logp_old)`，但边界动作不同：TIS clamp 后继续使用，MIS 越界后置零。

最后看第 69-106 行。`geometric_seq_is` 练的是 `(B,T)` mask、device 和长度归一化；`apply_veto` 练的是 log 空间阈值比较；`batch_normalize_weights` 练的是均值归一化与除零保护。

## 2. Patch reference：看最小正确实现

文件：[patch/reference/mismatch.py](patch/reference/mismatch.py)

按函数顺序读：

- 第 12-14 行：K3 把 ratio、`ratio - 1 - log_ratio` 和 mean 连起来。
- 第 24-26 行：TIS 先算 ratio，再 clamp，再乘 advantage。
- 第 36-38 行：MIS 构造区间 mask，越界 token 的输出为 0。
- 第 46-52 行：Geometric IS 生成 mask、求有效 token 的 log ratio 均值，再取 exp。
- 第 55-62 行：Veto 用 `math.log(threshold)`，Batch Normalize 用 mean clamp。

reference 的价值是把数学合同落成 3 到 8 行 PyTorch。读完后不要急着抄，先回头对照 starter 的 shape 注释。

## 3. Patch tests：每个测试在保护什么

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

先看第 21-23 行。`IMPL` 环境变量决定跑 starter 还是 reference，这让你能先用 reference 验证测试环境。

K3 相关测试在第 26-52 行。它们分别覆盖 aligned 为 0、手算公式 `e - 2`、随机输入非负。TIS 和 MIS 测试在第 55-84 行，重点是高 ratio 被 clip 和越界 token 被 mask。

Geometric IS 测试在第 87-99 行。注意 padding 位置故意填入 99.0，如果实现没有按 `seq_lens` 屏蔽 padding，就会失败。Veto 和 Batch Normalize 测试在第 102-124 行，覆盖极低概率丢弃、均值为 1 和相对顺序保持。

## 4. SLiME `mis.py`：生产分支如何组织

文件：[github_repo/slime/examples/train_infer_mismatch_helper/mis.py](../../github_repo/slime/examples/train_infer_mismatch_helper/mis.py)

第 12-19 行先给出 masked sum/mean。真实训练里 loss mask 决定哪些 token 参与统计，空 mask 要有安全分母。

第 82-100 行是 veto 的序列级版本。它从 log ratio 里找灾难性 token，并把 token fraction 和 sequence fraction 写进 metrics。

第 116-147 行是 clip/mask 边界处理。clip 记录上下界命中率后 clamp，mask 记录命中率后修改 loss mask。这个结构对应本讲的 TIS 与 MIS。

第 192-200 行定义 token、sequence、geometric 三种 log ratio 聚合层级。第 219-239 行把 TIS 应用到 log ratio，先做安全 clamp 防止 exp overflow，再按 truncate/clip/mask 分支处理。

第 258-264 行把 veto 接到 rejection sampling mask 上。第 272-297 行做 batch normalization，按 token 或 sequence 层级求均值，再把所有权重除以同一个 batch mean。第 299-307 行记录最终 mean/min/max，供训练日志排查。

## 5. 可以先跳过的内容

- context parallel 版本的 `compute_mis_weights_with_cp`，先掌握单机 list-of-tensor 路径。
- 与 Megatron 通信、all-gather、slice logprob 相关的分支。
- 完整训练循环和 reward 计算；这些属于后续 rollout 与 RL pipeline 课程。

## 读完后的自检问题

1. K3、TIS、MIS、Geometric IS、Veto 和 Batch Normalize 分别读写哪些张量？
2. pytest 中哪个测试能发现 padding 没有被 mask？
3. SLiME `mis.py` 里 token、sequence、geometric 三种 level 的差异是什么？
4. 生产日志里至少要记录哪些 ratio、mask 或 batch norm 指标？
