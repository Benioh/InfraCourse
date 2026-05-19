# L29.5 · Train-Infer Mismatch：让速度与精度同在

> 即便 Rollout 引擎和 Training 引擎用完全相同的权重，对相同的 token 序列算出的 log prob 也会有微小差异。
> 这种差异在 dense 模型上量级 `1e-5 ~ 1e-3`，在 MoE 模型上可达 `1e-3 ~ 1e-1`，长 RL 训练里能把训练干崩。
>
> 本关一次性写出 **5 个核心修正算子**：K3 KL（衡量）、TIS（截断 IS）、MIS（带 mask 的 IS）、Geometric Sequence IS、Veto。
> 这些是 slime / verl / AReaL 在线上真正用的修正算法。

## 真实事故

参考 slime 团队 [让速度与精度同在](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-cn.md)
和 [Richard Li 的 RL collapse 系列](https://richardli.xyz/rl-collapse)。Qwen30B-A3B 在 320 步附近因为 train-infer mismatch 导致 grad norm 从 0.07 跌到 0.02，紧接着 reward 骤降。
正确配置的 TIS + MIS 可以把它从崩溃边缘拉回。

## 闭环

```bash
cat labs/l29.5_train_infer_mismatch/patch/task.md
$EDITOR labs/l29.5_train_infer_mismatch/patch/starter/mismatch.py
make patch-test M=l29.5_train_infer_mismatch
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_k3_kl_zero_when_aligned` | log_p == log_q → K3 KL = 0 |
| `test_k3_kl_formula` | 经典 case：logp=-1, logq=-2 → K3 = e − 1 − 1 |
| `test_tis_clips_high_ratio` | 大 ratio 被 clamp 到上界 hi |
| `test_mis_masks_outliers` | 越界的 token 被 mask 为 0 |
| `test_geometric_seq_is` | 几何均值 = exp(mean(log_ratio)) |
| `test_veto_drops_extreme_low_prob` | logp < log(threshold) 时 mask=0 |
| `test_batch_normalize_mean_one` | 自归一化后均值 = 1 |

## 卡住怎么办

1. 先看 `notebooks/n19_train_infer_mismatch.ipynb` 把 K3 KL / IS 修正的几何直观跑一遍。
2. `make patch-hint M=l29.5_train_infer_mismatch` 看 TODO 与提示。
3. `make patch-show-solution M=l29.5_train_infer_mismatch` 看参考解。

## 写完之后你能做什么

- 解释 RL 训练为什么需要修正"训推不一致"，以及 Token-level / Sequence-level / Geometric-level IS 的偏差–方差取舍。
- 用 K3 KL 监控 RL 训练，区分"PPL 下降导致 KL 自然下降"和"模型快崩了 KL 飙升"两种状态。
- 在 RL 调试报告里给出 IS clip ratio 分布、veto 命中率、batch-normalize 前后的有效更新幅度。

## 配套源码研读（可选）

- `github_repo/slime/examples/train_infer_mismatch_helper/mis.yaml` —— slime 真实生产配置
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/slime/mismatch/blog-cn.md` —— 算法与实验全文
