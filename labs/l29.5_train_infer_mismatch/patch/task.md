# L32 Patch · Train-Infer Mismatch 修正算子

## 你要交付什么

实现 6 个可独立测试的 PyTorch 算子：

```python
def compute_k3_kl(logp_p, logp_q) -> Tensor: ...
def tis_correct(logp_old, logp_new, advantages, lo=0.5, hi=2.0) -> Tensor: ...
def mis_with_mask(logp_old, logp_new, advantages, lo=0.5, hi=2.0) -> Tensor: ...
def geometric_seq_is(logp_old, logp_new, seq_lens) -> Tensor: ...
def apply_veto(logp_rollout, threshold=1e-6) -> Tensor: ...
def batch_normalize_weights(weights) -> Tensor: ...
```

禁止使用 `torch.distributions.kl_divergence`。本讲只有采样 token 的 logprob，不掌握完整类别分布。允许使用 `torch.exp`、`torch.clamp`、`torch.where`、`mean`、`sum` 和基础张量操作。

## 概念地图

```text
rollout logprob       training logprob
       |                    |
       '---- log_ratio -----'
                |
          ratio = exp(log_ratio)
                |
   +------------+-------------+
   |            |             |
 K3 KL       TIS/MIS     Geometric IS
 monitor     token 修正   sequence 权重
                |
              Veto
                |
        Batch Normalize
                |
          policy loss 权重
```

## 接口契约

### 1. `compute_k3_kl(logp_p, logp_q) -> Tensor`

K3 KL 使用：

```text
log_ratio = logp_p - logp_q
ratio = exp(log_ratio)
k3 = ratio - 1 - log_ratio
```

返回 `k3.mean()`。aligned 输入应返回 0，随机输入下结果应非负。

### 2. `tis_correct(logp_old, logp_new, advantages, lo, hi) -> Tensor`

Truncated IS：

```text
ratio = exp(logp_new - logp_old)
output = clamp(ratio, lo, hi) * advantages
```

输出 shape 与 `advantages` 一致。ratio 超过上界时仍保留梯度贡献，但贡献被压到上界。

### 3. `mis_with_mask(logp_old, logp_new, advantages, lo, hi) -> Tensor`

Masked IS：

```text
ratio = exp(logp_new - logp_old)
mask = (lo <= ratio <= hi)
output = ratio * mask * advantages
```

ratio 越界时输出为 0。它更适合切断不可信 token 的贡献。

### 4. `geometric_seq_is(logp_old, logp_new, seq_lens) -> Tensor`

输入 `logp_old` 和 `logp_new` 是 `(B, T)`，`seq_lens` 是 `(B,)`，返回 `(B,)`：

```text
weight = exp(mean_valid_tokens(logp_new - logp_old))
```

必须用 `seq_lens` 生成 mask，padding 位置不能进入均值。`torch.arange(T)` 要放在同一 device 上。

### 5. `apply_veto(logp_rollout, threshold=1e-6) -> Tensor`

用 log 空间比较：

```text
keep = logp_rollout >= log(threshold)
```

保留返回 1，触发 veto 返回 0。不要先 `exp(logp_rollout)` 再比较。

### 6. `batch_normalize_weights(weights) -> Tensor`

返回：

```text
weights / weights.mean().clamp(min=1e-12)
```

归一化后均值应为 1，相对顺序保持不变。

## 不变量

1. `logp_p == logp_q` 时 K3 KL 为 0。
2. K3 KL 对任意正 ratio 非负。
3. TIS 的输出绝对值不会超过 `hi * abs(advantages)`。
4. MIS 在 ratio 越界时输出为 0。
5. Geometric IS 输出 shape 是 `(B,)`。
6. Veto 使用 log 阈值比较。
7. Batch Normalize 后的权重均值为 1。

## 怎么验证

```bash
make patch-test M=l29.5_train_infer_mismatch
```

10 个 CPU 测试：

| 测试 | 验证 |
|---|---|
| `test_k3_kl_zero_when_aligned` | logprob 一致时 K3 为 0 |
| `test_k3_kl_formula` | `logp=-1, logq=-2` 时 K3 为 `e - 2` |
| `test_k3_kl_nonnegative` | 随机输入下 K3 非负 |
| `test_tis_clips_high_ratio` | 高 ratio 被 clamp 到上界 |
| `test_tis_passes_through_in_range` | ratio 为 1 时 advantage 不变 |
| `test_mis_masks_outliers` | 越界 token 被置零 |
| `test_geometric_seq_is_matches_definition` | 只统计有效 token，输出 `(B,)` |
| `test_veto_drops_extreme_low_prob` | 低于阈值的 logprob 被 veto |
| `test_batch_normalize_mean_one` | 归一化后均值为 1 |
| `test_batch_normalize_preserves_relative_order` | 相对顺序保持 |

## 卡住怎么办

1. 先跑 `IMPL=reference make patch-test M=l29.5_train_infer_mismatch`，确认测试环境正常。
2. 再对照 `labs/l29.5_train_infer_mismatch/source_walkthrough.md`，找到失败测试对应的函数。
3. 最后用一两个小 tensor 手算公式方向，特别检查 `logp_new - logp_old`。

## 写完之后你能做什么

- 解释 K3 KL、ratio、TIS/MIS、Geometric IS、Veto 和 Batch Normalize 如何连接。
- 看懂 SLiME `examples/train_infer_mismatch_helper/mis.py` 中的主要分支。
- 在 RL debug 报告中补齐 ratio 分布、veto 命中率和 batch norm factor。
