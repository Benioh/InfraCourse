# L29.5 Patch · Train-Infer Mismatch 修正算子组

## 你要交付什么

实现 6 个 RL 框架在线上真正用的 mismatch 修正算子：

```python
def compute_k3_kl(logp_p, logp_q) -> Tensor: ...
def tis_correct(logp_old, logp_new, advantages, lo=0.5, hi=2.0) -> Tensor: ...
def mis_with_mask(logp_old, logp_new, advantages, lo=0.5, hi=2.0) -> Tensor: ...
def geometric_seq_is(logp_old, logp_new, seq_lens) -> Tensor: ...
def apply_veto(logp_rollout, threshold=1e-6) -> Tensor: ...
def batch_normalize_weights(weights) -> Tensor: ...
```

**禁止** 用 `torch.distributions.kl_divergence`（那是 closed-form，不是 RL 里能拿到的形式）。
**允许** `torch.exp / log / clamp / mean / where` 等基础 op。

补丁规模目标：80–130 行 Python（不算注释空行）。

## 概念地图（写代码前先在脑子里建立）

```
       Rollout Engine (πSGLang)        Training Engine (πMegatron)
                |                                |
                ↓ generate                       ↓ forward
          y_t, logp_old(y_t)                logp_new(y_t)
                          \              /
                           ↓            ↓
                    ratio = exp(logp_new - logp_old)
                                |
        ┌───────────────────┬─┴─┬───────────────────┐
        ↓                   ↓   ↓                   ↓
    TIS：clamp 到 [lo,hi]   MIS：越界 mask 为 0   Geometric：序列级
        ↓                   ↓                       ↓
                  乘上 advantages，作为 loss 权重
                                |
                       Veto：极端低概率序列直接丢
                                |
                  Batch Normalize：均值 = 1，避免学习率震荡
```

## 接口契约

### 1. `compute_k3_kl(logp_p, logp_q) -> Tensor`

K3 KL（[Schulman 估计器](http://joschu.net/blog/kl-approx.html)）：

$$k_3(x) = \frac{p(x)}{q(x)} - 1 - \log\frac{p(x)}{q(x)}$$

输入是 log 形式（`logp_p, logp_q`），返回标量（mean over batch）。
**为什么不用 k1 = log(p/q)？** k3 总是非负，且方差更小，是 RL 监控的工业标准。

### 2. `tis_correct(logp_old, logp_new, advantages, lo, hi) -> Tensor`

Truncated IS：`ratio = exp(logp_new - logp_old)`，`clamp(ratio, lo, hi) * advantages`。
slime / verl 里 `lo, hi = 0.5, 2.0` 是 dense 模型的常用区间。

### 3. `mis_with_mask(logp_old, logp_new, advantages, lo, hi) -> Tensor`

Masked IS：`ratio = exp(logp_new - logp_old)`，`mask = (lo <= ratio <= hi)`，`ratio * mask * advantages`。
区别于 TIS：超界的 token 直接 **梯度归零** 而不是被截断到边界，避免越界点引入有偏更新。

### 4. `geometric_seq_is(logp_old, logp_new, seq_lens) -> Tensor`

序列级几何均值：

$$w_{\text{seq}} = \exp\left(\frac{1}{|y|}\sum_t \log\frac{\pi_{\text{new}}(y_t)}{\pi_{\text{old}}(y_t)}\right)$$

输入 shape：`logp_*` 是 `(B, T)`，`seq_lens` 是 `(B,)`，返回 `(B,)` 的序列权重。
**长度归一化**：避免序列越长权重波动越大。

### 5. `apply_veto(logp_rollout, threshold=1e-6) -> Tensor`

如果 rollout 给某个 token 的概率 < `threshold`（即 `logp_rollout < log(threshold)`），返回 mask=0（drop）。
**为什么需要它？** 极小概率的 token 让 ratio 爆炸到 1e6 量级，clip / mask 都来不及，必须直接 drop 整个序列。

### 6. `batch_normalize_weights(weights) -> Tensor`

`weights / weights.mean()`，确保有效学习率稳定。SNIS（Self-Normalized IS）的核心。

## 不变量（写代码时心里要装着）

1. K3 KL 在 `logp_p == logp_q` 时严格为 0。
2. K3 KL 永远 ≥ 0（数学性质，可以用作 sanity check）。
3. TIS 输出绝对值 ≤ `hi * |advantages|`。
4. MIS 在 `ratio` 越界时输出严格为 0。
5. Geometric IS 是序列级的（输出 shape = `(B,)`），与 token-level 不同。
6. Veto 用对数比较：`logp >= log(threshold)`，不要先 `exp` 再比，否则下溢。
7. `batch_normalize_weights` 后 `weights.mean() == 1`（数值精度内）。

## 怎么验证

```bash
make patch-test M=l29.5_train_infer_mismatch
```

7 个测试，全部 CPU：

| 测试 | 验证 |
|---|---|
| `test_k3_kl_zero_when_aligned` | `log_p == log_q` 时 K3 = 0 |
| `test_k3_kl_formula` | logp=-1, logq=-2 → K3 = e − 1 − 1 ≈ 0.7183 |
| `test_k3_kl_nonnegative` | 任意输入下 K3 ≥ 0 |
| `test_tis_clips_high_ratio` | ratio = e² 在 hi=2.0 下被 clamp |
| `test_mis_masks_outliers` | ratio = e⁵ 在 [0.5, 2.0] 外被 mask 为 0 |
| `test_geometric_seq_is_matches_definition` | 直接对照公式验证 |
| `test_veto_drops_extreme_low_prob` | log(1e-7) 触发 veto |
| `test_batch_normalize_mean_one` | 归一化后 mean 严格等于 1 |

## 卡住怎么办

1. 跑 `notebooks/n19_train_infer_mismatch.ipynb`：可视化 K3 KL 在 PPL 下降阶段先降后升的真实曲线。
2. `make patch-hint M=l29.5_train_infer_mismatch` —— 看 TODO 与公式提示。
3. `make patch-show-solution M=l29.5_train_infer_mismatch` —— 看参考解。

## 写完之后你能做什么

- 一眼看出 RL 训练日志里 K3 KL 突然飙升 = 即将崩溃，并能给出 TIS / MIS 配置建议。
- 解释为什么 MoE 模型的 train-infer mismatch 比 dense 模型严重 1–2 个数量级（提示：路由不一致 → 激活 expert 不一致）。
- 给 SLiME / verl 框架的 RL 调试报告里加上 ratio 分布直方图与 veto 命中率。
- 看懂 [slime examples/train_infer_mismatch_helper](https://github.com/THUDM/slime/tree/main/examples/train_infer_mismatch_helper) 的全部配置项。
