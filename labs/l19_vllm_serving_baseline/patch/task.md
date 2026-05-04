# L07 Patch · Typical_p Sampling（Locally Typical Sampling）

## 你要交付什么

实现 **Locally Typical Sampling**（Meister et al. 2022）—— vLLM / HuggingFace 生成质量更稳的非贪婪采样：

```python
def typical_p_filter(
    logits: torch.Tensor,         # (batch, vocab)
    typical_p: float = 0.9,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """Mask 掉"远离平均信息量"的 token，返回 filter 后的 logits。"""
```

**禁止** 用 `transformers.generation.logits_process.TypicalLogitsWarper`。
**允许** `torch.softmax` / `torch.log` / `torch.sort` / `torch.scatter` 等基础算子。

补丁规模目标：30–60 行。

## 算法

1. `probs = softmax(logits)`
2. `info = -log(probs)`（每个 token 的 surprisal）
3. `entropy = sum(probs * info)` → 每个 batch 的期望信息量
4. `dist = |info - entropy|` → 距期望信息量的偏差
5. 按 `dist` **升序**排序，沿 vocab 累加 prob mass
6. cumsum > typical_p 时 cut off：超过的 token 设为 filter_value
7. 返回 filter 后的 logits

直觉：保留 surprisal 接近期望值的 "typical" tokens，剔除既太不可能（dist 太大）也太确定（dist 太大向另一头）的 outlier。

## 不变量

1. `typical_p = 1.0` → 几乎所有 token 都被保留（cumsum 必然 > 1.0 末尾）。
2. `typical_p = 0.0`（极端）→ 只保留 dist 最小的 1 个 token。
3. 输出 logits 保持原 batch 维度，shape 不变；非 filter_value 位置数值不变。
4. 至少保留 1 个 token（避免空分布）。
5. 输入 fp32 / fp16 应都正常工作。

## 怎么验证

```bash
make patch-test M=l19_vllm_serving_baseline
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_typical_p_one_keeps_all` | typical_p=1.0 时所有 token 都保留 |
| `test_typical_p_zero_keeps_one` | typical_p=0.0 时只剩 1 个 token |
| `test_filter_value_applied` | 被过滤位置等于 filter_value |
| `test_kept_tokens_dist_minimal` | 保留的 token 的 dist 比丢弃的小 |
| `test_works_with_batch` | (B, V) 输入每行独立处理 |

## 写完之后你能做什么

- 解释 typical_p vs top_p vs top_k vs temperature 的差别。
- 在 SGLang / vLLM 里加自定义 sampling（你已掌握同款 API 模式）。
- Capstone Stage B 的多模态推理可以选 typical_p 提高生成多样性。
