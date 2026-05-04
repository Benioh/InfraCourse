# L06.3 Patch · MinHash 文本去重

## 你要交付什么

实现 MinHash 签名 + Jaccard 估计 + 简单 dedup——image-caption 训练集去重的标准工具：

```python
def minhash_signature(text: str, num_perm: int = 64, n_gram: int = 3) -> List[int]: ...
def jaccard_estimate(sig1: List[int], sig2: List[int]) -> float: ...
def dedup(texts: List[str], threshold: float = 0.8, num_perm: int = 64) -> List[int]:
    """返回去重后保留的样本索引（first-occurrence）。"""
```

**禁止** 用 `datasketch` / `sklearn` 偷懒。
**允许** Python `hashlib`、`set`、`random` 等标准库。

补丁规模目标：50–80 行。

## 算法

**MinHash 签名**：
1. 把文本切成 char-level n-gram shingles（如 "hello world" + n=3 → {"hel", "ell", "llo", "lo ", "o w", " wo", "wor", "orl", "rld"}）
2. 用 num_perm 个不同的 hash 函数对每个 shingle 求哈希
3. 每个 hash 函数取所有 shingle 哈希的最小值 → 签名第 i 维

**Jaccard 估计**：
```
estimate = mean(sig1[i] == sig2[i] for i in range(num_perm))
```

**Dedup**：从前往后扫描；当前样本与已保留集合中任意签名的 jaccard ≥ threshold → 丢弃。

## 不变量

1. 完全相同的两段文本，签名完全相等，jaccard_estimate = 1.0。
2. 完全不同的文本（无任何 shingle 重叠），jaccard_estimate ≈ 0（< 0.05）。
3. dedup([a, a, b]) → 索引 [0, 2]（保留 a 一份 + b）。
4. num_perm 越大，jaccard_estimate 越接近真实 Jaccard。

## 怎么验证

```bash
make patch-test M=l18_data_engineering
```

5 个测试，CPU 即可：

| 测试 | 验证 |
|---|---|
| `test_signature_size` | 签名长度 = num_perm |
| `test_identical_text_same_signature` | 相同文本签名完全相等 |
| `test_jaccard_estimate_high_for_near_duplicates` | 近乎相同的文本估计 jaccard > 0.7 |
| `test_dedup_removes_exact_duplicates` | dedup([a, a, b]) 保留 [0, 2] |
| `test_dedup_preserves_distinct` | 完全不同的文本不被错杀 |

## 写完之后你能做什么

- 解释 MinHash 与 LSH 的数学原理（你已实现简化版）。
- 看懂大模型训练数据集（如 RedPajama / Dolma）的 dedup pipeline。
- 在 Capstone 给图像 caption 数据集做 dedup，避免训练集泄漏到 eval set。
