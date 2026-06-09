# L19 Patch · MinHash Text Dedup

本 patch 验收 MinHash 文本去重的最小合同：把文本转成 shingle 集合，生成固定长度签名，用签名相同位置比例估计 Jaccard，并按 first-occurrence 策略返回保留索引。

## 需要实现

文件：`labs/l18_data_engineering/patch/starter/dedup_minhash.py`

```python
def minhash_signature(text: str, num_perm: int = 64, n_gram: int = 3) -> list[int]:
    ...

def jaccard_estimate(sig1: list[int], sig2: list[int]) -> float:
    ...

def dedup(
    texts: list[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    n_gram: int = 3,
) -> list[int]:
    ...
```

## 行为合同

1. `_shingles(text, n_gram)` 生成 char-level n-gram 集合；短文本返回 `{text}`。
2. `minhash_signature` 返回长度等于 `num_perm` 的整数 list。
3. 每个签名维度使用不同 seed 的 `(a, b)` 参数。
4. 基础哈希必须稳定，不能使用 Python 内置 `hash()`。
5. `jaccard_estimate` 要求两个签名长度相同，返回相同位置比例。
6. `dedup` 从前往后扫描，和任一已保留样本相似度大于等于 threshold 时丢弃当前样本。
7. `dedup` 返回原始文本列表中的保留索引，不返回文本内容。

## 验证

```bash
IMPL=reference make patch-test M=l18_data_engineering
```

测试覆盖：签名长度、相同文本稳定性、近重复/无关文本估计、完全重复删除、不同文本保留。
