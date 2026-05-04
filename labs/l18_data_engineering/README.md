# L06.3 · 数据工程：MinHash 文本去重

> 本关只做一件事：**实现 MinHash + Jaccard 估计 + first-occurrence dedup**。

写完这关你能在 Capstone 给图像 caption 数据集做去重，避免训练集污染 eval。

## 闭环

```bash
cat labs/l18_data_engineering/patch/task.md
$EDITOR labs/l18_data_engineering/patch/starter/dedup_minhash.py
make patch-test M=l18_data_engineering
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_signature_size` | 签名长度 = num_perm |
| `test_identical_text_same_signature` | 相同文本签名相等，jaccard=1 |
| `test_jaccard_estimate_high_for_near_duplicates` | 改 1 词 jaccard > 0.7，无关文本 < 0.2 |
| `test_dedup_removes_exact_duplicates` | 完全重复样本被去掉 |
| `test_dedup_preserves_distinct` | 完全不同的文本不被错杀 |

## 卡住怎么办

1. 看 `notebooks/n16_minhash_dedup.ipynb`。
2. `make patch-hint M=l18_data_engineering`。
3. `make patch-show-solution M=l18_data_engineering`。

## 进入下一关

`make patch-test` 全绿后，继续做源码理解口试。下一关 [L07 vLLM baseline](../l19_vllm_serving_baseline/README.md) 让你给 vLLM-shape engine 加 typical_p 采样。
