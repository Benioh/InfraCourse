# L19 · Data Engineering: MinHash, WebDataset, Shard Resume

L19 讲训练数据进入大规模训练前的三件事：用 MinHash 找近重复文本，用 WebDataset-style shard 提升顺序读取效率，用 deterministic shuffle 和 resume cursor 处理坏 shard 与重启。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L19 在数据主线中的位置。
2. 读 [lecture.md](lecture.md)：从近重复问题、MinHash 数学、LSH banding 讲到 shard 与 resume。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra 和 smoke 脚本读源码。
4. 跑 notebook：[n16_minhash_dedup.ipynb](../../notebooks/n16_minhash_dedup.ipynb)。
5. 做 quiz：检查 shingle、Jaccard、num_perm、LSH 和数据泄漏。
6. 做 patch：实现 `minhash_signature`、`jaccard_estimate` 和 `dedup`。
7. 跑 smoke：生成 dedup、WDS throughput 和 shard recovery artifact。
8. 填写 [outputs/data_pipeline_template.md](outputs/data_pipeline_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Data pipeline |
| 解决的问题 | 文本近重复、数据供给吞吐、坏 shard 与重启恢复 |
| 上游 | raw text、caption、manifest、shard list |
| 下游 | tokenizer、DataLoader、训练 loop、eval leakage audit |
| patch 验收 | MinHash 签名、Jaccard 估计、first-occurrence dedup |

## 你会学到什么

- 把文本变成 shingle 集合，并解释 Jaccard 相似度。
- 用多组稳定哈希生成固定长度 MinHash signature。
- 用 signature hit ratio 估计 Jaccard，并说明 `num_perm` 的成本和误差。
- 用 first-occurrence 策略返回保留索引，维护 manifest 行号可追踪。
- 解释 WebDataset shard、detshuffle、prefetch、corrupt shard recovery 和 dedup 之间的边界。

## Patch 闭环

```bash
cat labs/l18_data_engineering/patch/task.md
$EDITOR labs/l18_data_engineering/patch/starter/dedup_minhash.py
IMPL=reference make patch-test M=l18_data_engineering
```

smoke：

```bash
python labs/l18_data_engineering/scripts/run_smoke.py --run-id l19_smoke
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 定位 dedup、shard、worker、resume cursor 问题 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch 和 MiniInfra 数据工程源码 |
| [outputs/data_pipeline_template.md](outputs/data_pipeline_template.md) | 记录一次数据工程 smoke 或训练排查 |

## 下一讲

L20 是样板课：vLLM Scheduler 与 KV Block Manager。
