# 源码带读：L19 Data Engineering

这份带读按“patch 合同 -> MiniInfra 算法 -> shard pipeline -> smoke 入口”组织。每一步都要回答：输入是什么，状态如何变化，输出给谁消费。

## 0. 源码地图

```text
labs/l18_data_engineering/patch/starter/dedup_minhash.py
labs/l18_data_engineering/patch/reference/dedup_minhash.py
labs/l18_data_engineering/patch/tests/test_patch.py
mini_infra/data/minhash_dedup.py
mini_infra/data/wds_pipeline.py
mini_infra/data/shard_resume.py
labs/l18_data_engineering/scripts/run_smoke.py
labs/l18_data_engineering/scripts/bench_wds.py
labs/l18_data_engineering/scripts/corrupt_drill.py
```

## 1. Patch starter

文件：`labs/l18_data_engineering/patch/starter/dedup_minhash.py`

- L20-L24：char-level shingle 的边界，短文本返回单元素集合。
- L27-L30：用 seed 生成 `(a,b)`，保证每一维签名不同。
- L36-L45：`minhash_signature` 的算法步骤。
- L46-L62：TODO 提示：基础哈希、num_perm 循环和最小值。
- L65-L72：`jaccard_estimate` 只比较相同位置比例。
- L75-L95：first-occurrence dedup 的朴素 O(N^2) 骨架。

读完要能回答：三个 patch 函数分别负责哪一段状态转换。

## 2. Patch reference

文件：`labs/l18_data_engineering/patch/reference/dedup_minhash.py`

- L13-L16：生成 char-level shingles。
- L19-L21：稳定地产生每个 seed 的 universal hash 参数。
- L24-L31：把 shingle 集合变成基础 hash 列表。
- L32-L37：每个 permutation 取一次最小哈希，形成签名。
- L40-L45：签名相同位置比例就是 Jaccard 估计。
- L48-L56：先为全部文本计算签名，再准备 keep 列表。
- L57-L62：当前样本只要撞上已保留签名，就按重复处理。

可以先跳过：类型注解和 import。主线是 text -> shingles -> signature -> keep indices。

## 3. Patch tests

文件：`labs/l18_data_engineering/patch/tests/test_patch.py`

- L21-L24：签名长度必须等于 `num_perm`。
- L27-L33：同一文本重复签名必须一致。
- L36-L43：近重复文本的估计值必须高。
- L45-L49：无关文本的估计值必须低。
- L52-L60：完全重复样本按 first-occurrence 删除。
- L63-L72：完全不同文本不应被误删。

读完要能回答：测试失败时先查签名、估计还是 dedup 策略。

## 4. MiniInfra MinHash

文件：`mini_infra/data/minhash_dedup.py`

- L30-L39：word-level width-gram shingle，并说明 width 取值影响误差。
- L42-L44：用 sha1 和 seed 做稳定哈希。
- L47-L51：每个 seed 取最小 hash，生成签名。
- L54-L59：按相同位置比例计算签名相似度。
- L62-L67：直接计算真实 Jaccard，供 smoke 对照。
- L70-L76：枚举 pairs，超过阈值时准备记录 near duplicate。
- L77-L83：duplicate pair payload 记录左右索引、估计值和真实 Jaccard。
- L84-L89：输出 num_docs、threshold、duplicate_pairs 和 dedup_ratio。

读完要能回答：MiniInfra 的 `dedup_ratio` 是怎么来的。

## 5. WDS pipeline

文件：`mini_infra/data/wds_pipeline.py`

- L44-L50：`simulate_pipeline` 接收 shard、worker、prefetch、shuffle 和 corrupt_count。
- L51-L56：detshuffle、corrupt shard、recovery、dedup 和 throughput 统一计算。
- L57-L67：返回吞吐、p99 latency、dedup_ratio、recovery 和 duplicate pairs。
- L70-L82：CLI 参数进入 run 目录和 artifact。
- L83-L87：写出 WDS throughput 和 dedup report。

读完要能回答：哪些指标属于数据质量，哪些指标属于供给性能。

## 6. Shard resume

文件：`mini_infra/data/shard_resume.py`

- L28-L40：`recovery_plan` 输入 shard list、corrupt set 和 resume cursor。
- L41-L51：cursor 前的 shard 被标记为 before_resume_cursor，坏 shard 被标记为 corrupt。
- L52-L59：返回 recovered、skipped、lost_samples 和 resume_cursor。
- L62-L67：detshuffle 用 seed 和 shard 名生成稳定顺序。

读完要能回答：重启后重复读样本时，应该检查 cursor 还是 dedup threshold。

## 7. Lab scripts

文件：`labs/l18_data_engineering/scripts/run_smoke.py`

- L13-L20：CLI 只负责把 run id 和 mode 交给半任务 runner。

文件：`labs/l18_data_engineering/scripts/bench_wds.py`

- L12-L22：workers 和 prefetch 参数进入 `simulate_pipeline` 并打印 JSON。

文件：`labs/l18_data_engineering/scripts/corrupt_drill.py`

- L12-L16：构造三个 shard，并把指定 shard 放入 corrupt set。

读完要能回答：patch-test、bench 和 corrupt drill 分别验证哪一层。

## 自检问题

1. `num_perm` 影响哪些成本？
2. 为什么 `dedup` 返回索引比返回文本更适合训练数据？
3. `wds_throughput_mbs`、`dedup_ratio`、`lost_samples` 分别回答什么问题？
4. fixed seed 下 detshuffle 为什么要保持 byte-equal 顺序？
