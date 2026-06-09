# L19 Source Reading Card

## 主路径

1. `labs/l18_data_engineering/patch/starter/dedup_minhash.py`：学生实现的最小 MinHash 合同。
2. `labs/l18_data_engineering/patch/reference/dedup_minhash.py`：reference 的 shingle、signature、estimate、dedup。
3. `labs/l18_data_engineering/patch/tests/test_patch.py`：5 个行为边界。
4. `mini_infra/data/minhash_dedup.py`：MiniInfra 版近重复检测。
5. `mini_infra/data/wds_pipeline.py`：WebDataset-style throughput、detshuffle、dedup 和 recovery smoke。
6. `mini_infra/data/shard_resume.py`：corrupt shard、resume cursor、detshuffle order。
7. `labs/l18_data_engineering/scripts/run_smoke.py`：半任务 smoke 入口。
8. `labs/l18_data_engineering/scripts/bench_wds.py`：WDS pipeline 参数入口。
9. `labs/l18_data_engineering/scripts/corrupt_drill.py`：坏 shard recovery 入口。

## 阅读方法

1. 先读 reference，确认数学合同。
2. 再读 tests，知道 patch-test 只覆盖哪些边界。
3. 然后读 MiniInfra，观察同一机制如何进入数据 pipeline metrics。
4. 最后读脚本，确认 smoke 和 drill artifact 从哪里生成。

## 自检

- 我能否手算两个短文本的 shingle 集合和 Jaccard？
- 我能否解释 `num_perm` 增加时成本和误差怎样变化？
- 我能否说明 dedup、WDS throughput、shard recovery 三类 artifact 各自回答什么问题？
