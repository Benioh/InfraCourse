# L19 Debug Checklist

## 1. 固定现场

- 记录命令、git commit、数据路径、样本数、语言/来源分桶、shingle 宽度、`num_perm`、threshold。
- 保存 patch-test 输出、notebook 参数、smoke run 目录、`dedup_report.json`、`wds_throughput.json`、`shard_recovery_log.txt`。
- 如果来自真实训练，补充 shard 数、shard 大小、worker 数、prefetch、shuffle seed、resume cursor。

## 2. 按层定位

| 层 | 先看什么 | 常见结论 |
|---|---|---|
| 文本规范化 | lowercase、标点、Unicode、空文本 | 相同内容被格式差异拆散 |
| shingle | n-gram 宽度、短文本策略 | 宽度太小误删，太大漏召回 |
| signature | 稳定哈希、`num_perm`、seed | 不可复现或估计方差过大 |
| dedup | threshold、first-occurrence、抽样 pairs | dedup ratio 异常或保留索引错 |
| WDS pipeline | shard 大小、workers、prefetch | GPU 等数据或 IPC 过高 |
| recovery | corrupt shard、resume cursor、lost samples | 重复读、漏读或静默跳过 |

## 3. Dedup 排查顺序

1. 对两条完全相同文本检查签名是否逐维相同。
2. 对近重复和无关文本分别打印 `jaccard_estimate`。
3. 检查 `dedup` 返回的是保留索引，且 first-occurrence 稳定。
4. 抽样查看被删除 pairs，记录误删和漏删。
5. 分桶统计 dedup ratio，避免某个语言、来源或长度段被系统性误删。

## 4. Pipeline 排查顺序

1. 先看 `wds_throughput_mbs` 和 `p99_batch_latency_ms`。
2. 比较 workers 和 prefetch 变化前后的吞吐，不只看单次结果。
3. 检查 detshuffle 是否在固定 seed 下重复得到同一顺序。
4. 对 corrupt shard 确认 recovered、skipped、lost_samples 和 resume cursor。
5. 真实训练中发现重复读时，先查 dataloader/shard state，再查 collator。

## 5. 结束条件

- 最小命令能复现现象。
- 关键参数和 artifact 已落盘。
- 能把问题归到 shingle、signature、dedup、shard、worker 或 recovery 层。
- 结论写入 `data_pipeline_template.md`，并包含下一步动作。
