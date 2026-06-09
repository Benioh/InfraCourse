# L08 Source Reading Card

这张卡用于快速复习 Megatron `.bin/.idx` 数据预处理的源码主路径。

## 1. Patch 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `labs/l07_dataset_megatron_bin/patch/starter/megatron_bin.py` | `text_to_megatron_bin`、`IndexedDataset` TODO | 本关要补齐写 bin、写 idx 和 mmap 读取 |
| `labs/l07_dataset_megatron_bin/patch/reference/megatron_bin.py` | dtype 校验、JSONL loop、offset/length、idx header、mmap | 教学版 IndexedDataset 的完整状态推进 |
| `labs/l07_dataset_megatron_bin/patch/tests/test_patch.py` | 6 个测试函数 | 测试覆盖文件存在、header、roundtrip、bytes、skip 和 dtype 越界 |

## 2. Drill 和 MiniInfra

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `labs/l07_dataset_megatron_bin/scripts/run_preprocess.py` | config、demo jsonl、tokenizer、summary、sample dump、Megatron command | drill 把 patch 结果变成可复查 artifact |
| `mini_infra/data/indexed_dataset.py` | `build_indexed_dataset` | bin/idx 分工可以简化成内容文件和索引文件 |

## 3. Megatron 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `github_repo/Megatron-LM/tools/preprocess_data.py` | `Encoder.initializer`、`Encoder.encode`、`Partition.process_json_file` | 真实 preprocess 用 worker pool tokenize，并把 doc 写进 builder |
| `github_repo/Megatron-LM/megatron/core/datasets/indexed_dataset.py` | `DType`、`_IndexWriter`、`_IndexReader`、`IndexedDatasetBuilder` | 真实 idx 多了 sequence/document 边界，但仍依赖 dtype、pointers 和 bin 文件 |

## 4. 最小不变量

- `.bin` 保存连续 token id，不保存 JSON 文本。
- `.idx` 保存 header、dtype、样本数、token 总数、offsets 和 lengths。
- offsets 是字节偏移。
- lengths 是 token 个数。
- dtype code 必须和 `.bin` 解释方式一致。
- `IndexedDataset[i]` 不全量加载 `.bin`。
- `--data-path` 传 prefix。

## 5. 自检

1. 我能否用一条样本手算 offset、length 和 `.bin` 字节数？
2. 我能否解释为什么 `uint16` 会溢出？
3. 我能否指出真实 Megatron 的 document indices 和本关 lengths/offsets 的关系？
4. 我能否说明 drill 产物如何连接到 L09 pretrain？
