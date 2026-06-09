# L08 · Megatron 数据预处理：把 JSONL 写成 `.bin/.idx`

这一讲解决 Megatron 预训练的数据入口问题：原始 JSONL 文本不能直接进入大规模训练热路径。训练时需要按 token id 顺序读取、按样本或 sequence 随机访问，并让多个 DataLoader worker 低开销共享同一份数据文件。Megatron 的 `.bin/.idx` IndexedDataset 就是在解决这个问题。

本讲的 lab 实现一个教学版 `.bin/.idx`：把 JSONL 中的 `text` 字段 tokenize，顺序写入 `<prefix>.bin`，再把 dtype、样本数、token 总数、offsets 和 lengths 写入 `<prefix>.idx`。然后用 `IndexedDataset[i]` 通过 mmap 读回第 i 条样本。

## 学习路线

建议按下面顺序走，先把数据格式和训练入口讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L08 在训练数据和 Megatron 预训练主线里的位置。
2. 读 [lecture.md](lecture.md)：理解 JSONL、tokenizer、`.bin/.idx`、offset/length、dtype、mmap 和真实 Megatron preprocess。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch reference、tests、drill、MiniInfra 和 Megatron 源码路径读。
4. 做 quiz：确认 IndexedDataset 文件格式、随机访问和 dtype 边界。
5. 做 patch：实现 `text_to_megatron_bin()` 和 `IndexedDataset`。
6. 跑 drill：生成 demo JSONL，写出 `.bin/.idx`，抽样读回并产出 Megatron 命令模板。
7. 填写 [outputs/data_pipeline_template.md](outputs/data_pipeline_template.md)，记录数据规模、dtype、样本数、token 数和 artifact。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training data pipeline / Megatron pretraining |
| 它解决什么问题 | 把原始 JSONL 文本离线转成 Megatron 训练可直接消费的 IndexedDataset |
| 它连接哪些指标或证据 | n_samples、total_tokens、dtype、bin bytes、idx header、sample roundtrip、preprocess docs/s |
| 它连接哪些源码 | `patch/reference/megatron_bin.py`、`scripts/run_preprocess.py`、`mini_infra/data/indexed_dataset.py`、Megatron `tools/preprocess_data.py`、Megatron `indexed_dataset.py` |
| lab 检验什么 | 写出 `.bin/.idx`、idx header 正确、样本 roundtrip、空文本跳过、dtype 越界报错、mmap 读取 |

## 你会学到什么

- 为什么预训练不会在每个 step 里实时 tokenize 原始 JSONL。
- `.bin` 和 `.idx` 分别保存什么，二者怎样配合做随机访问。
- offsets 为什么以字节为单位，lengths 为什么以 token 个数为单位。
- dtype 选择如何影响文件大小和 token id 上限。
- 空文本、缺失 `text` 字段、空 tokenizer 输出和 token 越界应该怎样处理。
- `np.memmap` 怎样让 reader 不在 `__getitem__` 里全量加载 `.bin`。
- 真实 Megatron preprocess 如何加入 worker pool、json key、sentence split、append_eod、document index 和 dtype 自动选择。

## Patch 闭环

```bash
cat labs/l07_dataset_megatron_bin/patch/task.md
$EDITOR labs/l07_dataset_megatron_bin/patch/starter/megatron_bin.py
make patch-test M=l07_dataset_megatron_bin
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_writes_bin_idx_pair` | 输出 `<prefix>.bin` 和 `<prefix>.idx`，summary 字段正确 |
| `test_idx_header_magic_version` | idx 文件头 magic、version、dtype code 正确 |
| `test_roundtrip_get_sample` | `IndexedDataset[i]` 与 tokenizer 输出一致 |
| `test_total_tokens_matches_concat` | `.bin` 字节数等于 token 数乘 dtype bytes |
| `test_empty_lines_skipped` | 空 text、缺失 text 和空行不写入样本 |
| `test_dtype_uint16_overflow_raises` | `uint16` 下 token id 超界时显式报错 |

## Drill 闭环

```bash
bash labs/l07_dataset_megatron_bin/scripts/run_preprocess.sh
```

默认配置 [configs/cpu_demo.yaml](configs/cpu_demo.yaml) 会：

- 生成 `data/wikitext_demo/train.jsonl` demo 输入。
- 写出 `data/wikitext_demo/indexed/wikitext_text_document.bin` 和 `.idx`。
- 用 `IndexedDataset` 抽样读回几条样本。
- 写出 `runs/l07_dataset_megatron_bin/<run-id>/artifacts/preprocess_summary.json`。
- 写出 `artifacts/real_megatron_command.sh`，展示后续 `pretrain_gpt.py --data-path <prefix>` 的连接方式。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查样本数异常、token 数异常、dtype 越界、idx/bin 不匹配和 Megatron 读取失败 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速复习 patch、drill、MiniInfra 和 Megatron 源码主路径 |
| [outputs/data_pipeline_template.md](outputs/data_pipeline_template.md) | 记录一次预处理运行的输入、输出、指标、命令和判断 |

## 进入下一讲

`make patch-test M=l07_dataset_megatron_bin` 通过，并完成一次 `run_preprocess.sh` 复盘后，进入 [L09 Megatron 文本预训练](../l08_megatron_text_pretrain/README.md)。下一讲会把这个 `--data-path` 接到训练 loop。
