# L03.5 · Megatron `.bin/.idx` 数据预处理：把 JSONL 切成 Megatron 能直接消费的 IndexedDataset

> 本关只做一件事：**把一份 jsonl 文本数据集 tokenize → 写成 Megatron 风格的 `.bin/.idx`**，
> 然后再实现 `IndexedDataset` 读出第 i 条样本。

之前学生没有"如何让 Megatron 真正消费数据"的概念。L03.5 把这一段补完，使得 L04 / L04.8
的真实预训练命令能够直接 `--data-path data/<prefix>` 跑起来。

## 闭环

```bash
cat labs/l07_dataset_megatron_bin/patch/task.md
$EDITOR labs/l07_dataset_megatron_bin/patch/starter/megatron_bin.py
make patch-test M=l07_dataset_megatron_bin
bash labs/l07_dataset_megatron_bin/scripts/run_preprocess.sh
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_writes_bin_idx_pair` | 输出 `<prefix>.bin` + `<prefix>.idx` |
| `test_idx_header_magic_version` | idx 文件头 magic / version / dtype 字段 |
| `test_roundtrip_get_sample` | `IndexedDataset[i]` 与原 token 序列完全一致 |
| `test_total_tokens_matches_concat` | bin 文件长度等于所有样本 token 数 × 4 |
| `test_empty_lines_skipped` | jsonl 中空 text 字段必须跳过 |
| `test_dtype_int32_default` | 默认 dtype int32（够覆盖 32k vocab） |

## Drill

`scripts/run_preprocess.py` 在 `data/wikitext_demo/` 上跑：

1. 读 `data/wikitext_demo/train.jsonl`（脚本会自动生成一份 100 行的 demo）
2. 用 patch 的 `text_to_megatron_bin` 写 `data/wikitext_demo/indexed/wikitext_text_document.{bin,idx}`
3. 用 patch 的 `IndexedDataset` 抽样读 5 条，与原 jsonl 比对
4. 打印 Megatron `pretrain_gpt.py --data-path data/wikitext_demo/indexed/wikitext_text_document` 的命令模板

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_demo.yaml` | 默认 demo：100 行假数据 |
| `configs/fineweb_edu_50m.yaml` | 真实 FineWeb-Edu 50M token 切片 |

## 为什么这关重要

很多学生学完 L04 还是不会"数据从哪来"——因为他们没看过 `.bin/.idx` 怎么写出来的。
学完本关你可以独立把任意 HuggingFace 数据集喂给 Megatron。

## 进入下一关

通过后进入 [L04 Megatron 文本预训练](../l08_megatron_text_pretrain/README.md)。
