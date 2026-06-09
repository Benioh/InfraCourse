# L08 讲义：Megatron `.bin/.idx` 数据预处理

这一讲讲训练数据如何进入 Megatron。

大模型预训练的输入通常先是 JSONL、Parquet 或 HF dataset。训练热路径需要的是 token id 序列，并且要能被多个 worker 反复、高吞吐地读取。如果每个 step 都从 JSONL 解析文本、调用 tokenizer、再构造 tensor，CPU 会成为瓶颈，训练结果也更难复现。Megatron 的做法是先离线预处理：把文本 tokenize，顺序写入 `.bin`，把样本边界和 dtype 等元数据写入 `.idx`，训练时用 `--data-path <prefix>` 读取这个 IndexedDataset。

本讲的 patch 是简化版。它保留最关键的数据合同：JSONL `text` 字段、tokenizer、dtype 检查、bin 拼接、idx header、offset/length 表和 mmap reader。真实 Megatron 会加入 worker pool、sentence split、append_eod、多 json key、document index、sequence pointer、storage client 和更复杂的采样逻辑。

## 1. 本讲目标

学完这一讲，你应该能回答：

1. 为什么 Megatron 预训练要先做离线数据预处理。
2. `.bin` 和 `.idx` 分别保存什么。
3. offsets、lengths、dtype bytes 怎样共同定位第 i 条样本。
4. 为什么 reader 要用 mmap，而不在 `__getitem__` 里全量加载。
5. `int32`、`uint16`、`int64` 在 token id 存储上有什么边界。
6. 真实 Megatron `tools/preprocess_data.py` 如何把 JSONL 变成 IndexedDataset。
7. 数据预处理复盘要记录哪些 artifact，才能接到后续 pretrain。

## 2. 真实问题：训练 loop 不该实时处理原始文本

先看一个朴素方案：

```text
training step
  -> read JSONL line
  -> json.loads
  -> tokenizer.encode
  -> pad / pack
  -> tensor
  -> forward
```

这个方案在教学 demo 中能跑，但在预训练里会带来几个问题。

第一，tokenizer 可能比模型 step 更不稳定。它有 Python 开销、依赖词表文件、特殊 token 配置和多进程初始化成本。

第二，随机访问困难。训练通常要 shuffle、按 sequence 取样、恢复 checkpoint 后继续从某个位置读。只靠 JSONL 行号很难直接映射到 token 序列边界。

第三，多 worker 共享成本高。如果每个 worker 都重复 parse JSON 和 tokenize，同一份数据会反复消耗 CPU。

第四，artifact 不够稳定。训练后很难证明某次实验到底用了哪个 tokenizer、哪些空文本被跳过、最终 token 总量是多少。

离线 IndexedDataset 把这些问题前移到 preprocess 阶段。训练 loop 只面对 token id 文件和索引文件，读取行为更稳定，artifact 也更可复查。

## 3. `.bin` 和 `.idx` 的分工

本关格式有两个文件：

```text
<prefix>.bin
<prefix>.idx
```

`.bin` 只保存 token id 的连续二进制数组。假设三条样本 tokenize 后分别是：

```text
[10, 11]
[20, 21, 22]
[30]
```

按 `int32` 写入 `.bin` 后，逻辑上就是：

```text
10, 11, 20, 21, 22, 30
```

`.idx` 保存怎样切回样本：

```text
offsets = [0, 8, 20]     # 字节偏移，int32 每个 token 4 字节
lengths = [2, 3, 1]      # token 个数
```

读取第 1 条样本时：

```text
byte_offset = offsets[1] = 8
start_token = byte_offset / 4 = 2
length = lengths[1] = 3
bin[2:5] = [20, 21, 22]
```

这个设计把大文件内容和样本边界拆开。`.bin` 可以 mmap，`.idx` 可以快速读入内存。训练时不需要扫描前面的样本，就能定位第 i 条。

## 4. 本关简化 idx 格式

patch task 规定了一个教学格式：

```text
magic        b"MGTRNIDX"
version      uint32 = 1
dtype_code   uint8
n_samples    uint64
total_tokens uint64
offsets      uint64[n_samples]
lengths      uint64[n_samples]
```

这里有几个关键点。

`magic` 用来拒绝错误文件。读到的前 8 字节不是 `MGTRNIDX`，reader 应该报错。

`version` 让未来格式升级有边界。本关只接受 1。

`dtype_code` 把 idx 和 bin 的 token id dtype 绑定起来。reader 必须用相同 dtype mmap `.bin`，否则同一串字节会被解释成错误 token。

`n_samples` 和 `total_tokens` 是总量证据。drill 和 report 要记录它们，后续训练也需要知道数据规模。

`offsets` 是每条样本在 `.bin` 中的起始字节。`lengths` 是 token 个数。二者不能混淆。

## 5. dtype 边界

本关支持三种 dtype：

| dtype | bytes/token | token id 上限 | 适用场景 |
|---|---:|---:|---|
| `uint16` | 2 | 65535 | 小 vocab，追求文件更小 |
| `int32` | 4 | 2^31 - 1 | 默认选择，给 special token 和较大 vocab 留余量 |
| `int64` | 8 | 2^63 - 1 | 很少需要，文件更大 |

dtype 影响两个东西：`.bin` 文件大小和 token id 能否表示。`uint16` 可以把文件压小一半，但只要 tokenizer 产出 70000，就必须报错。本关测试 `test_dtype_uint16_overflow_raises` 专门抓这个边界。

真实 Megatron 里有 `DType.optimal_dtype(cardinality)`，当 vocabulary cardinality 小于阈值时可选择 `uint16`，否则使用 `int32`。课堂默认 `int32`，是为了减少误配。

## 6. 写入流程

`text_to_megatron_bin(jsonl_path, tokenizer, output_prefix, dtype_str)` 的输入是 JSONL 路径、tokenizer 函数、输出 prefix 和 dtype。

写入主路径如下：

```text
validate dtype
open <prefix>.bin
for each json line:
  strip line
  skip empty line
  json.loads(line)
  text = payload.get("text", "")
  skip empty text
  tokens = tokenizer(text)
  skip empty tokens
  range check token ids
  write tokens to bin as dtype
  offsets.append(cursor)
  lengths.append(len(tokens))
  cursor += len(tokens) * dtype_bytes
write <prefix>.idx
return summary
```

输出 summary 至少要包含：

```python
{
    "n_samples": ...,
    "total_tokens": ...,
    "bin_path": "...",
    "idx_path": "...",
    "dtype": "int32",
}
```

这些字段会进入 drill artifact。后续训练排查时，`n_samples` 偏少通常先查空文本、json key 或 tokenizer 空输出；`total_tokens` 偏少则继续查清洗、dedup、截断和 tokenizer 配置。

## 7. 读取流程

`IndexedDataset(prefix)` 的构造函数读取 `<prefix>.idx`：

1. 校验 magic。
2. 校验 version。
3. 读取 dtype code。
4. 读取样本数和 token 总数。
5. 读取 offsets 和 lengths。
6. 用 dtype mmap `<prefix>.bin`。

`__getitem__(i)` 的输入是样本下标。它先做下标边界检查，然后：

```python
start = offsets[i] // dtype_bytes
length = lengths[i]
return mmap[start : start + length]
```

这个读取路径是 O(1) 定位样本边界，再按样本长度切片。它不会在每次读取时解析 JSON，也不会全量加载 `.bin`。

## 8. 真实 Megatron 源码怎样扩展这条链

真实 `tools/preprocess_data.py` 的 `Encoder` 做两件事：初始化 tokenizer，并把 JSONL 中配置的 key 转成 token id。开启 sentence split 时，它还会通过 NLTK 把文档切成句子；不开启时，`IdentitySplitter` 保留整段文本。

`Partition.process_json_file` 用 multiprocessing pool 并发 encode 输入。对每个 json key，它创建 `IndexedDatasetBuilder`，把 encoded docs 写进去。builder 负责写 `.bin` 和 `.idx`。

真实 `indexed_dataset.py` 的 `_IndexWriter` 和本关格式相似但字段更多。它会写固定 header、version、dtype code、sequence count、document count、sequence lengths、sequence pointers、document indices，以及可选 sequence modes。`IndexedDatasetBuilder.add_document` 会把文档 token 写入 data file，并记录 sequence lengths 和 document boundary。`finalize` 时才写 index。

这说明本关 patch 不是随意格式。它是把真实 IndexedDataset 中最容易理解的部分抽出来：二进制 token 文件、索引元数据、dtype 和 mmap 读取。

## 9. Drill 和 artifact 怎么看

运行：

```bash
bash labs/l07_dataset_megatron_bin/scripts/run_preprocess.sh
```

默认会用 `configs/cpu_demo.yaml`。脚本找不到输入时会生成 demo JSONL，然后调用 patch 或 reference 写数据。它会在 run 目录里保存：

- `config.resolved.yaml`：本次实际配置。
- `artifacts/preprocess_summary.json`：样本数、token 数、dtype、bin/idx 路径。
- `artifacts/sample_dump.json`：抽样读回的 token 长度和头部 token。
- `artifacts/real_megatron_command.sh`：后续真实 Megatron 的命令模板。
- `metrics.jsonl`：带时间戳的 preprocess 指标。
- `report.md`：本次复盘摘要。

验收时不要只确认文件存在。至少要检查：

1. `n_samples` 是否达到预期。
2. `total_tokens` 是否与输入规模同量级。
3. `sample_dump` 是否能读回非空样本。
4. `real_megatron_command.sh` 的 `--data-path` 是否指向 prefix，而不是 `.bin` 或 `.idx` 单个文件。

## 10. 生产边界

真实预处理会继续遇到这些问题：

- tokenizer 版本或 vocab 文件变化，导致 token id 不一致。
- 清洗、dedup、过滤空文本后样本数变化。
- append EOD、BOS/EOS、sentence split 改变 sequence 边界。
- 多 worker 处理顺序和输出 shard 合并。
- 大文件下 mmap、文件系统和 page cache 影响吞吐。
- 数据 prefix、train/valid/test split 和 checkpoint resume 的采样位置需要一致。

这些都不是 patch 要实现的内容，但复盘必须记录。预处理是训练可复现性的入口，数据 artifact 缺字段时，后续 loss、tokens/s 或 resume 问题会很难追。

## 11. 小结

L08 的核心链路是：

```text
JSONL text
  -> tokenizer
  -> token ids
  -> dtype checked binary concat
  -> idx metadata
  -> mmap random access
  -> Megatron --data-path
```

完成本讲后，你应该能把一份文本数据变成可复查的数据 artifact，并能解释真实 Megatron preprocess 源码在这条链上增加了哪些生产复杂度。
