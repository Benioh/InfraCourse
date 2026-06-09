# L08 源码带读：Megatron IndexedDataset 数据链路

这份带读按“教学格式到真实 Megatron”的顺序走。先读 [lecture.md](lecture.md)，再打开源码。每一步只看少量代码，先抓主路径，再看生产分支。

## 0. 源码地图

```text
labs/l07_dataset_megatron_bin/patch/starter/megatron_bin.py
  -> 学生要实现的最小写入和读取接口

labs/l07_dataset_megatron_bin/patch/reference/megatron_bin.py
  -> 教学版 .bin/.idx 的完整状态推进

labs/l07_dataset_megatron_bin/patch/tests/test_patch.py
  -> 文件格式、roundtrip、空文本和 dtype 边界

labs/l07_dataset_megatron_bin/scripts/run_preprocess.py
  -> demo 输入、artifact、sample dump 和 Megatron 命令模板

mini_infra/data/indexed_dataset.py
  -> 更小的文本 indexed dataset，用于理解 bin/idx 分工

github_repo/Megatron-LM/tools/preprocess_data.py
  -> 真实 preprocess 的 tokenizer worker 和 builder 调用

github_repo/Megatron-LM/megatron/core/datasets/indexed_dataset.py
  -> 真实 IndexedDatasetBuilder、IndexWriter 和 IndexReader
```

## 1. Starter：先看接口边界

文件：[megatron_bin.py](patch/starter/megatron_bin.py)

先看第 17-23 行。`text_to_megatron_bin` 的四个输入正好对应预处理运行的四个变量：输入 JSONL、tokenizer、输出 prefix 和 dtype。

再看第 24-31 行。TODO 列出的顺序就是写入主路径：校验 dtype、打开 `.bin`、遍历 JSONL、tokenize、range check、写 tokens、记录 offsets/lengths、写 `.idx`、返回 summary。

看第 35-43 行。`IndexedDataset.__init__` 要读 `.idx`，恢复 dtype、样本数、token 总量、offsets、lengths，并 mmap `.bin`。

最后看第 45-52 行。`__len__` 只返回样本数；`__getitem__` 必须用 offset 和 dtype bytes 算出 mmap 切片范围。

## 2. Reference：按写入流程读

文件：[megatron_bin.py](patch/reference/megatron_bin.py)

先看第 21-35 行。reference 先校验 dtype，创建输出目录，确定 `.bin/.idx` 路径，拿到 dtype bytes、NumPy dtype 和 token id 上限。

看第 37-48 行。循环读取 JSONL，跳过空行、缺失 text 和空 text。这是数据清洗的最小边界。

看第 49-61 行。tokenizer 输出被转成 list，空 token 序列跳过，token id 做范围检查，然后写入 `.bin`。写完后记录当前 cursor 和 token length，再推进 cursor。

看第 63-72 行。reference 写 `.idx` header 和两张表。`offsets` 和 `lengths` 都按 `uint64` 写入。

看第 74-80 行。summary 是 drill 和 report 的证据入口，后续要用它判断样本数、token 数和输出路径。

## 3. Reference：按读取流程读

仍然是 [megatron_bin.py](patch/reference/megatron_bin.py)。

第 83-95 行读取 idx header，并校验 magic 和 version。这里能抓住错 prefix、旧格式或损坏 idx 文件。

第 96-105 行读取样本数、token 总量、dtype、offsets 和 lengths，然后用 `np.memmap` 打开 `.bin`。

第 118-123 行是读取样本的核心。下标先做边界检查，再用 `offset // dtype_bytes` 转成 token 下标，并切出 `length` 个 token。

读完 reference 后，应能解释为什么 offsets 用字节、lengths 用 token 数。

## 4. Tests：用断言反推格式合同

文件：[test_patch.py](patch/tests/test_patch.py)

第 32-40 行验证 `.bin/.idx` 都存在，summary 中样本数和 dtype 正确。

第 43-54 行直接读 idx header，确认 magic、version 和 dtype code。

第 57-66 行做 roundtrip。写入若干字符串，再用 `IndexedDataset[i]` 读回，结果必须等于 tokenizer 输出。

第 69-77 行验证 `.bin` 文件大小和 total tokens。int32 下文件字节数应等于 token 数乘 4。

第 80-90 行验证空文本、缺失 text 和空行都会被跳过，最终只保留两条有效样本。

第 93-100 行验证 `uint16` 越界。token id 70000 必须抛 `ValueError`。

## 5. Drill 脚本：看 artifact 如何落盘

文件：[run_preprocess.py](scripts/run_preprocess.py)

第 33-44 行选择实现。脚本支持通过 `IMPL` 选择 starter 或 reference；未指定时优先尝试 starter，再回退 reference import。

第 47-60 行生成 demo JSONL。它构造多行 `{"text": ...}`，让 preprocess 有稳定输入。

第 63-72 行构造 tokenizer。默认 `char_mod_30k` 是演示用 tokenizer；配置为 `hf` 时会加载 HuggingFace tokenizer。

第 81-89 行读取配置、创建 run 目录，并写入 resolved config。真实复盘必须保留这一步，否则数据输出很难复现。

第 103-110 行调用 `text_to_megatron_bin`，输出 prefix 来自配置。

第 112-118 行用 `IndexedDataset` 抽样读回几条样本，并保存 token 长度和头部 token。

第 126-133 行生成真实 Megatron 命令模板。注意 `--data-path` 指向 prefix。

第 135-143 行写 metrics。`n_samples`、`total_tokens`、输出路径和 accepted 状态都进入 JSONL。

## 6. MiniInfra：更小的 indexed dataset

文件：[indexed_dataset.py](../../mini_infra/data/indexed_dataset.py)

第 13-25 行展示一个更小的思路：读 JSONL、把文本 bytes 写入 `.bin`，把 id、offset 和 length 写进 `.idx` JSON。它不是 Megatron 格式，但说明 bin/idx 分工可以抽象成“内容文件 + 索引文件”。

第 34-40 行是 CLI 入口。它接收 input 和 output-prefix，调用 builder，再打印 JSON summary。

MiniInfra 用文本 bytes 而不是 token ids，适合作为概念对照。L08 patch 则更接近 Megatron token id 数据路径。

## 7. Megatron preprocess_data：真实 tokenizer worker

文件：[preprocess_data.py](../../github_repo/Megatron-LM/tools/preprocess_data.py)

第 47-59 行看 `Encoder.initializer`。真实 preprocess 会在 worker 进程里构造 tokenizer，并按配置处理 sentence splitter。

第 86-108 行看 `Encoder.encode`。它读取 JSON 中的 key，把文本切成句子列表，调用 tokenizer，必要时追加 EOD，并返回 token ids 和 sentence lengths。

第 151-160 行看 `Partition.process_json_file` 的入口。它打开输入文件，构造 encoder/tokenizer，并用 multiprocessing pool 并发 encode。

第 166-178 行创建每个 json key 对应的输出文件和 `IndexedDatasetBuilder`。dtype 会根据 tokenizer vocab size 选择。

第 184-194 行把 encoded docs 写入 builder，并在结束时 finalize idx。

读完这部分要得到一个结论：真实 preprocess 的核心仍然是 “json -> tokens -> builder.add_document -> finalize”。

## 8. Megatron indexed_dataset：真实格式边界

文件：[indexed_dataset.py](../../github_repo/Megatron-LM/megatron/core/datasets/indexed_dataset.py)

第 107-119 行看 `DType.optimal_dtype`。vocab cardinality 小于阈值时可用 `uint16`，否则使用 `int32`。

第 141-151 行看 `_IndexWriter.__enter__`。真实 idx 会写固定 header、version 和 dtype code。

第 190-207 行看 `_IndexWriter.write`。真实 idx 会写 sequence count、document count、sequence lengths、sequence pointers 和 document indices。

第 224-230 行看 `_sequence_pointers`。它按 length 和 dtype size 计算每条 sequence 的字节指针，这和本关 offsets 逻辑同构。

第 264-278 行看 `_IndexReader` 的 header 读取。它校验 header/version，恢复 dtype、sequence count 和 document count。

第 291-299 行看 reader 从 idx buffer 中恢复 sequence pointers。真实 reader 也是先读索引，再定位 bin。

第 937-963 行看 `IndexedDatasetBuilder` 初始化。builder 打开 bin 文件，保存 dtype，并初始化 sequence lengths 和 document indices。

第 979-995 行看 `add_document`。它把 document tokens 写进 data file，并扩展 sequence lengths 和 document boundary。

第 1029-1037 行看 `finalize`。builder 关闭 data file，并通过 `_IndexWriter` 写 idx。

## 9. 读完后的自检问题

1. `.bin` 和 `.idx` 的职责分别是什么？
2. 本关 offset 为什么用字节，读取时为什么要除以 dtype bytes？
3. 哪些测试能证明 roundtrip 正确？
4. `uint16` 越界会在写入的哪一步被发现？
5. 真实 Megatron 相比本关 patch 多了哪些 idx 字段？
6. `preprocess_data.py` 中 tokenizer worker 和 builder 的交界在哪里？
