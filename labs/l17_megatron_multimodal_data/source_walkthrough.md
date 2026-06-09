# 源码带读：L18 Multimodal Data Collator

这份带读按“patch 合同 -> lab 数据管线 -> MiniInfra -> Megatron Energon”的顺序读。目标是找到 batch 合同和数据边界落在哪些函数里，不需要把每个文件从头读到尾。

## 0. 源码地图

```text
labs/l17_megatron_multimodal_data/patch/starter/multimodal_collate.py
labs/l17_megatron_multimodal_data/patch/reference/multimodal_collate.py
labs/l17_megatron_multimodal_data/patch/tests/test_patch.py
labs/l17_megatron_multimodal_data/scripts/build_manifest.py
labs/l17_megatron_multimodal_data/scripts/build_webdataset_shards.py
labs/l17_megatron_multimodal_data/scripts/run_multimodal_data_lab.py
mini_infra/data/manifest.py
github_repo/Megatron-LM/examples/multimodal/dataset_helpers.py
github_repo/Megatron-LM/examples/multimodal/dataloader_provider.py
```

## 1. Patch starter：先读学生要补齐的合同

文件：`labs/l17_megatron_multimodal_data/patch/starter/multimodal_collate.py`

建议阅读：

- L30-L35：函数签名，确认输入 batch 和两个控制参数。
- L38-L47：image placeholder 和 modal id 的构造提示。
- L48-L57：audio placeholder 按 `min(T, max_audio_frames)` 计算。
- L58-L68：`input_ids`、`attention_mask`、`modal_type_ids` 统一 pad。
- L70-L83：图像侧通道和 `image_batch_indices`。
- L84-L90：音频截断、pad 和 `audio_mask`。
- L91-L101：返回字段清单。

读完要能回答：哪些字段属于主序列，哪些字段属于 side channel。

## 2. Patch reference：看状态怎样一步步落到张量

文件：`labs/l17_megatron_multimodal_data/patch/reference/multimodal_collate.py`

建议阅读：

- L27-L34：每个样本创建 `ids` 和 `modal`。
- L35-L42：音频和文本段追加到同一条 per-sample sequence。
- L44-L52：计算 batch 最大长度，并填三张 `(B,S)` 表。
- L54-L64：只 stack 有图样本，并保存原 batch 行号。
- L66-L78：初始化音频 side tensor 列表，计算截断长度。
- L79-L89：pad mel、写 `audio_mask`、保存原 batch 行号。
- L90-L104：有音频时 stack，没有时保留 `None`，最后返回固定字段。

可以先跳过：类型注解细节和 import。主线是 sequence -> padding -> side tensors。

## 3. Patch tests：每个测试守住一条不变量

文件：`labs/l17_megatron_multimodal_data/patch/tests/test_patch.py`

建议阅读：

- L38-L49：纯文本 batch 的 side tensors 为空，modal 全 0。
- L52-L68：图像样本的前 16 个位置是 image placeholder。
- L71-L85：音频样本的主序列长度和 `audio_mask`。
- L88-L104：mixed batch 的四行长度如何决定 `S=19`。
- L106-L113：mixed batch 的 image/audio side tensors 和 batch indices。
- L115-L126：padding 位置 attention mask、PAD id、modal id。
- L129-L141：超长音频截断到 `max_audio_frames`。
- L144-L158：`pixel_values[k]` 对应原 batch 的图像。

读完要能回答：一个测试失败时，应该先检查哪一个返回字段。

## 4. Lab scripts：manifest、shard 和 smoke artifact

文件：`labs/l17_megatron_multimodal_data/scripts/build_manifest.py`

- L4-L13：图文 caption 样本写入 manifest 行。
- L14-L22：语音转文本样本写入 manifest 行。
- L23-L33：音频分类样本写入 manifest，并落成 JSONL。

文件：`labs/l17_megatron_multimodal_data/scripts/build_webdataset_shards.py`

- L7-L10：创建 shard 目录并打开 tar。
- L11-L20：每个样本写 metadata，再把所有文件打进同一个 shard。

文件：`labs/l17_megatron_multimodal_data/scripts/run_multimodal_data_lab.py`

- L50-L58：依次生成 toy raw files、manifest、shard 和 batch preview。
- L59-L73：运行 manifest validation、shard inspection 和 loader smoke。
- L83-L94：把 rows、missing、shard_count、sample_count 写入 metrics。
- L95-L104：把主要结果写入 report。

读完要能回答：patch-test 和 smoke 分别验证数据系统的哪一层。

## 5. MiniInfra manifest：教学版 schema check

文件：`mini_infra/data/manifest.py`

- L11-L18：构造 image-caption 样本，显式列出 image/audio 字段。
- L19-L25：构造 audio-text 样本。
- L26-L32：对 image/audio 路径做存在性检查。
- L33-L39：写出 manifest payload 并返回。

读完要能回答：manifest 层能发现什么问题，哪些问题要留给 collator 或 forward。

## 6. Megatron TaskEncoder：真实多模态 batch 的字段更复杂

文件：`github_repo/Megatron-LM/examples/multimodal/dataset_helpers.py`

- L35-L46：`ImageTaskSample` 保存 imgs、num_tiles、tokens、total_len、labels。
- L70-L90：`ImageTaskBatchPacked` 保存 batch 级 tokens、labels、imgs、num_tiles、length metadata。
- L168-L179：按图像尺寸、patch、tiling 等参数计算每个 tile 对应多少 image embedding。
- L195-L200：总长度由 text tokens 和 image tiles 共同决定。
- L381-L393：把 `<image-N>` tag 替换成统一 image token，并统计 tag 数。
- L496-L502：检查 image token 数、tile 数和 image tensor 数是否一致。
- L791-L798：batch 阶段把所有 sample 的图片 stack。
- L799-L818：tokens 和 labels pad 到 batch 统一长度。
- L851-L862：返回 Megatron 使用的 batch dataclass。

读完要能回答：L18 的 `pixel_values`、`image_batch_indices` 在真实 Megatron 中分别对应哪些更复杂的字段。

## 7. Megatron dataloader provider：rank、worker 和恢复状态

文件：`github_repo/Megatron-LM/examples/multimodal/dataloader_provider.py`

- L27-L43：创建 train dataset，并传入 task encoder、worker config 和 image decode 策略。
- L84-L92：只在指定 TP/PP rank 上跑 dataloader。
- L95-L104：非 dataloader rank 直接返回空 dataloader。
- L113-L123：用 DP rank/world size 构造 `WorkerConfig`，并创建 savable loader。
- L124-L141：load 时尝试恢复 dataloader state。
- L152-L165：wrapper 暴露 iterator 和 `save_state`。

读完要能回答：为什么多模态数据读取不仅是 collate 函数，还要考虑 rank 和 resume state。

## 读完后的自检问题

1. mixed batch 中 `S` 由哪一行决定？
2. `attention_mask` 和 `modal_type_ids` 的错误分别会影响哪个阶段？
3. Megatron TaskEncoder 在哪里检查 image token 数和 image tile 数一致？
4. dataloader state 恢复失败时，训练会出现重复读、漏读，还是 shape mismatch？
