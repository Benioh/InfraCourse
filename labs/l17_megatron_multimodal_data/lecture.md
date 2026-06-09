# L18：Multimodal Data Collator

多模态训练的第一个工程边界在数据层。一个 batch 里可能同时出现纯文本样本、图文样本、语音文本样本和图文语音样本。文本长度来自 tokenizer，图像长度来自 vision encoder 的 patch 或 tile 数，音频长度来自 mel 帧数。模型 forward 希望拿到形状规则、语义清楚的张量；collator 就负责把这些变长样本整理成同一份 batch 合同。

本讲的目标是把这个合同讲清楚。你要能手算 mixed batch 的 `input_ids.shape`，解释 `attention_mask` 和 `modal_type_ids` 的区别，说明真实图像和音频为什么走 side channel，并能读懂 Megatron multimodal TaskEncoder 如何把 tokens、imgs、labels、num_tiles 组织起来。

## 0. 学完后要能做什么

- 解释多模态 batch 中主序列和 side tensors 的分工。
- 按 `[image tokens][audio tokens][text ids]` 构造每个样本的序列。
- 正确设置 `attention_mask`、`modal_type_ids`、`pixel_values`、`audio_features` 和 batch indices。
- 用 patch tests 定位 padding、placeholder、音频截断、图像索引等错误。
- 从 Megatron Energon dataloader 源码里找到 sample 编码、batch padding、dataloader state restore 的位置。

## 1. 问题背景：多模态 batch 的长度来自多套规则

纯文本 batch 的长度主要由 tokenizer 输出决定。多模态 batch 还要加入视觉和音频位置。常见 VLM 会把一张图展开成多个 image embedding；语音模型会把音频转成一段 mel 或 encoder frame。于是同样是 10 个文本 token，带图样本和纯文本样本的序列长度可能完全不同。

本讲把系统拆成四层：

1. manifest：列出样本 id、任务类型和文件路径，先证明样本集合可读。
2. dataset / processor：读取单个样本，产出 `text_ids`、`image`、`audio_mel`。
3. collator：把一批单样本打包成 shape 对齐的 batch 字典。
4. forward：调用 image/audio encoder，并把 encoder 输出替换到 placeholder 位置。

如果缺文件，先查 manifest；如果单样本 shape 异常，查 dataset 或 processor；如果 batch 内 mask、padding、indices 错，查 collator；如果 placeholder 数和 encoder 输出数对不上，查 forward replacement。

## 2. Batch 合同：主序列和 side channel

L18 patch 的输入样本形状很小：

```python
{
    "text_ids": [10, 11, 12],
    "image": torch.Tensor(... ) or None,
    "audio_mel": torch.Tensor(T, 80) or None,
}
```

collator 返回八类字段：

| 字段 | 作用 |
|---|---|
| `input_ids` | 主序列，包含 image/audio placeholder 和真实 text token |
| `attention_mask` | 标记主序列中哪些位置有效 |
| `modal_type_ids` | 标记有效位置属于 text、image 还是 audio |
| `pixel_values` | 只 stack batch 内真实存在的图像 |
| `image_batch_indices` | 每张图对应原 batch 哪一行 |
| `audio_features` | 音频 mel 截断并 pad 到固定上限后的张量 |
| `audio_mask` | 每段音频中哪些帧是真实帧 |
| `audio_batch_indices` | 每段音频对应原 batch 哪一行 |

主序列负责“排座位”，side channel 负责保存真实内容。`IMAGE_TOKEN_ID` 和 `AUDIO_TOKEN_ID` 是占位符，后续 forward 会用 `modal_type_ids` 找到对应位置，再把 encoder 输出写进去。

## 3. 序列拼接：先按样本构造，再按 batch pad

每个样本按固定顺序拼接：

```text
[IMAGE_TOKEN_ID × image_num_patches]
[AUDIO_TOKEN_ID × min(T, max_audio_frames)]
[text_ids...]
```

没有图像就跳过 image 段，没有音频就跳过 audio 段。主序列只为真实保留的音频帧生成 audio token，音频 padding 帧不会出现在 `input_ids` 里。

推荐实现顺序：

1. 遍历 batch，先构造 `per_item_ids` 和 `per_item_modal`。
2. 计算 `S = max(len(ids) for ids in per_item_ids)`。
3. 创建 `(B, S)` 的 `input_ids`、`attention_mask`、`modal_type_ids`。
4. 把每行有效前缀填进去，padding 保持默认值。

举例：`image_num_patches=4`，`max_audio_frames=32`。一个 mixed batch 有四行：

| 行 | 样本 | 主序列长度 |
|---|---|---|
| 0 | 纯文本，3 个 token | 3 |
| 1 | 图像 + 2 个文本 token | 4 + 2 = 6 |
| 2 | 8 帧音频 + 1 个文本 token | 8 + 1 = 9 |
| 3 | 图像 + 12 帧音频 + 3 个文本 token | 4 + 12 + 3 = 19 |

最终 `input_ids.shape == (4, 19)`。短样本右侧 padding，`attention_mask` 在 padding 位置为 False，`modal_type_ids` 在 padding 位置保留 0。

## 4. 两个 mask：有效位置和模态来源

`attention_mask` 回答“这一格是否参与主序列计算”。它是 bool 张量，真实 image/audio/text placeholder 都是 True，padding 是 False。

`modal_type_ids` 回答“这一格的 embedding 来源是哪类”。L18 使用 0 表示 text 或 padding，1 表示 image，2 表示 audio。真实文本和 padding 的 modal 值都为 0，所以判断 padding 时必须同时看 `attention_mask`。

常见错误有三类：

- padding 位置 `attention_mask=True`，导致模型把 padding 当有效 token。
- image/audio 段的 `modal_type_ids` 没写对，forward 找不到要替换的位置。
- `ids` 和 `modal` 分开构造，二者长度不一致。

排查时先打印每行 `attention_mask.sum()`，再数 `modal_type_ids == 1` 和 `modal_type_ids == 2` 的数量，通常能快速定位 off-by-one。

## 5. 图像侧通道：真实图像不进入 `input_ids`

如果 batch 中第 1 行和第 3 行有图，`pixel_values.shape` 是 `(2, C, H, W)`，`image_batch_indices == [1, 3]`。它不需要补成 `(B, C, H, W)`。这样 forward 能按 `image_batch_indices[k]` 找回原 batch 行，再按 `modal_type_ids[row] == 1` 找到 placeholder 位置。

这个设计避免两类问题：

- 为纯文本样本制造空图，浪费 encoder 计算。
- 依赖 side tensor 的隐式顺序，排查时很难判断第几张图属于哪一行。

真实 Megatron 里同类概念会变成 `imgs`、`num_tiles`、`tokens` 和 `labels`。图片可能被切成多个 tile，token 序列里 image token 的数量必须和 tile 及 image embedding 数对应。

## 6. 音频侧通道：固定上限和真实帧 mask

音频长度差异比图像更大。L18 约定每段 `audio_mel` 形状是 `(T, 80)`，collator 做两件事：

1. `T_eff = min(T, max_audio_frames)`，主序列生成 `T_eff` 个 `AUDIO_TOKEN_ID`。
2. side tensor 截断到 `T_eff` 后 pad 到 `(max_audio_frames, 80)`，并写 `audio_mask[:T_eff] = True`。

如果 `T=10`，`max_audio_frames=64`，主序列里有 10 个 audio token，`audio_features` 这一行是 `(64, 80)`，`audio_mask` 前 10 个为 True。如果 `T=200`，主序列里有 64 个 audio token，`audio_mask` 全 True。

这里的代价很明确：固定上限便于 stack 和缓存 shape，但短音频会有 padding 存储，长音频会截断尾部。真实训练通常还会做长度分桶、chunking 或更复杂的 audio encoder mask。

## 7. 真实源码：Megatron Energon 的样本、batch 和恢复

Megatron 的多模态示例比 L18 patch 多出三类生产复杂度。

第一类是 sample encoder。`dataset_helpers.py` 里的 `TaskEncoder` 接收 Energon sample，做图片 transform、prompt/tokenize、image tag 处理、tile 数计算和训练标签构造。它还会检查 image token 数和 tile 数是否一致。

第二类是 batch builder。`TaskEncoder.batch` 会把多条 sample 的 `imgs` stack 起来，把 tokens 和 labels pad 到统一长度，并在 FP8 / context parallel 场景下继续补齐序列。这和 L18 patch 的 `pixel_values`、`input_ids`、`attention_mask` 是同一类问题，只是字段更贴近 Megatron 训练循环。

第三类是 dataloader state。`dataloader_provider.py` 只在指定 TP/PP rank 上创建 dataloader，并用 `get_savable_loader` 支持保存和恢复 worker 状态。对于大规模训练，这决定 resume 后是否重复读或漏读样本。L18 patch 不覆盖这个行为，但 smoke 和 outputs 模板要求你保留 artifact，方便后续 L19 和 Capstone 继续追踪。

## 8. Lab 验收和排查边界

patch-test 的 7 个测试覆盖这些不变量：

| 测试 | 验收点 |
|---|---|
| `test_text_only_batch` | 无图无音时 side tensors 为 `None`，modal 全 0 |
| `test_with_image_batch` | image placeholder 数量和 `pixel_values` 对齐 |
| `test_with_audio_batch` | audio token 数、`audio_features` 和 `audio_mask` 对齐 |
| `test_mixed_modality_batch` | 四种样本组合能混在一个 batch |
| `test_padding_blocks_attention` | padding 不参与 attention，modal 保持 0 |
| `test_audio_truncation` | 超长音频按上限截断 |
| `test_image_batch_indices_correct` | `pixel_values` 与原 batch 行号一致 |

patch 通过后，还要跑 smoke：

```bash
python labs/l17_megatron_multimodal_data/scripts/run_multimodal_data_lab.py --run-id l18_validation
```

这个脚本会生成 toy raw files、manifest、tar shard，并写入 manifest validation、shard inspection 和 loader smoke artifact。它验证的是数据集合和 shard 组织，不会替你验证 `multimodal_collate` 的全部语义；两者要分开看。

## 9. 小结

L18 的核心是一个清楚的 batch 合同：主序列保存 placeholder 和文本位置，mask 标记有效性和模态来源，side tensors 保存真实图像和音频，batch indices 把 side tensors 映射回原样本行。掌握这个合同后，读 LLaVA、Qwen2-VL、Qwen-Audio 或 Megatron Energon 的数据代码时，就能把复杂字段放回同一张图里。
