# L18 · Multimodal Data Collator

L18 讲多模态训练数据进入模型前的 batch 合同：文本 token、图像 tensor 和音频 mel 帧怎样被整理成同一个 `input_ids`、`attention_mask`、`modal_type_ids` 和 side tensors。

## 学习路线

1. 读 [system_map.md](system_map.md)：先确认 L18 在数据与多模态主线中的位置。
2. 读 [lecture.md](lecture.md)：从 batch 合同、占位 token、mask 和 side channel 讲到真实 dataloader。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、Megatron Energon 的主路径带读源码。
4. 做 quiz：检查 placeholder、mask、audio padding、image indices 和 dataloader 边界。
5. 做 patch：实现 `multimodal_collate` 的最小行为合同。
6. 跑 smoke：构建 toy manifest、tar shard，并检查 loader artifact。
7. 填写 [outputs/data_pipeline_template.md](outputs/data_pipeline_template.md)，保留复盘证据。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Data pipeline / Multimodal training input |
| 解决的问题 | 变长 image + audio + text 样本如何进入同一个 Transformer batch |
| 上游 | manifest、dataset、processor、WebDataset shard |
| 下游 | image/audio encoder、embedding replacement、LLM forward |
| patch 验收 | `input_ids`、mask、side tensors 和 batch indices 的最小合同 |

## 你会学到什么

- 解释 `input_ids`、`attention_mask`、`modal_type_ids` 的不同职责。
- 按 `[image tokens][audio tokens][text tokens]` 拼接变长样本，并 pad 到 batch 内统一长度。
- 用 `pixel_values/image_batch_indices` 和 `audio_features/audio_mask/audio_batch_indices` 传递真实模态内容。
- 读懂 Megatron multimodal task encoder 如何把图片、tokens、labels 和 tiles 组织成 batch。
- 区分 collator、processor、dataloader worker、GPU encoder 的工程边界。

## Patch 闭环

```bash
cat labs/l17_megatron_multimodal_data/patch/task.md
$EDITOR labs/l17_megatron_multimodal_data/patch/starter/multimodal_collate.py
IMPL=reference make patch-test M=l17_megatron_multimodal_data
```

smoke：

```bash
python labs/l17_megatron_multimodal_data/scripts/run_multimodal_data_lab.py --run-id l18_smoke
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 定位 manifest、collator、shard 和 dataloader 问题 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、MiniInfra 和 Megatron 源码主路径 |
| [outputs/data_pipeline_template.md](outputs/data_pipeline_template.md) | 记录一次数据管线 smoke 或训练排查 |

## 下一讲

L19 会继续进入数据工程：MinHash 去重、WebDataset pipeline、detshuffle 和 shard resume。
