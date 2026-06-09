# L18 Debug Checklist

## 1. 固定现场

- 记录命令、git commit、Python/PyTorch 版本、数据路径、batch size、`image_num_patches`、`max_audio_frames`。
- 保存 patch-test 输出、smoke run 目录、manifest validation、shard inspection、loader smoke artifact。
- 如果来自真实训练，补充 dataloader 配置：`num_workers`、`prefetch_factor`、shuffle、resume cursor、rank/world size。

## 2. 先按层定位

| 层 | 先看什么 | 常见结论 |
|---|---|---|
| manifest | task、files、missing list | 样本集合本身不可读 |
| dataset/processor | 单样本 `text_ids`、`image`、`audio_mel` | 单样本字段缺失或 shape 异常 |
| collator | `input_ids`、mask、side tensors、indices | batch 合同不一致 |
| shard/loader | shard members、sample keys、worker state | 重复读、漏读或 shard 键不匹配 |
| forward | placeholder 数、encoder 输出数、replacement mask | embedding 替换位置或数量错误 |

## 3. Collator 排查顺序

1. 打印每个样本的 `len(text_ids)`、是否有图、`audio_mel.shape[0]`。
2. 计算每行期望长度：`image_num_patches + min(T, max_audio_frames) + len(text_ids)`。
3. 检查 `attention_mask.sum(dim=1)` 是否等于每行期望长度。
4. 检查 `modal_type_ids == 1` 的数量是否等于有图样本数乘 `image_num_patches`。
5. 检查 `audio_mask.sum(dim=1)` 是否等于每段音频的截断长度。
6. 检查 `pixel_values[k]` 和 `batch[image_batch_indices[k]]["image"]` 是否对应。

## 4. 常见错误

- padding 位置的 `attention_mask` 仍为 True。
- padding 位置继承了 image/audio modal id。
- `pixel_values` 用空图补到 `B`，但没有清楚的 batch index 语义。
- 为音频 padding 帧生成 `AUDIO_TOKEN_ID`，导致序列长度和真实帧数不一致。
- 在 DataLoader worker 里调用 GPU encoder，造成 CUDA context 和 worker 生命周期混乱。

## 5. 结束条件

- 问题能被最小 batch 或 smoke 命令复现。
- 关键张量 shape、mask sum、batch indices 和 artifact 已记录。
- 能指出故障发生在 manifest、dataset、collator、loader 或 forward 哪一层。
- 结论写入 `data_pipeline_template.md`，并包含下一步动作。
