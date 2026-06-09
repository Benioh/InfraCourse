# L18 Patch · Multimodal Collator

本 patch 只验收一件事：把已经加载好的 `text_ids`、`image`、`audio_mel`
打包成模型 forward 可以消费的 batch 字典。它不负责下载数据、图片 resize、
音频解码、tokenizer 或 GPU encoder。

## 交付函数

在 `patch/starter/multimodal_collate.py` 中实现：

```python
def multimodal_collate(
    batch: List[Dict[str, Any]],
    image_num_patches: int = 16,
    max_audio_frames: int = 64,
) -> Dict[str, torch.Tensor]:
    ...
```

每个输入样本是一个 dict：

```python
{
    "text_ids": List[int],              # 必需，真实文本 token id
    "image": Tensor(C, H, W) or None,   # 可选，已经变成 tensor 的单张图
    "audio_mel": Tensor(T, 80) or None, # 可选，已经提取好的 mel 帧
}
```

返回字段：

| 字段 | 形状 | 语义 |
|---|---|---|
| `input_ids` | `(B, S)` long | 文本 token 加 image/audio placeholder 后的序列 |
| `attention_mask` | `(B, S)` bool | True 表示真实位置，False 表示 padding |
| `modal_type_ids` | `(B, S)` int8 | 0=text/pad，1=image，2=audio |
| `pixel_values` | `(B_img, C, H, W)` 或 `None` | batch 中真实存在的图像 |
| `image_batch_indices` | `(B_img,)` 或 `None` | 每张图来自原 batch 的哪一行 |
| `audio_features` | `(B_audio, max_audio_frames, 80)` 或 `None` | 截断并 pad 后的音频特征 |
| `audio_mask` | `(B_audio, max_audio_frames)` 或 `None` | True 表示真实音频帧 |
| `audio_batch_indices` | `(B_audio,)` 或 `None` | 每段音频来自原 batch 的哪一行 |

## 序列拼接约定

每个样本的主序列按固定顺序拼接：

```text
[IMAGE_TOKEN_ID × image_num_patches]  # 有图时出现
[AUDIO_TOKEN_ID × min(T, max_audio_frames)]  # 有音频时出现
[text_ids...]
```

保留 token id：

| token | id |
|---|---|
| `PAD_TOKEN_ID` | 0 |
| `IMAGE_TOKEN_ID` | 1 |
| `AUDIO_TOKEN_ID` | 2 |

真实图像和音频内容不放进 `input_ids`。`input_ids` 里的 image/audio token
只是占位，后续 forward 会根据 `modal_type_ids` 找到这些位置，再用视觉或音频
encoder 的输出替换 embedding。

## 必守不变量

1. `attention_mask[i, j]` 只在真实 token 位置为 True，padding 位置为 False。
2. `modal_type_ids` 在 image placeholder 位置为 1，在 audio placeholder 位置为 2，文本和 padding 为 0。
3. `pixel_values[k]` 必须来自 `batch[image_batch_indices[k]]["image"]`。
4. `audio_features[k]` 必须来自 `batch[audio_batch_indices[k]]["audio_mel"]` 的截断/pad 结果。
5. 音频主序列长度只按真实保留帧数计算，不能为音频 padding 帧额外生成 `AUDIO_TOKEN_ID`。
6. batch 内没有图像时，`pixel_values` 和 `image_batch_indices` 都返回 `None`；没有音频时同理。

## 推荐实现顺序

1. 遍历 batch，先为每个样本构造 `ids` 和 `modal` 两个等长 list。
2. 计算 `S = max(len(ids) for ids in per_item_ids)`，创建 `(B, S)` 的 `input_ids`、`attention_mask` 和 `modal_type_ids`。
3. 再独立收集图像侧通道：`pixel_values` 与 `image_batch_indices`。
4. 最后收集音频侧通道：每段 mel 截断到 `max_audio_frames`，pad 到固定长度，并写 `audio_mask`。

## 验证

```bash
IMPL=reference make patch-test M=l17_megatron_multimodal_data
```

7 个测试覆盖：纯文本 batch、图像 placeholder、音频 mask、混合模态、padding、
音频截断和 `image_batch_indices` 顺序。
