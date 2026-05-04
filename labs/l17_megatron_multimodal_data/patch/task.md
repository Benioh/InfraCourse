# L06 Patch · 多模态 Collator（图像 + 语音 + 文本）

> ⭐ 这是本课的多模态主战场，对应你 Capstone (MM-Tiny-Omni) 的数据管道核心。

## 你要交付什么

实现一个能把**变长 image + audio + text**统一打包成 batch tensor 的 collator。

```python
def multimodal_collate(
    batch: List[Dict[str, Any]],
    image_num_patches: int = 16,
    max_audio_frames: int = 64,
) -> Dict[str, torch.Tensor]:
    """每个 batch 元素都是 dict：
        {
            "text_ids": List[int],              # 文本 token id（必须）
            "image": Tensor(C, H, W) or None,   # 单张图（可选）
            "audio_mel": Tensor(T, 80) or None, # mel-spec 帧序列（可选, T 变长）
        }

    返回 dict，所有 batch 维度一致 (B = len(batch))：
        input_ids:       (B, S) long
        attention_mask:  (B, S) bool
        modal_type_ids:  (B, S) int8 ∈ {0=text, 1=image, 2=audio}
        pixel_values:    (B', C, H, W) where B' = #items with image  (None 如全无图)
        image_batch_indices: (B',) long, 对应原 batch idx
        audio_features:  (B'', max_audio_frames, 80)      (None 如全无音频)
        audio_mask:      (B'', max_audio_frames) bool
        audio_batch_indices: (B'',) long
    """
```

**禁止** 用 `transformers.AutoProcessor` / `torchvision.transforms` 偷懒。
**允许** `torch.cat` / `F.pad` / 普通 list 操作。

补丁规模目标：80–150 行。

## 序列拼接约定

每个 batch 元素的 sequence 拼接顺序：
```
[image_token × image_num_patches]   (如果有图)
[audio_token × audio_T]             (如果有音频)
[text_ids...]                       (永远存在)
```

token id 约定：
- `IMAGE_TOKEN_ID = 1` （placeholder，会被 image encoder 输出的 embedding 替换）
- `AUDIO_TOKEN_ID = 2`
- `PAD_TOKEN_ID = 0`
- 真实文本 token id 都 ≥ 3

modal_type_ids 让 forward 知道哪一段是图、哪一段是音、哪一段是文。

## 不变量

1. `attention_mask[i, j] = True` 当且仅当位置 j 是 batch_i 真实 token（不是 padding）。
2. `modal_type_ids` 在 image_token 位置 = 1；audio_token 位置 = 2；其它（包含 padding）= 0。
3. `pixel_values` 顺序与 `image_batch_indices` 一一对应（不一定是 batch 顺序）。
4. 音频被 pad 到 `max_audio_frames`，但**真实长度** ≤ `max_audio_frames`（超长直接截断）。
5. 全无图的 batch：`pixel_values = None`，`image_batch_indices = None`。

## 怎么验证

```bash
make patch-test M=l17_megatron_multimodal_data
```

7 个测试，CPU 友好：

| 测试 | 验证 |
|---|---|
| `test_text_only_batch` | 没图没音时输出形状正确，pixel_values/audio_features 为 None |
| `test_with_image_batch` | 一个图，input_ids 前 16 个是 IMAGE_TOKEN_ID |
| `test_with_audio_batch` | 一个变长音频，audio_mask 标记真实位置 |
| `test_mixed_modality_batch` | 同 batch 4 个样本：纯文本、文本+图、文本+音、文本+图+音 |
| `test_padding_blocks_attention` | padding 位置 attention_mask=False，modal_type_ids=0 |
| `test_audio_truncation` | 超长音频被截断到 max_audio_frames |
| `test_image_batch_indices_correct` | pixel_values 与 image_batch_indices 顺序匹配 |

## 写完之后你能做什么

- 直接用在 Capstone Stage A：把 COCO 图片 + LibriSpeech 语音 + Qwen tokenizer 文本组成 batch。
- 解释 LLaVA / Qwen2-VL / Qwen-Audio 的 image_token 占位 trick。
- 在 Stage B 推理时复用：处理 `(image_url, audio_url, text)` 多模态请求。
