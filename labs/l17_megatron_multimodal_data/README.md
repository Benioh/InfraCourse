# L06 · 多模态 Collator（图像 + 语音 + 文本）★ 核心

> 本关只做一件事：**实现一个能把变长 image + audio + text 打包成 batch tensor 的 collator**——直接是 LLaVA / Qwen2-VL / Qwen-Audio 的数据管道核心。

写完这关，Capstone Stage A 你能直接复用这个 collator 拼装 COCO 图 + LibriSpeech 语音 + Qwen tokenizer 的 batch。

## 闭环

```bash
cat labs/l17_megatron_multimodal_data/patch/task.md
$EDITOR labs/l17_megatron_multimodal_data/patch/starter/multimodal_collate.py
make patch-test M=l17_megatron_multimodal_data   # 7 个测试，CPU OK
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_text_only_batch` | 没图没音时形状/值正确，pixel_values=None |
| `test_with_image_batch` | 前 N 个 token 是 IMAGE_TOKEN_ID，modal=1 |
| `test_with_audio_batch` | audio_mask 正确标记真实位置 |
| `test_mixed_modality_batch` | 4 种组合（纯文本 / +图 / +音 / 三模态）混合 |
| `test_padding_blocks_attention` | padding 位置 attention_mask=False |
| `test_audio_truncation` | 超长音频被截断到 max_audio_frames |
| `test_image_batch_indices_correct` | pixel_values 顺序与 batch_indices 一致 |

## 卡住怎么办

1. 重读 task.md 的"序列拼接约定"那段。
2. `make patch-hint M=l17_megatron_multimodal_data`。
3. `make patch-show-solution M=l17_megatron_multimodal_data`。

## 进入下一关

`make patch-test` 全绿后，继续做源码理解口试。下一关 [L06.3 数据工程](../l18_data_engineering/README.md) 让你写 minhash 去重。
