# L41 Patch · MM-Tiny-Omni Bronze Components

## 你要交付什么

实现三个 Bronze 组件：

1. `MultimodalProjector`
   - 输入图像 feature：`(B, num_image_tokens, vit_dim)`
   - 输入语音 feature：`(B, T, whisper_dim)`
   - 输出 LLM hidden dim：`image_emb` / `audio_emb`
   - 图像路径需要加 learnable `image_pos_emb`

2. `wer_score(hypothesis, reference)`
   - 按空格切 word
   - 用 word-level Levenshtein edit distance
   - 返回 `distance / max(1, len(reference_words))`

3. `clip_score(image_embed, text_embed)`
   - 对两个向量做 L2 normalize
   - 返回 cosine similarity

## 不变量

1. image-only 时 `audio_emb is None`。
2. audio-only 时 `image_emb is None`。
3. 双模态输入时两个 key 都存在并保持 batch 维。
4. WER 完全匹配为 0。
5. WER 的分母是 reference word 数。
6. 相同 CLIP embedding 得分为 1。
7. 正交 embedding 得分为 0。

## 怎么验证

```bash
make patch-test M=l35_multimodal_capstone
IMPL=reference make patch-test M=l35_multimodal_capstone
```

聚合器格式验证：

```bash
python labs/l35_multimodal_capstone/scripts/run_capstone_aggregator.py --run-id l41_validation
```

## 写完之后你能做什么

- 解释多模态 projector 如何把 CLIP/Whisper feature 接入 LLM hidden space。
- 判断 WER 和 CLIP score 的数值边界。
- 给 Silver/Gold 阶段补上训练、服务、RL 和 final artifacts 证据。
